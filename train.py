import os
import gc
import cv2
import time
import random
import warnings
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from src.losses import DBLoss
from src.models import DBTextModel
from src.text_metrics import cal_text_score, RunningScore, QuadMetric
from src.utils import (
    setup_determinism,
    setup_logger,
    dict_to_device,
    visualize_tfb,
)
from src.postprocess import SegDetectorRepresenter
from src.my_data_loader import ICDAR2015DatasetIter

from config import Config

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def arg_parser():
    parser = argparse.ArgumentParser(description='NHD Trainer')
    parser.add_argument('--train', help='Evaluate', default='True', action='store_true')

    args = parser.parse_args()

    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}
    main()


class Dataset:
    def __init__(self, cfg):
        self.cfg = cfg

    def Load(self):
        dataset_name = self.cfg.dataset.name
        ignore_tags = self.cfg.data[dataset_name].ignore_tags
        train_dir = self.cfg.data[dataset_name].train_dir
        test_dir = self.cfg.data[dataset_name].test_dir
        train_gt_dir = self.cfg.data[dataset_name].train_gt_dir
        test_gt_dir = self.cfg.data[dataset_name].test_gt_dir
        return dataset_name, ignore_tags, train_dir, test_dir, train_gt_dir, test_gt_dir


class ICDAR:
    def __init__(self, cfg, ignore_tags, train_dir, test_dir, train_gt_dir, test_gt_dir):
        self.ignore_tags = ignore_tags
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_gt_dir = train_gt_dir
        self.test_gt_dir = test_gt_dir
        self.cfg = cfg

    def icdarloader(self):
        TextDatasetIter = ICDAR2015DatasetIter

        train_iter = TextDatasetIter(
            self.train_dir,
            self.train_gt_dir,
            self.ignore_tags,
            image_size=self.cfg.hps.img_size,
            is_training=True,
            debug=False,
        )
        test_iter = TextDatasetIter(
            self.test_dir,
            self.test_gt_dir,
            self.ignore_tags,
            image_size=self.cfg.hps.img_size,
            is_training=False,
            debug=False,
        )

        train_loader = DataLoader(
            dataset=train_iter, batch_size=self.cfg.hps.batch_size, shuffle=True, num_workers=1
        )
        test_loader = DataLoader(
            dataset=test_iter,
            batch_size=self.cfg.hps.test_batch_size,
            shuffle=False,
            num_workers=0,
        )
        return train_loader, test_loader


class Train:
    def __init__(self, cfg, train_loader, test_loader, dataset_name, ignore_tags, train_dir, test_dir, train_gt_dir,
                 test_gt_dir):
        self.cfg = cfg
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.dataset_name = dataset_name
        self.ignore_tags = ignore_tags
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_gt_dir = train_gt_dir
        self.test_gt_dir = test_gt_dir

    def main_trainer(self):
        # setup logger
        logger = setup_logger(os.path.join(self.cfg.meta.root_dir, self.cfg.logging.logger_file))

        # setup log folder
        log_dir_path = os.path.join(self.cfg.meta.root_dir, "logs")
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        tfb_log_dir = os.path.join(log_dir_path, str(int(time.time())))
        logger.info(tfb_log_dir)
        if not os.path.exists(tfb_log_dir):
            os.makedirs(tfb_log_dir)
        tfb_writer = SummaryWriter(tfb_log_dir)

        # setup save folder
        save_pth = str(Path(self.cfg.meta.root_dir).joinpath(self.cfg.model.last_cp_path).parent)

        Path(save_pth).mkdir(parents=True, exist_ok=True)

        device = self.cfg.meta.device
        logger.info(device)
        dbnet = DBTextModel().to(device)

        lr_optim = self.cfg.optimizer.lr

        # # load best cp
        # if cfg.model.finetune_cp_path:
        #     cp_path = os.path.join(cfg.meta.root_dir, cfg.model.finetune_cp_path)
        #     if os.path.exists(cp_path) and cp_path.endswith(".pth"):
        #         lr_optim = cfg.optimizer.lr_finetune
        #         logger.info("Loading best checkpoint: {}".format(cp_path))
        #         dbnet.load_state_dict(torch.load(cp_path, map_location=device))

        dbnet.train()

        criterion = DBLoss(
            alpha=self.cfg.optimizer.alpha,
            beta=self.cfg.optimizer.beta,
            negative_ratio=self.cfg.optimizer.negative_ratio,
            reduction=self.cfg.optimizer.reduction,
        ).to(device)

        db_optimizer = torch_optim.Adam(
            dbnet.parameters(),
            lr=lr_optim,
            weight_decay=self.cfg.optimizer.weight_decay,
            amsgrad=self.cfg.optimizer.amsgrad,
        )

        # setup model checkpoint
        best_test_loss = np.inf
        best_train_loss = np.inf
        best_hmean = 0

        db_scheduler = None
        lrs_mode = self.cfg.lrs.mode
        logger.info("Learning rate scheduler: {}".format(lrs_mode))

        db_scheduler = torch_optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=db_optimizer,
            mode="min",
            factor=self.cfg.lrs.factor,
            patience=self.cfg.lrs.patience,
            verbose=True,
        )

        # get data loaders
        dataset_name = self.cfg.dataset.name
        logger.info("Dataset name: {}".format(dataset_name))
        logger.info("Ignore tags: {}".format(self.cfg.data[dataset_name].ignore_tags))

        # train model
        logger.info("Start training!")
        torch.cuda.empty_cache()
        gc.collect()
        global_steps = 0
        for epoch in range(self.cfg.hps.no_epochs):

            # TRAINING
            dbnet.train()
            train_loss = 0
            running_metric_text = RunningScore(self.cfg.hps.no_classes)
            for batch_index, batch in enumerate(self.train_loader):
                lr = db_optimizer.param_groups[0]["lr"]
                global_steps += 1

                batch = dict_to_device(batch, device=device)
                preds = dbnet(batch["img"])
                assert preds.size(1) == 3

                _batch = torch.stack(
                    [
                        batch["prob_map"],
                        batch["supervision_mask"],
                        batch["thresh_map"],
                        batch["text_area_map"],
                    ]
                )
                prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss = (
                    criterion(preds, _batch)  # noqa
                )
                db_optimizer.zero_grad()

                total_loss.backward()
                db_optimizer.step()
                if lrs_mode == "poly":
                    db_scheduler.step()

                score_shrink_map = cal_text_score(
                    preds[:, 0, :, :],
                    batch["prob_map"],
                    batch["supervision_mask"],
                    running_metric_text,
                    thresh=self.cfg.metric.thred_text_score,
                )

                train_loss += total_loss
                acc = score_shrink_map["Mean Acc"]
                iou_shrink_map = score_shrink_map["Mean IoU"]

                # tf-board
                tfb_writer.add_scalar("TRAIN/LOSS/total_loss", total_loss, global_steps)
                tfb_writer.add_scalar("TRAIN/LOSS/loss", prob_threshold_loss, global_steps)
                tfb_writer.add_scalar("TRAIN/LOSS/prob_loss", prob_loss, global_steps)
                tfb_writer.add_scalar(
                    "TRAIN/LOSS/threshold_loss", threshold_loss, global_steps
                )
                tfb_writer.add_scalar("TRAIN/LOSS/binary_loss", binary_loss, global_steps)
                tfb_writer.add_scalar("TRAIN/ACC_IOU/acc", acc, global_steps)
                tfb_writer.add_scalar(
                    "TRAIN/ACC_IOU/iou_shrink_map", iou_shrink_map, global_steps
                )
                tfb_writer.add_scalar("TRAIN/HPs/lr", lr, global_steps)

                if global_steps % self.cfg.hps.log_iter == 0:
                    logger.info(
                        "[{}-{}] - lr: {} - total_loss: {} - loss: {} - acc: {} - iou: {}".format(  # noqa
                            epoch + 1,
                            global_steps,
                            lr,
                            total_loss,
                            prob_threshold_loss,
                            acc,
                            iou_shrink_map,
                        )
                    )

            end_epoch_loss = train_loss / len(self.train_loader)
            logger.info("Train loss: {}".format(end_epoch_loss))
            gc.collect()

            # TFB IMGs
            # shuffle = True
            visualize_tfb(
                tfb_writer,
                batch["img"],
                preds,
                global_steps=global_steps,
                thresh=self.cfg.metric.thred_text_score,
                mode="TRAIN",
            )

            seg_obj = SegDetectorRepresenter(
                thresh=self.cfg.metric.thred_text_score,
                box_thresh=self.cfg.metric.prob_threshold,
                unclip_ratio=self.cfg.metric.unclip_ratio,
            )
            metric_cls = QuadMetric()

            # EVAL
            dbnet.eval()
            test_running_metric_text = RunningScore(self.cfg.hps.no_classes)
            test_loss = 0
            raw_metrics = []
            test_visualize_index = random.choice(range(len(self.test_loader)))
            for test_batch_index, test_batch in tqdm(
                    enumerate(self.test_loader), total=len(self.test_loader)
            ):

                with torch.no_grad():
                    test_batch = dict_to_device(test_batch, device)

                    test_preds = dbnet(test_batch["img"])
                    assert test_preds.size(1) == 2

                    _batch = torch.stack(
                        [
                            test_batch["prob_map"],
                            test_batch["supervision_mask"],
                            test_batch["thresh_map"],
                            test_batch["text_area_map"],
                        ]
                    )
                    test_total_loss = criterion(test_preds, _batch)
                    test_loss += test_total_loss

                    # visualize predicted image with tfb
                    if test_batch_index == test_visualize_index:
                        visualize_tfb(
                            tfb_writer,
                            test_batch["img"],
                            test_preds,
                            global_steps=global_steps,
                            thresh=self.cfg.metric.thred_text_score,
                            mode="TEST",
                        )

                    test_score_shrink_map = cal_text_score(
                        test_preds[:, 0, :, :],
                        test_batch["prob_map"],
                        test_batch["supervision_mask"],
                        test_running_metric_text,
                        thresh=self.cfg.metric.thred_text_score,
                    )
                    test_acc = test_score_shrink_map["Mean Acc"]
                    test_iou_shrink_map = test_score_shrink_map["Mean IoU"]
                    tfb_writer.add_scalar(
                        "TEST/LOSS/val_loss", test_total_loss, global_steps
                    )
                    tfb_writer.add_scalar("TEST/ACC_IOU/val_acc", test_acc, global_steps)
                    tfb_writer.add_scalar(
                        "TEST/ACC_IOU/val_iou_shrink_map", test_iou_shrink_map, global_steps
                    )

                    # Cal P/R/Hmean
                    batch_shape = {"shape": [(self.cfg.hps.img_size, self.cfg.hps.img_size)]}
                    box_list, score_list = seg_obj(
                        batch_shape,
                        test_preds,
                        is_output_polygon=self.cfg.metric.is_output_polygon,
                    )
                    raw_metric = metric_cls.validate_measure(
                        test_batch, (box_list, score_list)
                    )
                    raw_metrics.append(raw_metric)
            metrics = metric_cls.gather_measure(raw_metrics)
            recall = metrics["recall"].avg
            precision = metrics["precision"].avg
            hmean = metrics["fmeasure"].avg

            if hmean >= best_hmean:
                best_hmean = hmean
                torch.save(
                    dbnet.state_dict(),
                    os.path.join(self.cfg.meta.root_dir, self.cfg.model.best_hmean_cp_path),
                )

            logger.info(
                "TEST/Recall: {} - TEST/Precision: {} - TEST/HMean: {}".format(
                    recall, precision, hmean
                )
            )
            tfb_writer.add_scalar("TEST/recall", recall, global_steps)
            tfb_writer.add_scalar("TEST/precision", precision, global_steps)
            tfb_writer.add_scalar("TEST/hmean", hmean, global_steps)

            test_loss = test_loss / len(self.test_loader)
            logger.info("[{}] - test_loss: {}".format(global_steps, test_loss))

            if test_loss <= best_test_loss and train_loss <= best_train_loss:
                best_test_loss = test_loss
                best_train_loss = train_loss
                torch.save(
                    dbnet.state_dict(),
                    os.path.join(self.cfg.meta.root_dir, self.cfg.model.best_cp_path),
                )

            if lrs_mode == "reduce":
                db_scheduler.step(test_loss)

            torch.cuda.empty_cache()
            gc.collect()

        logger.info("Training completed")
        torch.save(
            dbnet.state_dict(), os.path.join(self.cfg.meta.root_dir, self.cfg.model.last_cp_path)
        )
        logger.info("Saved model")


def main():
    cfg = Config.config
    dataset_name, ignore_tags, train_dir, test_dir, train_gt_dir, test_gt_dir = Dataset(cfg).Load
    train_loader, test_loader = ICDAR(cfg, ignore_tags, train_dir, test_dir, train_gt_dir, test_gt_dir).icdarloader
    Train(cfg, train_loader, test_loader, dataset_name, ignore_tags, train_dir, test_dir, train_gt_dir,
          test_gt_dir).main_trainer()


if __name__ == '__main__':
    main()
