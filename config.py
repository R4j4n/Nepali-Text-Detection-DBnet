import os
from pathlib import Path
from easydict import EasyDict as edict

root_pth = str(Path(__file__).parent)

class Config:
    def __init__(self):

        self.c = edict()

        # Meta settings
        self.c.meta = edict()
        self.c.meta.root_dir = root_pth
        self.c.meta.round_number = 8
        self.c.meta.debug = False
        self.c.meta.device = "cuda"
        self.c.meta.test_device = "cpu"

        # Hyperparameters
        self.c.hps = edict()
        self.c.hps.batch_size = 4
        self.c.hps.test_batch_size = 1  # must be set to 1 to evaluate metric
        self.c.hps.img_size = 640
        self.c.hps.no_epochs = 100
        self.c.hps.warmup_epochs = 10
        self.c.hps.no_classes = 2
        self.c.hps.log_iter = 50

        # Learning rates
        self.c.lrs = edict()
        self.c.lrs.mode = "reduce"  # reduce / poly
        self.c.lrs.warmup_iters = 10
        self.c.lrs.factor = 0.2
        self.c.lrs.patience = 4

        # Augmentation settings
        self.c.augmentation = edict()

        # Callbacks settings
        self.c.callbacks = edict()

        # Data settings
        self.c.data = edict()
        self.c.data.icdar2015 = edict()
        self.c.data.icdar2015.train_dir = Path(root_pth).joinpath(
            "Nepali_Text_Detection_Dataset/train/images/"
        )
        self.c.data.icdar2015.test_dir = Path(root_pth).joinpath(
            "Nepali_Text_Detection_Dataset/test/images/"
        )
        self.c.data.icdar2015.train_gt_dir = Path(root_pth).joinpath(
            "Nepali_Text_Detection_Dataset/train/gt"
        )
        self.c.data.icdar2015.test_gt_dir = Path(root_pth).joinpath(
            "Nepali_Text_Detection_Dataset/test/gt/"
        )
        self.c.data.icdar2015.ignore_tags = ["###"]


        # Dataset settings
        self.c.dataset = edict()
        self.c.dataset.name = "icdar2015"
        self.c.dataset.return_dict = True

        # Logging settings
        self.c.logging = edict()
        self.c.logging.logger_file = "train.log"

        # Loss settings
        self.c.loss = edict()

        # Model settings
        self.c.model = edict()
        self.c.model.finetune_cp_path = ""
        self.c.model.best_cp_path = "models/nepali/nepali_td_best.pth"
        self.c.model.last_cp_path = "models/nepali/nepali_td.pth"
        self.c.model.best_hmean_cp_path = "models/nepali/nepali_td_best_hmean.pth"

        # Optimizer settings
        self.c.optimizer = edict()
        self.c.optimizer.type = "adam"
        self.c.optimizer.lr = 0.005
        self.c.optimizer.lr_finetune = 0.001
        self.c.optimizer.weight_decay = 0.0
        self.c.optimizer.reduction = "mean"
        self.c.optimizer.alpha = 1
        self.c.optimizer.beta = 10
        self.c.optimizer.negative_ratio = 3
        self.c.optimizer.amsgrad = False

        # Metric settings
        self.c.metric = edict()
        self.c.metric.thred_text_score = 0.25
        self.c.metric.prob_threshold = 0.50
        self.c.metric.unclip_ratio = 1.50
        self.c.metric.is_output_polygon = True

        # Private settings
        self.c.private = edict()

        # Scheduler settings
        self.c.scheduler = edict()

        # Trainer settings
        self.c.trainer = edict()

        # Training settings
        self.c.training = edict()

    def config(self):
        cfg = self.c
        return cfg
