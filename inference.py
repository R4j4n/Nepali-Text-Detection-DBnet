import os
import glob
import random
from typing import Any
from pathlib import Path

import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))

from src.models import DBTextModel
from src.utils import (
    read_img,
    test_preprocess,
    visualize_heatmap,
    visualize_polygon,
    str_to_bool,
)
from src.postprocess import SegDetectorRepresenter

H, W = 14, 12


class CFG:
    model_path = "/mnt/c/Users/rjn/Documents/Dbnet/models/nepali/nepali_td_best.pth"
    device = "cuda"

    thresh = 0.25
    box_thresh = 0.5
    unclip_ratio = 1.5
    prob_thred = 0.5
    alpha = 0.6


class Inference:

    def __init__(self, args) -> None:

        self.args = args
        self.model = self.load_model(args)
        self.model.eval()

    @staticmethod
    def load_model(args):

        assert os.path.exists(args.model_path)
        dbnet = DBTextModel().to(args.device)
        dbnet.load_state_dict(torch.load(args.model_path, map_location=args.device))
        return dbnet

    @staticmethod
    def minmax_scaler_img(img):
        img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
            "uint8"
        )  # noqa
        return img

    def get_heatmap(self, tmp_img, tmp_pred, config):
        pred_prob = tmp_pred[0]
        pred_prob[pred_prob <= config.prob_thred] = 0
        pred_prob[pred_prob > config.prob_thred] = 1

        np_img = self.minmax_scaler_img(tmp_img[0].cpu().numpy().transpose((1, 2, 0)))

        fig_width, fig_height = H, W
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(np_img)
        ax.imshow(pred_prob, cmap="jet", alpha=config.alpha)

        return fig

    @staticmethod
    def draw_bbox(img, result, color=(255, 0, 0), thickness=3):
        """
        :input: RGB img
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        img = img.copy()
        for point in result:
            point = point.astype(int)
            cv2.polylines(img, [point], True, color, thickness)
        return img

    def visualize_polygon(self, args, img_fn, origin_info, batch, preds, poly=False):
        img_origin, h_origin, w_origin = origin_info

        seg_obj = SegDetectorRepresenter(
            thresh=args.thresh,
            box_thresh=args.box_thresh,
            unclip_ratio=args.unclip_ratio,
        )
        box_list, score_list = seg_obj(batch, preds, is_output_polygon=poly)

        box_list, score_list = box_list[0], score_list[0]

        if len(box_list) > 0:
            if poly:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []

        # tmp_img = self.draw_bbox(img_origin, np.array(box_list))
        tmp_img = self.draw_bbox(img_origin, np.array(box_list, dtype=object))

        tmp_pred = cv2.resize(preds[0, 0, :, :].cpu().numpy(), (w_origin, h_origin))

        fig_width, fig_height = H, W
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(tmp_img)
        ax.imshow(tmp_pred, cmap="inferno", alpha=args.alpha)
        return fig, box_list, score_list

    def __call__(self, img_path, poly_only=False) -> Any:

        assert os.path.exists(img_path)
        img_fn = img_path.split("/")[-1]
        # read image
        img_origin, h_origin, w_origin = read_img(img_path)

        # resize and convert to tensor
        tmp_img = test_preprocess(img_origin, to_tensor=True, pad=False).to(
            self.args.device
        )

        torch.cuda.empty_cache()

        with torch.no_grad():
            preds = self.model(tmp_img)

        # get heatmap image
        heat_map = self.get_heatmap(
            tmp_img=tmp_img, tmp_pred=preds.to("cpu")[0].numpy(), config=self.args
        )

        # get polygons
        batch = {"shape": [(h_origin, w_origin)]}
        poly, box_list, score_list = self.visualize_polygon(
            args=self.args,
            img_fn=img_fn,
            origin_info=(img_origin, h_origin, w_origin),
            batch=batch,
            preds=preds,
            poly=poly_only,
        )

        return heat_map, poly, box_list, score_list


# if __name__ == "__main__":

class Infer:
    def infer():
        config = CFG()

        paths = glob.glob("sample/*.*")

        tdd = Inference(config)

        for pth in paths:
            # polygon plot
            heat_map, poly, box, score = tdd(img_path=pth, poly_only=True)
            save_pth = str(Path(pth).parent.joinpath("output"))

            heat_map.savefig(
                f"{save_pth}/{str(Path(pth).stem)}_heat_map.png"
            )
            poly.savefig(f"{save_pth}/{str(Path(pth).stem)}_poly.png")

            # rectangle plot
            _, poly, _, _ = tdd(img_path=pth, poly_only=False)
            poly.savefig(f"{save_pth}/{str(Path(pth).stem)}_rect.png")
