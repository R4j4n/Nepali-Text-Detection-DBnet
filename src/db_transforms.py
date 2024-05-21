import imgaug
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


class Thresh:
    def __init__(self, polygon, canvas, mask, shrink_ratio=0.4):
        self.polygon = polygon
        self.canvas = canvas
        self.mask = mask
        self.shrink_ratio = shrink_ratio

    def draw_thresh_map(self):
        polygon = np.array(self.polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (
                1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(self.mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width),
            (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1),
            (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width),
                                dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        xmin_valid = min(max(0, xmin), self.canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), self.canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), self.canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), self.canvas.shape[0] - 1)

        # canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
        #     1 - distance_map[ymin_valid - ymin:ymax_valid - ymin + 1,  # add 1
        #                      xmin_valid - xmin:xmax_valid - xmin + 1  # add 1
        #                      ],
        #     canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        self.canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            self.canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def compute_distance(self, xs, ys, point_1, point_2):
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] -
                                    point_2[0]) + np.square(point_1[1] -
                                                            point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin /
                         square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1,
                                            square_distance_2))[cosin < 0]
        return result


class Transform:
    def __init__(self, aug, image, anns):
        self.aug = aug
        self.image = image
        self.anns = anns

    def transform(self):
        image_shape = self.image.shape
        image = self.aug.augment_image(self.image)
        new_anns = []
        for ann in self.anns:
            keypoints = [imgaug.Keypoint(p[0], p[1]) for p in ann['poly']]
            keypoints = self.aug.augment_keypoints(
                [imgaug.KeypointsOnImage(keypoints,
                                         shape=image_shape)])[0].keypoints
            poly = [(min(max(0, p.x),
                         image.shape[1] - 1), min(max(0, p.y), image.shape[0] - 1))
                    for p in keypoints]
            new_ann = {'poly': poly, 'text': ann['text']}
            new_anns.append(new_ann)
        return image, new_anns


class Region:
    def __init__(self, axis):
        self.axis = axis

    def split_regions(self):
        regions = []
        min_axis_index = 0
        for i in range(1, self.axis.shape[0]):
            if self.axis[i] != self.axis[i - 1] + 1:
                region = self.axis[min_axis_index:i]
                min_axis_index = i
                regions.append(region)
        return regions

    def random_select(self):
        xx = np.random.choice(self.axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        return xmin, xmax


def region_wise_random_select(regions):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


class Crop:
    def __init__(self, image, anns, max_tries=10, min_crop_side_ratio=0.1):
        self.image = image
        self.anns = anns
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio


    def crop(self):
        h, w, _ = self.image.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for ann in self.anns:
            points = np.round(ann['poly'], decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return self.image, self.anns

        h_regions = Region(h_axis).split_regions
        w_regions = Region(w_axis).split_regions

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = region_wise_random_select(w_regions)
            else:
                xmin, xmax = Region(w_axis).random_select
            if len(h_regions) > 1:
                ymin, ymax = region_wise_random_select(h_regions)
            else:
                ymin, ymax = Region(h_axis).random_select

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            new_anns = []
            for ann in self.anns:
                poly = np.array(ann['poly'])
                if not (poly[:, 0].min() > xmax or poly[:, 0].max() < xmin
                        or poly[:, 1].min() > ymax or poly[:, 1].max() < ymin):
                    poly[:, 0] -= xmin
                    poly[:, 0] = np.clip(poly[:, 0], 0., (xmax - xmin - 1) * 1.)
                    poly[:, 1] -= ymin
                    poly[:, 1] = np.clip(poly[:, 1], 0., (ymax - ymin - 1) * 1.)
                    new_ann = {'poly': poly.tolist(), 'text': ann['text']}
                    new_anns.append(new_ann)

            if len(new_anns) > 0:
                return self.image[ymin:ymax, xmin:xmax], new_anns

        return self.image, self.anns

class Resize:
    def __init__(self, size, image, anns):
        self.size = size
        self.image = image
        self.anns = anns

    def resize(self):
        h, w, c = self.image.shape
        scale_w = self.size / w
        scale_h = self.size / h
        scale = min(scale_w, scale_h)
        h = int(h * scale)
        w = int(w * scale)
        padimg = np.zeros((self.size, self.size, c), self.image.dtype)
        padimg[:h, :w] = cv2.resize(self.image, (w, h))
        new_anns = []
        for ann in self.anns:
            poly = np.array(ann['poly']).astype(np.float64)
            poly *= scale
            new_ann = {'poly': poly.tolist(), 'text': ann['text']}
            new_anns.append(new_ann)
        return padimg, new_anns
