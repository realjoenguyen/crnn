import os
import random
import numpy as np
import cv2
import torch
import config

class ToTensor(object):
    def __call__(self, sample):
        sample["img"] = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()
        sample["seq"] = torch.Tensor(sample["seq"]).int()
        return sample


class Resize(object):
    def __init__(self, size, data_augmen=False):
        self.size = size
        self.data_augmen = data_augmen

    def __call__(self, sample):
        img = sample["img"]
        assert img is not None

        # increase dataset size by applying random stretches to the images
        if self.data_augmen:
            stretch = (random.random() - 0.5)  # -0.5 .. +0.5
            wStretched = max(int(img.shape[1] * (1 + stretch)), 1)  # random width, but at least 1
            img = cv2.resize(img, (wStretched, img.shape[0]))  # stretch horizontally by factor 0.5 .. 1.5

        (wt, ht) = self.size
        (h, w, c) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1),
                   max(min(ht, int(h / f)), 1))  # scale according to f (result at least 1 and at most wt or ht)
        img = cv2.resize(img, newSize)
        target = np.ones([ht, wt, 3]) * 255

        if self.data_augmen:
            left_upper_h = random.randint(0, ht - newSize[1])
            left_upper_w = random.randint(0, wt - newSize[0])
            target[left_upper_h:newSize[1] + left_upper_h, left_upper_w:newSize[0] + left_upper_w, :] = img
        else:
            target[0:newSize[1], 0:newSize[0], :] = img

        # # normalize
        # (m, s) = cv2.meanStdDev(img)
        # m = int(m[0][0])
        # s = s[0][0]
        # img = img - m
        # # img = int(img / s) if s > 0 else img
        # img = np.rint(img / s) if s > 0 else img
        # sample["img"] = cv2.resize(sample["img"], self.size)
        # visualize
        # print ("print sample to", os.path.join(config.output_dir, sample["name"]))

        sample["img"] = target
        return sample


class Rotation(object):
    def __init__(self, angle=20, fill_value=0, p = 0.5):
        self.angle = angle
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h,w,_ = sample["img"].shape
        ang_rot = np.random.uniform(self.angle) - self.angle/2
        transform = cv2.getRotationMatrix2D((w/2, h/2), ang_rot, 1)
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample


class Translation(object):
    def __init__(self, fill_value=0, p = 0.5):
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h,w,_ = sample["img"].shape
        trans_range = [w / 10, h / 10]
        tr_x = trans_range[0]*np.random.uniform()-trans_range[0]/2
        tr_y = trans_range[1]*np.random.uniform()-trans_range[1]/2
        transform = np.float32([[1,0, tr_x], [0,1, tr_y]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample


class Scale(object):
    def __init__(self, scale=[0.5, 1.2], fill_value=0, p = 0.5):
        self.scale = scale
        self.fill_value = fill_value
        self.p = p

    def __call__(self, sample):
        if np.random.uniform(0.0, 1.0) < self.p or not sample["aug"]:
            return sample
        h, w, _ = sample["img"].shape
        scale = np.random.uniform(self.scale[0], self.scale[1])
        transform = np.float32([[scale, 0, 0],[0, scale, 0]])
        sample["img"] = cv2.warpAffine(sample["img"], transform, (w,h), borderValue = self.fill_value)
        return sample
