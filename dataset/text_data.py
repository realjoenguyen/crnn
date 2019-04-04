from typing import List

from torch.utils.data import Dataset
import json
import os
import cv2

import config
from config import json_file_path


class TextDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        print('Open config file from', json_file_path)
        self.config = json.load(open(json_file_path, encoding='utf-8'))

        print ("Cut half train")
        train_config = self.config["train"]
        self.config["train"] = train_config[:int(len(train_config) / 2)]

        print("Merging test_annotated")
        self.config["train"].extend(self.config["test_annotated"])

        self.transform = transform
        self.cnt_log = 0

    def get_abc(self):
        if type(self.config["abc"]) == list:
            self.config["abc"] = ''.join(self.config["abc"])
        assert type(self.config["abc"]) == str
        return self.config["abc"]

    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        # if self.mode == "test":
        #     return int(len(self.config[self.mode]) * 0.01)
        return len(self.config[self.mode])

    def __getitem__(self, idx):
        name = self.config[self.mode][idx]["name"]
        text = self.config[self.mode][idx]["text"]
        # for ch in text: assert ch in config.abc

        img = cv2.imread(os.path.join(self.data_path, self.mode, "files", name))
        assert img is not None
        seq = self.text_to_seq(text)
        assert seq is not None

        # sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        sample = {"img": img,
                  "seq": seq,
                  "seq_len": len(seq),
                  "aug": self.mode == "train",
                  "name": name}
        origin_img = img
        if self.transform:
            sample = self.transform(sample)
        new_img = sample["img"]

        if config.output_transform:
            if self.cnt_log == 0:
                print("Remove the old", os.path.join(config.output_dir, "input_images"))
                for file in os.path.join(config.output_dir, "input_images"):
                    if os.path.isfile(file):
                        os.remove(file)

            if self.cnt_log < config.num_write_input_img:
                cv2.imwrite(os.path.join(config.output_dir, "input_images", sample["name"][:-4] + "_before.jpg"),
                            origin_img)
                cv2.imwrite(os.path.join(config.output_dir, "input_images", sample["name"][:-4] + "_after.jpg"),
                            new_img)
                self.cnt_log += 1
        return sample

    def text_to_seq(self, text):
        seq = []
        for c in text:
            seq.append(self.config["abc"].find(c) + 1)  # blank == 0
        return seq
