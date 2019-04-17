from typing import List

from torch.utils.data import Dataset
import json
import os
import cv2

import config
from config import json_file_path

def detect_label_ocr(input_str):
    keywords = [
        ["Tong Tien", "Total", "Tổng Tiền", "tổng tiền", "TỔNG TIỀN", "TONG TIEN"],
        ["Dia chi", "địa chỉ", "ĐỊA CHỈ", "Địa chỉ", "DIA CHI", "Address", "ADDRESS", "huyện", "thanh pho", "xã"],
        ["MaKH", "Mã khách hàng", "Mã KH", "Mã kh", "mã kh", "mã khách hàng", "MKH", "mkh"],
        ["Mã NV", "Ma NV", "mã nhân viên", "MÃ NV", "Mã nhân viên"],
        ["VAT", "MST", "MÃ SỐ THUẾ", "mã số thuế", "ma so thue", "Mã số thuế", "Ma so thue"],
        ["Công ty", "Cong ty", "CONG TY", "cong ty", "doanh nghiep", "DN", "doanh nghiệp", "hợp tác xã", "cửa hàng", "tư nhân", "cty", "trường", "htx", "chi nhánh", "văn phòng", "trung tâm"],
        ["Tên KH", "KH", "Tên khách hàng", "Ten khach hang", "Khách hàng", "khách hàng", "KHÁCH HÀNG",
         "TEN KHACH HANG", "khach hang", "KHACH HANG"],
        ["NHAN VIEN", "Nhan Vien", "Nhan vien", "nhan vien", "Ten nhan vien", "Cashier",
         "NHÂN VIÊN", "Nhân Viên", "Nhân viên", "nhân viên", "Tên nhân viên", "NV", "nguyễn"]]

    kw = {
        0: "Tổng tiền",
        1: "Địa chỉ",
        2: "Mã khách hàng",
        3: "Mã nhân viên",
        4: "Mã số thuế / VAT",
        5: "Công ty",
        6: "Tên khách hàng",
        7: "Tên nhân viên",
        8: "Other"
    }
    input_str = input_str.lower()
    for index, list_keywords in enumerate(keywords):
        for key in list_keywords:
            if key in input_str or key.lower() in input_str:
                # return kw[index]
                return index
    print ("WARNING: ", input_str, "is not detected")
    return 8

label_dict = {
    0: "Tổng tiền",
    1: "Địa chỉ",
    2: "Mã khách hàng",
    3: "Mã nhân viên",
    4: "Mã số thuế / VAT",
    5: "Công ty",
    6: "Tên khách hàng",
    7: "Tên nhân viên",
    8: "Other"
}


class TextDataset(Dataset):
    def __init__(self, data_path, mode="train", transform=None):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        print('Open config file from', json_file_path)
        self.config = json.load(open(json_file_path, encoding='utf-8'))

        # print ("Cut half train")
        # train_config = self.config["train"]
        # self.config["train"] = train_config[:int(len(train_config) / 2)]

        # print("Merging test_annotated")
        # self.config["train"].extend(self.config["test_annotated"])

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

        img_path = os.path.join(self.data_path, self.mode, name)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("img is None at", img_path)
        seq = self.text_to_seq(text)
        assert seq is not None

        # sample = {"img": img, "seq": seq, "seq_len": len(seq), "aug": self.mode == "train"}
        sample = {"img": img,
                  "seq": seq,
                  "seq_len": len(seq),
                  "aug": self.mode == "train",
                  "name": name,
                  "label": detect_label_ocr(text),
                  "text": text}

        origin_img = img
        if self.transform:
            sample = self.transform(sample)
        new_img = sample["img"]

        if config.output_transform:
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
