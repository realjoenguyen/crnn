import torch
import numpy as np

def text_collate(batch):
    img = list()
    seq = list()
    seq_len = list()
    name = list()

    for sample in batch:
        img.append(torch.from_numpy(sample["img"].transpose((2, 0, 1))).float())
        seq.extend(sample["seq"])
        seq_len.append(sample["seq_len"])
        name.append(sample["name"])

    img = torch.stack(img)
    seq = torch.Tensor(seq).int()
    seq_len = torch.Tensor(seq_len).int()
    batch = {"img": img, "seq": seq, "seq_len": seq_len, "name" : name}
    return batch
