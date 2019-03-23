import csv
from typing import List

import cv2

import os

from tqdm import tqdm
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

import config
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale
from models.model_loader import load_model
from torchvision.transforms import Compose

import editdistance

from ulti import Timing


def test(net, data, abc, visualize, batch_size, num_workers=0,
         output_csv=False, output_image=False):
    assert data.mode != "train"
    data_loader = DataLoader(data, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False, collate_fn=text_collate)
    num_instance = 0
    tp = 0
    sum_ed = 0
    log = [] # type: List[tuple]
    net = net.eval()
    output_image_dir = None
    if output_image:
        output_image_dir = os.path.join(config.output_dir, "input_images")
        if not os.path.exists(output_image_dir):
            print("Creating output_image_dir")
            os.makedirs(output_image_dir, exist_ok=True)

    iterator = tqdm(data_loader)
    with torch.no_grad():
        for sample in iterator:
            imgs = Variable(sample["img"])
            imgs = imgs.cuda()
            out = net(imgs, decode=True)
            gt = (sample["seq"].numpy() - 1).tolist()
            lens = sample["seq_len"].numpy().tolist()
            pos = 0
            # key = ''
            for i in range(len(out)):
                gts = ''.join(abc[c] for c in gt[pos:pos+lens[i]])
                pos += lens[i]
                cur_dist = 0
                if gts == out[i]:
                    tp += 1
                else:
                    cur_dist = editdistance.eval(out[i], gts) / max(len(gts), len(out[i]))
                    sum_ed += cur_dist

                num_instance += 1
                img = None
                if output_image:
                    img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)

                if visualize:
                    log.append((sample["name"][i], out[i], gts, cur_dist, img))

    eds = None
    if visualize:
        # print wrongest instances
        log = sorted(log, key=lambda tuple: tuple[3], reverse=True)
        for name, pred, gt, dist, _ in log[:10]:
            print (name, pred, gt, dist)
        eds = [x[3] for x in log]

    if output_image:
        with Timing("Write input_images"):
            for name, _, _, _, img in log[:config.num_write_input_img]:
                cv2.imwrite(os.path.join(output_image_dir, name), img)

    with Timing("Write csv file"):
        if output_csv:
            csv_file = open(os.path.join(config.output_dir, "output.csv"), "w")
            csv_writer = csv.writer(csv_file)
            for name, pred, gt, dist, _ in log:
                csv_writer.writerow(name, pred, gt, dist)

    acc = tp / num_instance
    avg_ed = sum_ed / num_instance
    if visualize:
        assert np.isclose(avg_ed, np.mean(eds))
    return acc, avg_ed

# @click.command()
# @click.option('--data-path', type=str, default=None, help='Path to dataset')
# @click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
# @click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
# @click.option('--backend', type=str, default="resnet18", help='Backend network')
# @click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
# @click.option('--input-size', type=str, default="320x32", help='Input size')
# @click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
# @click.option('--visualize', type=bool, default=False, help='Visualize output')


def main():
    input_size = [int(x) for x in config.input_size.split('x')]
    transform = Compose([
        Rotation(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    # if data_path is not None:
    data = TextDataset(data_path=config.test_path, mode=config.test_mode, transform=transform)
    # else:
    #     data = TestDataset(transform=transform, abc=abc)
    # seq_proj = [int(x) for x in config.seq_proj.split('x')]

    input_size = [int(x) for x in config.input_size.split('x')]
    net = load_model(input_size, data.get_abc(), None, config.backend, config.snapshot).eval()

    assert data.mode == config.test_mode
    acc, avg_ed = test(net, data, data.get_abc(), visualize=True,
                       batch_size=config.batch_size, num_workers=0,
                       output_csv=config.output_csv, output_image=config.output_image)

    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))


if __name__ == '__main__':
    main()
