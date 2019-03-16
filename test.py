import random

import os
import cv2
import string
from tqdm import tqdm
import click
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate
from dataset.data_transform import Resize, Rotation, Translation, Scale
from models.model_loader import load_model
from torchvision.transforms import Compose

import editdistance

def test(net, data, abc, visualize, batch_size, num_workers=0):
    data_loader = DataLoader(data, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False, collate_fn=text_collate)
    num_instance = 0
    tp = 0
    sum_ed = 0
    log = []
    net = net.eval()
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
                if visualize:
                    log.append((sample["name"][i], out[i], gts, cur_dist))
                    # if random.random() < 0.01:
                    #     log.append("pred: {}; gt: {}; dist: {}".format(out[i], gts, cur_dist))
                    # iterator.set_description(status)
                    # img = imgs[i].permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
                    # cv2.imshow("img", img)
                    # key = chr(cv2.waitKey() & 255)
                    # if key == 'q':
                    #     break
            # if key == 'q':
            #     break
            # if not visualize:
            #     iterator.set_description("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))
            # print ("acc: {0:.4f}; avg_ed: {0:.4f}".format(tp / count, avg_ed / count))

    eds = None
    if visualize:
        log = sorted(log, key=lambda tuple: tuple[3], reverse=True)
        for mess in log[:10]: print (mess)
        eds = [x[3] for x in log]

    acc = tp / num_instance
    avg_ed = sum_ed / num_instance
    if visualize:
        assert np.isclose(avg_ed, np.mean(eds))
    return acc, avg_ed

@click.command()
@click.option('--data-path', type=str, default=None, help='Path to dataset')
@click.option('--abc', type=str, default=string.digits+string.ascii_uppercase, help='Alphabet')
@click.option('--seq-proj', type=str, default="10x20", help='Projection of sequence')
@click.option('--backend', type=str, default="resnet18", help='Backend network')
@click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
@click.option('--input-size', type=str, default="320x32", help='Input size')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--visualize', type=bool, default=False, help='Visualize output')
def main(data_path, abc, seq_proj, backend, snapshot, input_size, gpu, visualize):
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # cuda = True if gpu is not '' else False

    input_size = [int(x) for x in input_size.split('x')]
    transform = Compose([
        Rotation(),
        Resize(size=(input_size[0], input_size[1]))
    ])
    if data_path is not None:
        data = TextDataset(data_path=data_path, mode="test", transform=transform)
    else:
        data = TestDataset(transform=transform, abc=abc)
    seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(data.get_abc(), seq_proj, backend, snapshot).eval()
    acc, avg_ed = test(net, data, data.get_abc(), visualize)
    print("Accuracy: {}".format(acc))
    print("Edit distance: {}".format(avg_ed))

if __name__ == '__main__':
    main()
