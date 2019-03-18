import os
import numpy as np
from tqdm import tqdm
from models.model_loader import load_model
from torchvision.transforms import Compose
from dataset.data_transform import Resize, Rotation, Translation, Scale
from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from inference import test


def main():
    input_size = [int(x) for x in config.input_size.split('x')]
    # TODO: 1) Sử dụng elastic transform 2) Random erasor một phần của bức ảnh. de data augmentation
    transform = Compose([
        Rotation(),
        Resize(size=(input_size[0], input_size[1]), data_augmen=True)
    ])
    # if config.data_path is not None:
    data = TextDataset(data_path=config.train_dev_path, mode="train", transform=transform)
    # else:
    #     data = TestDataset(transform=transform, abc=config.abc)

    # seq_proj = [int(x) for x in seq_proj.split('x')]
    net = load_model(input_size, data.get_abc(), None, config.backend, config.snapshot)

    # optimizer = optim.Adam(net.parameters(), lr = config.base_lr, weight_decay=0.0001)
    optimizer = optim.Adam(net.parameters(), lr = config.base_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=5, verbose=True)
    loss_function = CTCLoss()

    dev_avg_ed_best = float("inf")
    epoch_count = 0
    print ("Start running ...")

    while True:
        # if (config.dev_epoch is not None and epoch_count != 0 and epoch_count % config.dev_epoch == 0) or (config.dev_init and epoch_count == 0):
        #     print("dev phase")
        #     data.set_mode("test")
        #     net = net.eval()
        #     _, avg_ed = test(net, data, data.get_abc(), cuda, visualize=True,
        #                        batch_size=config.batch_size, num_workers=0)
        #
        #     net = net.train()
        #     data.set_mode("train")
        #     if avg_ed > avg_ed_best:
        #         if config.output_dir is not None:
        #             torch.save(net.state_dict(), os.path.join(config.output_dir,
        #                                                       "crnn_" + config.backend + "_best"))
        #             print ("Saving best model to", os.path.join(config.output_dir,
        #                                                       "crnn_" + config.backend + "_best"))
        #         avg_ed_best = avg_ed
        #     print("avg_ed: {}; avg_ed_best: {}".format(avg_ed, avg_ed_best))

        # test dev phrase
        if epoch_count == 0:
            print("dev phase")
            data.set_mode("dev")
            acc, dev_avg_ed = test(net, data, data.get_abc(), visualize=True,
                             batch_size=config.batch_size, num_workers=0)
            print("acc: {}; avg_ed: {}; avg_ed_best: {}".format(acc, dev_avg_ed, dev_avg_ed_best))

        net = net.train()
        data.set_mode("train")
        data_loader = DataLoader(data, batch_size=config.batch_size,
                                 num_workers=0, shuffle=True, collate_fn=text_collate)
        loss_mean = []
        iterator = tqdm(data_loader)
        for sample in iterator:
            optimizer.zero_grad()
            imgs = Variable(sample["img"])
            labels = Variable(sample["seq"]).view(-1)
            label_lens = Variable(sample["seq_len"].int())
            imgs = imgs.cuda()

            preds, softmax = net(imgs, print_softmax=True)
            preds = preds.cpu()
            pred_lens = Variable(Tensor([preds.size(0)] * len(label_lens)).int())

            # ctc loss len > label_len
            assert preds.size()[0] > max(label_lens).item()
            loss = loss_function(preds, labels, pred_lens, label_lens)

            # unit test
            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()
            assert loss.item() != 0
            loss.backward()
            for name, para in net.named_parameters():
                if (para.grad is None or para.grad.equal(torch.zeros_like(para.grad))) and para.requires_grad:
                    print ("WARNING: There is no grad at", name)

            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            loss_mean.append(loss.item())
            optimizer.step()

        print("dev phase")
        data.set_mode("dev")
        acc, dev_avg_ed = test(net, data, data.get_abc(), visualize=True,
                           batch_size=config.batch_size, num_workers=0)

        if dev_avg_ed < dev_avg_ed_best:
            assert config.output_dir is not None
            torch.save(net.state_dict(), os.path.join(config.output_dir,
                                                      "crnn_" + config.backend + "_best"))
            print ("Saving best model to", os.path.join(config.output_dir,
                                                      "crnn_" + config.backend + "_best"))
            dev_avg_ed_best = dev_avg_ed

        # TODO: print avg_ed & acc in train epoch
        print ("train: epoch: {}; loss_mean: {}".format(epoch_count, np.mean(loss_mean)))
        print ("dev: acc: {}; avg_ed: {}; avg_ed_best: {}".format(acc, dev_avg_ed, dev_avg_ed_best))

        # TODO: add tensorboard to visualize loss_mean & avg_ed & acc
        lr_scheduler.step(dev_avg_ed)
        epoch_count += 1

if __name__ == '__main__':
    main()
