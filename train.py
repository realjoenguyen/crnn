import glob

import os
import numpy as np
from tqdm import tqdm
from models.model_loader import load_model
from torchvision.transforms import Compose
# from dataset.data_transform import Resize, Rotation, Translation, Scale
from dataset.data_transform import Resize
# from dataset.test_data import TestDataset
from dataset.text_data import TextDataset
from dataset.collate_fn import text_collate

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from inference import test


def main():
    input_size = [int(x) for x in config.input_size.split('x')]
    # TODO: 1) Sử dụng elastic transform 2) Random erasor một phần của bức ảnh. de data augmentation
    transform = Compose([
        # Rotation(),
        # Resize(size=(input_size[0], input_size[1]), data_augmen=True)
        Resize(size=(input_size[0], input_size[1]))
    ])
    data = TextDataset(data_path=config.data_path, mode="train", transform=transform)
    print("Len of train =", len(data))
    data.set_mode("dev")
    print("Len of dev =", len(data))
    data.set_mode("test")
    print("Len of test =", len(data))
    data.set_mode("test_annotated")
    print("Len of test_annotated =", len(data))
    data.set_mode("train")

    net = load_model(input_size, data.get_abc(), None, config.backend, config.snapshot)
    total_params = sum(p.numel() for p in net.parameters())
    train_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("# of parameters =", total_params)
    print("# of non-training parameters =", total_params - train_total_params)
    print("")
    if config.output_image:
        input_img_path = os.path.join(config.output_dir, "input_images")
        file_list = glob.glob(input_img_path + "/*")
        print("Remove the old", input_img_path)
        for file in file_list:
            if os.path.isfile(file):
                os.remove(file)

    optimizer = optim.Adam(net.parameters(), lr=config.base_lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    loss_function = CTCLoss(blank=0)
    loss_label = nn.NLLLoss()

    dev_avg_ed_best = float("inf")
    anno_avg_ed_best = 0.1544685954462857
    epoch_count = 0
    print("Start running ...")

    while True:
        # test dev phrase
        # if epoch_count == 0:
        #     print("dev phase")
        #     data.set_mode("dev")
        #     acc, dev_avg_ed = test(net, data, data.get_abc(), visualize=True,
        #                            batch_size=config.batch_size, num_workers=config.num_worker)
        #     print("DEV: acc: {}; avg_ed: {}; avg_ed_best: {}".format(acc, dev_avg_ed, dev_avg_ed_best))
        #
        #     data.set_mode("test_annotated")
        #     annotated_acc, annotated_avg_ed = test(net, data, data.get_abc(), visualize=True,
        #                                            batch_size=config.batch_size, num_workers=config.num_worker)
        #     print("ANNOTATED: acc: {}; avg_ed: {}".format(annotated_acc, annotated_avg_ed))

        net = net.train()
        data.set_mode("train")
        data_loader = DataLoader(data, batch_size=config.batch_size,
                                 num_workers=config.num_worker, shuffle=True, collate_fn=text_collate)
        loss_mean = []
        iterator = tqdm(data_loader)
        for sample in iterator:
            optimizer.zero_grad()
            imgs = Variable(sample["img"])
            labels_ocr = Variable(sample["seq"]).view(-1)
            labels_ocr_len = Variable(sample["seq_len"].int())
            labels = Variable(sample["label"].long())
            imgs = imgs.cuda()

            preds, label_logsoftmax = net(imgs)
            preds = preds.cpu()
            label_logsoftmax = label_logsoftmax.cpu()
            pred_lens = Variable(Tensor([preds.size(0)] * len(labels_ocr_len)).int())

            # ctc loss len > label_len
            assert preds.size()[0] > max(labels_ocr_len).item()
            loss = loss_function(preds, labels_ocr, pred_lens, labels_ocr_len) + loss_label(label_logsoftmax, labels)

            # unit test
            assert not torch.isnan(loss).any()
            assert not torch.isinf(loss).any()
            assert loss.item() != 0
            loss.backward()
            for name, para in net.named_parameters():
                if (para.grad is None or para.grad.equal(torch.zeros_like(para.grad))) and para.requires_grad:
                    print("WARNING: There is no grad at", name)

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
            print("Saving best model to", os.path.join(config.output_dir,
                                                       "crnn_" + config.backend + "_best"))
            dev_avg_ed_best = dev_avg_ed

        # TODO: print avg_ed & acc in train epoch
        print("train: epoch: {}; loss_mean: {}".format(epoch_count, np.mean(loss_mean)))
        print("dev: acc: {}; avg_ed: {}; avg_ed_best: {}".format(acc, dev_avg_ed, dev_avg_ed_best))

        data.set_mode("test_annotated")
        annotated_acc, annotated_avg_ed = test(net, data, data.get_abc(), visualize=True,
                                               batch_size=config.batch_size, num_workers=config.num_worker)
        if annotated_avg_ed < anno_avg_ed_best:
            assert config.output_dir is not None
            torch.save(net.state_dict(), os.path.join(config.output_dir,
                                                      "crnn_" + config.backend + "_best_anno"))
            print("Saving best model to", os.path.join(config.output_dir,
                                                       "crnn_" + config.backend + "_best_anno"))
            anno_avg_ed_best = annotated_avg_ed
        print("ANNOTATED: acc: {}; avg_ed: {}, best: {}".format(annotated_acc, annotated_avg_ed, anno_avg_ed_best))

        # TODO: add tensorboard to visualize loss_mean & avg_ed & acc
        lr_scheduler.step(dev_avg_ed)
        epoch_count += 1


if __name__ == '__main__':
    main()
