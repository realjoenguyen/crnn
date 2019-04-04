import cv2
import torch
from torchvision.transforms import Compose
from dataset.data_transform import Resize
from models.model_loader import load_model
import config


def forward(img):
    assert img is not None
    sample = {"img": img}
    transform = Compose([
        Resize(size=(input_size[0], input_size[1]))
    ])
    sample = transform(sample)
    img = torch.from_numpy(sample["img"].transpose((2, 0, 1))).float()

    net = load_model(input_size, config.abc, None, config.backend, config.snapshot, cuda=False)
    # assert not net.is_cuda
    assert not next(net.parameters()).is_cuda
    net = net.eval()
    with torch.no_grad():
        img = img.unsqueeze(0)
        assert img.size()[0] == 1 and img.size()[1] == 3 and img.size()[2] < img.size()[3]
        out = net(img, decode=True)
        result = out[0]
        return result


if __name__ == '__main__':
    input_size = [int(x) for x in config.input_size.split('x')]
    test_img_path = "/root/textGenerator/source/out/train/files/plain_100_1747.jpg"
    img = cv2.imread(test_img_path)
    print(forward(img))
