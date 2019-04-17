from collections import OrderedDict

import torch

from .crnn import CRNN
def load_weights(target, snapshot, cuda=True):
    if cuda:
        source_state = torch.load(snapshot)
    else:
        print ("WARNING: use cpu")
        source_state = torch.load(snapshot, map_location="cpu")

    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        else:
            print ("WARNING: this weight is the same: ", k)
            new_dict[k] = v
    target.load_state_dict(new_dict)

def load_model(input_size, abc, seq_proj=[0, 0], backend='resnet18', snapshot=None, cuda=True):
    assert type(abc) == str
    net = CRNN(input_size=input_size, abc=abc, seq_proj=seq_proj, backend=backend)
    if snapshot is not None:
        print ("Loading model from", snapshot)
        load_weights(net, snapshot, cuda)
    if cuda:
        net = net.cuda()
    return net
