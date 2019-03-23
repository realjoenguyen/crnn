import os

dau_cau ="àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
abc = dau_cau + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;?@[\\]^_`{|}~ "
# no duplicate char
assert len(set(abc)) == len(abc)

# data
train_dev_path = "/root/TA/data/clean/train_dev/"
current_path = os.path.dirname(__file__)
json_file_path = os.path.join(current_path, "desc.json")
output_dir = os.path.join(current_path, "out")

test_mode = "test_annotated"
# test_path = "/root/TA/data/clean/"
test_path = os.path.join("/root/TA/data/clean/", test_mode)

# basemodel
backend = "resnet18"
# snapshot = os.path.join(current_path, "out/crnn_resnet18_best")
snapshot = os.path.join(current_path, "out/crnn_" + backend + "_best")

# logging
num_write_input_img = 30
output_csv = False
output_image = True
output_transform = False

# model config
# input_size = "1920x128"
input_size = "3840x128"
base_lr = 1e-3
step_size=500
max_iter  = 6000
batch_size = 100
dropout = 0.25

# CNN
num_filter = 256
downrate = 2 ** 4

# LSTM
lstm_input_size = 512
lstm_hidden_size = 512
lstm_num_layers = 2

# @click.command()
# @click.option('--data-path', type=str, default=None, help='Path to dataset')
# @click.option('--abc', type=str, default=abc_vocab, help='Alphabet')
# @click.option('--seq-proj', type=str, default="10x50", help='Projection of sequence')
# @click.option('--backend', type=str, default="resnet18", help='Backend network')
# @click.option('--snapshot', type=str, default=None, help='Pre-trained weights')
# @click.option('--input-size', type=str, default="1280x96", help='Input size')
# @click.option('--base-lr', type=float, default=1e-3, help='Base learning rate')
# @click.option('--step-size', type=int, default=500, help='Step size')
# @click.option('--max-iter', type=int, default=6000, help='Max iterations')
# @click.option('--batch-size', type=int, default=200, help='Batch size')
# @click.option('--output-dir', type=str, default=None, help='Path for snapshot')
# @click.option('--test-epoch', type=int, default=None, help='Test epoch')
# @click.option('--test-init', type=bool, default=False, help='Test initialization')
# @click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')

