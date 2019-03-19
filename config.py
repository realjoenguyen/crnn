import os

dau_cau ="àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
abc = dau_cau + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;?@[\\]^_`{|}~ "
# no duplicate char
assert len(set(abc)) == len(abc)

train_dev_path = "/root/TA/data/clean/train_dev/"
test_path = "/root/TA/data/clean/test/"

current_path = os.path.dirname(__file__)
json_file_path = os.path.join(current_path, "desc.json")
backend = "resnet18"
snapshot = os.path.join(current_path, "out/crnn_resnet18_best")
output_dir = os.path.join(current_path, "out")

test_mode = "test"

# logging
num_write_input_img = 30
output_csv = False
output_image = True

input_size = "1920x128"
base_lr = 1e-3
step_size=500
max_iter  = 6000
batch_size = 100

num_filter = 256

lstm_input_size = 256
lstm_hidden_size = 512
lstm_num_layers = 2
dropout = 0.25

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

