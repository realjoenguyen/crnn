json_path = '/root/TA/data/'
dau_cau ="àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
abc = dau_cau + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;?@[\\]^_`{|}~ "
# no duplicate char
assert len(set(abc)) == len(abc)

data_path = "/root/TA/data/clean/train_dev/"
backend = "resnet18"
snapshot = "/root/crnn/crnn_simple/out/crnn_resnet18_best"
input_size = "1920x128"
base_lr = 1e-3
step_size=500
max_iter  = 6000
batch_size = 100
output_dir = "/root/crnn/crnn_simple/out"
dev_epoch = 5
dev_init = True


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

