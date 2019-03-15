# coding=utf-8
import io
import os
files_path = {}
labels_path = {}
files_path["train"] = '/home/ta/Projects/capstone/data/data_gen/source/out/train/files/*'
files_path["test"] = '/home/ta/Projects/capstone/data/data_gen/source/out/dev/files/*'
labels_path["train"] = '/home/ta/Projects/capstone/data/data_gen/source/out/train/labels/'
labels_path["test"] = '/home/ta/Projects/capstone/data/data_gen/source/out/dev/labels/'

import json
import glob
dau_cau ="àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
abc_vocab = dau_cau + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;?@[\\]^_`{|}~ "
abc_vocab = list(abc_vocab)

data = {}
data["abc"] = abc_vocab
data["train"] = []
data["test"] = []
text_vocab = set()
cnt = 0

def create_data(type_dataset):
    cnt = 0
    print ("Creating data", type_dataset)
    for image_file_path in glob.glob(files_path[type_dataset]):
        image_file_name = os.path.basename(image_file_path)
        text_file_path = os.path.join(labels_path[type_dataset], image_file_name[:-4] + '.gt.txt')
        instance = {}
        instance["name"] = image_file_name
        instance["text"] = open(text_file_path, 'r').read()

        if len([e for e in list(instance["text"]) if e not in abc_vocab]) > 0:
            print (instance["name"])
            print (instance["text"])
            continue
        if len(list(instance["text"])) > 80:
            cnt += 1
            continue

        text_vocab.update(list(instance["text"]))
        data[type_dataset].append(instance)

    print ('Len =', len(data[type_dataset]))
    print ("Skip > 80=", cnt)

create_data("train")
create_data("test")

print("Finish creating json")
with open("desc.json", "w") as outfile:
    json.dump(data, outfile, ensure_ascii=False)