# coding=utf-8
import csv
from ulti import Timing
import os

files_path = {}
labels_path = {}
files_path["train"] = '/home/ta/Projects/capstone/data/data_gen/out/train/files/*'
files_path["dev"] = '/home/ta/Projects/capstone/data/data_gen/out/dev/files/*'
files_path["test"] = '/home/ta/Projects/capstone/data/data_gen/out/test/files/*'

labels_path["train"] = '/home/ta/Projects/capstone/data/data_gen/out/train/labels/'
labels_path["dev"] = '/home/ta/Projects/capstone/data/data_gen/out/dev/labels/'
labels_path["test"] = '/home/ta/Projects/capstone/data/data_gen/out/test/labels/'

files_path["test_annotated"] = "/home/ta/Projects/capstone/data/annotated/final/*"
labels_path["test_annotated"] = "/home/ta/Projects/capstone/data/annotated/labels_info.csv"

import json
import glob
dau_cau ="àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹýÀÁÃẠẢĂẮẰẲẴẶÂẤẦẨẪẬÈÉẸẺẼÊỀẾỂỄỆĐÌÍĨỈỊÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢÙÚŨỤỦƯỨỪỬỮỰỲỴỶỸÝ"
abc_vocab = dau_cau + "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;?@[\\]^_`{|}~ "
abc_vocab = list(abc_vocab)

data = {}
data["abc"] = abc_vocab
data["train"] = []
data["dev"] = []
data["test"] = []
data["test_annotated"] = []
# text_vocab = set()
cnt = 0

def create_data(type_dataset):
    cnt = 0
    # with Timing("Creating data " + type_dataset):
    print ("Creating data", type_dataset)
    lst_file = glob.glob(files_path[type_dataset])
    assert len(lst_file) > 0

    for image_file_path in lst_file:
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

        # text_vocab.update(list(instance["text"]))
        data[type_dataset].append(instance)

    print ('Len =', len(data[type_dataset]))
    print ("Skip > 80 =", cnt)
    print ("")

create_data("train")
create_data("dev")
create_data("test")

# create test annotated
labels_test_annotated = {}
print ("Creating test_annotated")
with Timing("Create labels"):
    f = open(labels_path["test_annotated"])
    reader = csv.reader(f, delimiter=',')
    for line in reader:
        image_file_name = line[0]
        ocr_text = line[1]
        labels_test_annotated[image_file_name] = ocr_text

# with Timing("Create json infor"):
cnt = 0
lst_file = glob.glob(files_path["test_annotated"])
assert len(lst_file) > 0

for image_file_path in lst_file:
    image_file_name = os.path.basename(image_file_path)

    instance = {}
    instance["name"] = image_file_name
    instance["text"] = labels_test_annotated[image_file_name]

    if len([e for e in list(instance["text"]) if e not in abc_vocab]) > 0:
        # print ("NOT have in vocab")
        print (instance["name"])
        print (instance["text"])
        # continue

    if len(list(instance["text"])) > 80:
        cnt += 1
        print (instance["name"])
        print (instance["text"])
        # continue

    data["test_annotated"].append(instance)

print ('Len =', len(data["test_annotated"]))
print ("Skip > 80 =", cnt)

with Timing("Dumping json"):
    with open("desc.json", "w") as outfile:
        json.dump(data, outfile, ensure_ascii=False)