import json
import os

from config import json_path
json_file_path = os.path.join(json_path, "desc.json")
a = json.load(open(json_file_path, encoding="utf-8"))

def more_max_len(key, max_len):
    print ('key, max_len =', key, max_len)
    longest_text = None
    lst_name = []
    lst_text = []

    for ex in a[key]:
        if len(ex["text"]) > max_len:
            # max_len = len(ex["text"])
            # longest_text = ex["text"]
            # name =  ex["name"]
            lst_name.append(ex["name"])
            lst_text.append((ex["text"], len(ex["text"])))

    print (max_len)
    print (longest_text)
    print (len(lst_name))
    # print (lst_name)
    print (lst_text)

more_max_len("train", 60)
more_max_len("test", 60)
