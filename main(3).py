from sklearn.ensemble import RandomForestClassifier
from utils import flat_scores
import time
import json
import os
import numpy as np
import random
import sys
from colorama import Fore, Style

project_name = "logging-log4j2"


def readjson_all(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return json.load(f)


def get_data_from_file(filename):
    data = readjson_all(filename)  # 读取单个 JSON 文件
    if data['sample_type'] == 'NEGATIVE':
        Ytrain = 0
    else:
        Ytrain = 1

    tmp = [
        data['add_annotation_line'], data['add_call_line'], data['add_classname_line'],
        data['add_condition_line'], data['add_field_line'], data['add_import_line'],
        data['add_packageid_line'], data['add_parameter_line'], data['add_return_line'],
        data['del_annotation_line'], data['del_call_line'], data['del_classname_line'],
        data['del_condition_line'], data['del_field_line'], data['del_import_line'],
        data['del_packageid_line'], data['del_parameter_line'], data['del_return_line']
    ]

    return np.array(tmp), Ytrain


def load_all_data_from_folder(folder_path):
    Xtrain_total = []
    Ytrain_total = []

    dir_list = os.listdir(folder_path)
    random.seed(None)
    random.shuffle(dir_list)

    for filename in dir_list:
        if filename.endswith(".json"):
            full_path = os.path.join(folder_path, filename)
            Xtrain, Ytrain = get_data_from_file(full_path)
            Xtrain_total.append(Xtrain)
            Ytrain_total.append(Ytrain)

    return np.array(Xtrain_total), np.array(Ytrain_total)




time1 = time.time()
folder_path = './'+project_name+'/TRAIN/ALL'
Xtrain, Ytrain = load_all_data_from_folder(folder_path)

np.set_printoptions(threshold=sys.maxsize)
print(Ytrain)
print("由上可知：训练集已成功随机打乱")


rfc = RandomForestClassifier()
rfc = rfc.fit(Xtrain,Ytrain)
time2 = time.time()
print("------------------------------------------------------------------------")
print(Fore.RED +"SUCCESSFUL!"+Style.RESET_ALL)
print("------------------------------------------------------------------------")
print("Random_forest model ready!")

folder_path = './'+project_name+'/TEST/ALL'
Xtest, Ytest = load_all_data_from_folder(folder_path)
Ypredict = rfc.predict(Xtest)

pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)


print("------------------------------------------------------------------------")
print(Fore.RED+"f1_score:{0}\nPrecision:{1}\nrecall:{2}".format(f1, pre, recall)+Style.RESET_ALL)
time3 = time.time()
print("------------------------------------------------------------------------")
print("dataset and training: ",time2 - time1)
print("analysing:", time3 - time2)

