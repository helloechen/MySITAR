import json
import os
import random
import numpy as np

def readjson_all(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        return json.load(f)


def get_data_from_file(filename):
    data = readjson_all(filename)  # 读取单个 JSON 文件
    if data['sample_type'] == 'NEGATIVE':
        Ytrain = 0
    elif data['test_typ'] == "CREATE":
        Ytrain = 1
    elif data['test_typ'] == "DELETE":
        Ytrain = 2
    elif data['test_typ'] == "EDIT":
        Ytrain = 3

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

def flat_scores(preds, labels, scores):
    pred_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    TP = np.sum(np.logical_and(labels_flat >=1, pred_flat>=1))
    FP = np.sum(np.logical_and(np.equal(labels_flat,0),pred_flat>=1))
    TN = np.sum(np.logical_and(np.equal(labels_flat,0),np.equal(pred_flat,0)))
    FN = np.sum(np.logical_and(labels_flat >=1,np.equal(pred_flat,0)))
    if scores == 'detail':
        return TP, FP, TN, FN

    if scores == 'precision':
        if TP == 0 and FP == 0:
            return 0
        else:
            P = float(TP/(TP+FP))
            return P
    elif scores == 'recall':
        if TP == 0 and FN == 0:
            return 0
        else:
            R = float(TP/(TP+FN))
            return R
