import json
from tqdm import tqdm
import numpy as np


def flat_scores(preds, labels, scores):
    pred_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    TP = np.sum(np.logical_and(np.equal(labels_flat,1),np.equal(pred_flat,1)))
    #false positive
    FP = np.sum(np.logical_and(np.equal(labels_flat,0),np.equal(pred_flat,1)))
    #true negative
    TN = np.sum(np.logical_and(np.equal(labels_flat,0),np.equal(pred_flat,0)))
    #false negative
    FN = np.sum(np.logical_and(np.equal(labels_flat,1),np.equal(pred_flat,0)))
    if scores == 'detail':
        return TP, FP, TN, FN
    # print('TP:',TP)
    # print('TN:',TN)
    # print('FP:',FP)
    # print('FN:',FN)
    
    # print('P:',P)
    # print('R:',R)
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
    elif scores == 'f1':
        if np.isnan(P) or np.isnan(R) or (P == 0 and R == 0):
            return 0
        else:
            P = float(TP/(TP+FP))
            R = float(TP/(TP+FN))
            return 2*P*R/(P+R)
    elif scores == 'tpr':
        if TP + FN == 0:
            return 0
        else:
            return TP/(TP+FN)
    elif scores == 'fpr':
        if FP + TN == 0:
            return 0
        else:
            return FP/(FP+TN)
        # TPR = TP/(TP+FN)
        # FPR = FP/(FP+TN)
    elif scores == 'acc':
        return (TP+TN)/(TP+TN+FP+FN)
