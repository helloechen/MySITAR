from sklearn.linear_model import LogisticRegression
from utils import flat_scores,load_all_data_from_folder
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
import sys
from colorama import Fore, Style

project_name = ""

time1 = time.time()
folder_path = './'+project_name+'/TRAIN/ALL'
Xtrain, Ytrain = load_all_data_from_folder(folder_path)

np.set_printoptions(threshold=sys.maxsize)
print(Ytrain)
print("由上可知：训练集已成功随机打乱")

log_reg = LogisticRegression(max_iter=3000)
clm=log_reg.fit(Xtrain,Ytrain)

time2 = time.time()
print("------------------------------------------------------------------------")
print(Fore.RED +"SUCCESSFUL!"+Style.RESET_ALL)
print("------------------------------------------------------------------------")
print("LogisticRegression model ready!")

folder_path = './'+project_name+'/TEST/ALL'
Xtest, Ytest = load_all_data_from_folder(folder_path)
Ypredict = clm.predict(Xtest)

pre = flat_scores(Ypredict, Ytest, 'precision')
recall = flat_scores(Ypredict, Ytest, 'recall')
f1 = 2*pre*recall/(pre+recall)

print("------------------------------------------------------------------------")
print(Fore.RED+"f1_score:{0}\nPrecision:{1}\nrecall:{2}".format(f1, pre, recall)+Style.RESET_ALL)
time3 = time.time()
print("------------------------------------------------------------------------")
cm = confusion_matrix(Ytest, Ypredict)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
print("Confusion Matrix:")
print(cm_df)