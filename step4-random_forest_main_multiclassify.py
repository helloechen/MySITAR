from sklearn.ensemble import RandomForestClassifier
from utils_multi import flat_scores,load_all_data_from_folder
import time
import numpy as np
import sys
from colorama import Fore, Style

# if data['sample_type'] == 'NEGATIVE':
#     Y = 0
# elif data['test_typ'] == "CREATE":
#     Y = 1
# elif data['test_typ'] == "DELETE":
#     Y = 2
# elif data['test_typ'] == "EDIT":
#     Y = 3

project_name = "biojava"

time1 = time.time()
folder_path = './'+project_name+'/TRAIN/ALL'
Xtrain, Ytrain = load_all_data_from_folder(folder_path)

np.set_printoptions(threshold=sys.maxsize)
#print(Ytrain)



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
print("Y_predict")
print(Ypredict)
print("------------------------------------------------------------------------")
print("Y_test")
print(Ytest)
print("------------------------------------------------------------------------")
total_correct = np.sum(np.equal(Ypredict,Ytest))
print("total_multi_correct_rate")
print(total_correct/len(Ypredict),total_correct,"in",len(Ypredict))
print("------------------------------------------------------------------------")
