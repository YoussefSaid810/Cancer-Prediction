import pandas as pd

import warnings
from preprocess import preprocessing
from load_lr import lr
from load_ddt import tree_load
from load_svm import svm_load

warnings.filterwarnings("ignore")
df = pd.read_csv("cancer.csv")
dec = int(input("do you want to enter? 1-excel 2-csv "))
path = input("Enter path: ")
if dec == 1:
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

########################################################################################
prepro = preprocessing(df)
prepro.process()
prepro.split()
########################################################################################
loadlr = lr(prepro.X_train, prepro.X_test, prepro.y_train, prepro.y_test)
loadlr.load()

dload = tree_load(prepro.X_train, prepro.X_test, prepro.y_train, prepro.y_test)
dload.load()
########################################################################################

svmload = svm_load(prepro.X_train, prepro.X_test, prepro.y_train, prepro.y_test)
svmload.load()
########################################################################################


voting = []

for i in range(len(loadlr.y_pred_1)):
    x = loadlr.y_pred_1[i] + svmload.y_pred_3[i] + dload.y_pred_2[i]
    if x >= 2:
        voting.append(1)
    else:
        voting.append(0)
print("Final Prediction : ")        
print(voting)
print('*' * 60)
