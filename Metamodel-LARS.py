import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn import metrics
from scipy.stats import pearsonr
from time import *
from sklearn import linear_model

begin_time = time()

def run_training(fold):
    df = pd.read_csv("SELGMPHT.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    #print(np.shape(df))

    xtrain = df_train.iloc[:, 3:11]
    xvalid = df_valid.iloc[:, 3:11]
    #print(xtrain)
    #print(xvalid)

    ytrain = df_train.PHT.values
    yvalid = df_valid.PHT.values
    #print(ytrain)
    #print(yvalid)

    clf = linear_model.LassoLars(alpha=0.001, normalize=False)
    clf.fit(xtrain,ytrain)
    pred = clf.predict(xvalid)

    MSE = mean_squared_error(yvalid, pred)
    RMSE = np.sqrt(mean_squared_error(yvalid, pred))
    MAE = mean_absolute_error(yvalid, pred)
    PCCs = pearsonr(yvalid, pred)

    print(f"fold={fold}, MSE={MSE}")
    print(f"fold={fold}, RMSE={RMSE}")
    print(f"fold={fold}, MAE={MAE}")
    print(f"fold={fold}, PCCs={PCCs}")

    df_valid.loc[:, "LARS_pred"] = pred

    return df_valid[["id", "length", "kfold", "LARS_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(10):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("LARSPHT_pred.csv",index=False)

end_time = time()
run_time = end_time - begin_time
print("LARS program run time:" , run_time)