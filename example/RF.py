from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn import metrics
from scipy.stats import pearsonr
from time import *

begin_time = time()

def run_training(fold):
    df = pd.read_csv("train_PHT_folds.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.iloc[:, 11:33719]
    xvalid = df_valid.iloc[:, 11:33719]
    #print(xtrain)
    #print(xvalid)

    ytrain = df_train.PHT.values
    yvalid = df_valid.PHT.values
    #print(ytrain)
    #print(yvalid)

    clf = RandomForestRegressor(max_depth=10,
                                max_features='sqrt',
                                n_estimators=1500,
                                random_state=0)
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

    df_valid.loc[:, "RF_pred"] = pred

    return df_valid[["id", "PHT", "kfold", "RF_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(10):
        temp_df = run_training(j)
        dfs.append(temp_df)
    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("RF8.csv",index=False)

end_time = time()
run_time = end_time - begin_time
print("RF program run time:" , run_time)