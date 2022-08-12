import glob
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr

if __name__ == "__main__":
    files = glob.glob(r"D:\Combining learning_cv10\PHT_model_pred\*.csv") #Base models prediction valves
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on=None, how="left")
    print(df)
    df.to_csv("Metadata_PHT.csv", index=False) #Generate a new dataset-Mate dataset

    targets = df.PHT.values
    pred_cols = ["GBDT_pred", "KNN_pred", "KRR_pred", "linearSVR_pred",
                 "RF_pred", "NuSVR_pred",  "SVR_pred", "XGBoost_pred"]

    for col in pred_cols:
        MAE = mean_absolute_error(targets, df[col].values)
        MSE = mean_squared_error(targets, df[col].values)
        RMSE = np.sqrt(mean_squared_error(targets, df[col].values))
        PCCs = pearsonr(targets, df[col].values)
        print(f"{col}, overall_MAE{MAE}")
        print(f"{col}, overall_MSE{MSE}")
        print(f"{col}, overall_RMSE{RMSE}")
        print(f"{col}, overall_PCCs{PCCs}")

    print("average")#Averaging
    avg_pred = np.mean(df[["GBDT_pred", "KNN_pred", "KRR_pred", "linearSVR_pred",
                 "RF_pred", "NuSVR_pred",  "SVR_pred", "XGBoost_pred"]].values, axis=1)
    print("MAE = ",mean_absolute_error(targets, avg_pred))
    print("MSE = ",mean_squared_error(targets, avg_pred))
    print("RMSE = ",np.sqrt((mean_squared_error(targets, avg_pred))))
    print("PCCs = ",pearsonr(targets, avg_pred))

    print("weight average")#Weighted averaging
    GBDT_pred = df.GBDT_pred.values
    KNN_pred = df.KNN_pred.values
    KRR_pred = df.KRR_pred.values
    RF_pred = df.RF_pred.values
    SVR_pred = df.SVR_pred.values
    linearSVR_pred = df.linearSVR_pred.values
    NuSVR_pred = df.NuSVR_pred.values
    XGBoost_pred = df.XGBoost_pred.values
    avg_pred = ((1.30804052)*GBDT_pred +
                (0.01819525)*KNN_pred +
                (-0.21156222)*KRR_pred +
                (0.11509102)*RF_pred +
                ( 0.04782879)*SVR_pred +
                (0.10942958)*linearSVR_pred +
                (-0.10072333)*NuSVR_pred +
                (-0.23791961)*XGBoost_pred)
    print("MAE = ", mean_absolute_error(targets, avg_pred))
    print("MSE = ", mean_squared_error(targets, avg_pred))
    print("RMSE = ", np.sqrt((mean_squared_error(targets, avg_pred))))
    print("PCCs = ", pearsonr(targets, avg_pred))


