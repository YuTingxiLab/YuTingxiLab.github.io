import glob
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import fmin
from functools import partial
from scipy.stats import pearsonr
"""
Different evaluation metrics can be selected
to optimize the weights of each model.
"""
class OptimizeR2:
    def __init__(self):
        self.coef_ = 0

    def _R2(self, coef, X, y):
        x_coef = X * coef
        predictions = np.sum(x_coef, axis=1)
        R2 = r2_score(y, predictions)
        return -1.0 * R2

    def fit(self, X, y):
        partial_loss = partial(self._R2, X=X, y=y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp=True)

    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.sum(x_coef, axis=1)
        return predictions

def run_training(pred_df, fold):
    train_df = pred_df[pred_df.kfold != fold].reset_index(drop=True)
    valid_df = pred_df[pred_df.kfold == fold].reset_index(drop=True)
    xtrain = train_df[[ "RF_pred","GBDT_pred","XGBoost_pred",
                     "KNN_pred", "NuSVR_pred","SVR_pred","KRR_pred","linearSVR_pred"
                       ]].values
    xvalid = valid_df[[  "RF_pred","GBDT_pred","XGBoost_pred",
                     "KNN_pred", "NuSVR_pred","SVR_pred","KRR_pred","linearSVR_pred"]].values

    opt = OptimizeR2()
    opt.fit(xtrain, train_df.Prot.values)
    preds = opt.predict(xvalid)
    R2 = r2_score(valid_df.Prot.values, preds)
    print(f"{fold}, {R2}")

    return opt.coef_

if __name__ == "__main__":
    files = glob.glob(r"D:\Combining learning_cv10\Prot_model_pred\*.csv")
    df = None
    for f in files:
        if df is None:
            df = pd.read_csv(f)
        else:
            temp_df = pd.read_csv(f)
            df = df.merge(temp_df, on=None, how="left")
    # print(df.head(6))
    targets = df.Prot.values
    pred_cols = [ "RF_pred","GBDT_pred","XGBoost_pred",
                     "KNN_pred", "NuSVR_pred","SVR_pred","KRR_pred","linearSVR_pred"]
    coefs = []
    for j in range(7):
        coefs.append(run_training(df, j))

    coefs = np.array(coefs)
    print(coefs)
    coefs = np.mean(coefs, axis=0)
    print(coefs)

    wt_avg = (+ coefs[0] * df.GBDT_pred.values
              + coefs[1] * df.KNN_pred.values
              + coefs[2] * df.KRR_pred.values
              + coefs[3] * df.RF_pred.values
              + coefs[4] * df.SVR_pred.values
              + coefs[5] * df.XGBoost_pred.values
              + coefs[6] * df.NuSVR_pred.values
              + coefs[7] * df.linearSVR_pred.values
            )
    print("optimal R2 after finding coefs")
    print(r2_score(targets, wt_avg))
    print("optimal PCCs,MSE,MAE,and RMSE after finding coefs")
    print("PCCs:", pearsonr(targets, wt_avg))
    print("MSE:", mean_squared_error(targets, wt_avg))
    print("MAE:", mean_absolute_error(targets, wt_avg))
    print("RMSE:", np.sqrt((mean_absolute_error(targets, wt_avg))))

