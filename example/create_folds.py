import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("Wheatdata.csv")
    df.loc[:, "kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    y = df.PHT.values
    skf = model_selection.KFold(n_splits=10)

    for f, (t_, v_)in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    df.to_csv("train_PHT_folds.csv", index=False)