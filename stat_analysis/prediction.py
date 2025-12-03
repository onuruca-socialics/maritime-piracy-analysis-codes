from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_absolute_percentage_error, accuracy_score, f1_score,
)
import numpy as np


def predict_rfr(df):
    X = df[["vessel_type", "attack_type"]]
    y = df[["region"]]
    reg = RandomForestRegressor()
    reg.fit(X, y)
    yhat = reg.predict(X)
    err = mean_absolute_percentage_error(y, yhat)
    print(err)


def predict_dtc(df):
    X = df[["vessel_type", "place", "region"]]
    y = df[["attack_type"]]
    # y.loc[y["attack_type"] == 2] = 1
    y.loc[y["attack_type"] != 1] = 2
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    yhat = dtc.predict(X)
    score = accuracy_score(y, yhat)
    print(score)
    scores = cross_val_score(
        dtc, X, y,
        scoring="accuracy", cv=20,
    )
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    scores = cross_val_score(
        dtc, X, y,
        scoring="recall", cv=20,
    )
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

def predict_svc(df):
    X = df[["vessel_type", "place", "region"]]
    y = df[["attack_type"]]
    # y.loc[y["attack_type"] == 2] = 1
    y.loc[y["attack_type"] != 1] = 2
    svc = SVC()
    svc.fit(X, y)
    yhat = svc.predict(X)
    score = accuracy_score(y, yhat)
    print(score)


def predict_dtr(df):
    X = df[["vessel_type", "attack_type"]]
    y = df[["region"]]
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    yhat = reg.predict(X)
    err = mean_absolute_percentage_error(y, yhat)
    print(err)
    return
    scores = cross_val_score(
        tree_reg, X, y,
        scoring="neg_mean_absolute_percentage_error", cv=10,
    )
    tree_rmse_scores = -scores
    print("Scores:", tree_rmse_scores)
    print("Mean:", tree_rmse_scores.mean())
    print("Standard deviation:", tree_rmse_scores.std())
