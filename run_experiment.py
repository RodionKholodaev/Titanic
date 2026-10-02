import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.cv import evaluate, log_experiment


class GenderBaseline(ClassifierMixin, BaseEstimator):
    """Женщины выжили, мужчины нет. Ожидает колонку Sex: 'female'/'male' или 1/0."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        sex = X["Sex"]
        return ((sex == "female") | (sex == 1)).astype(int).to_numpy()


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    X, y = train.drop(columns="Survived"), train["Survived"]
    result = evaluate(GenderBaseline(), X, y)
    print({k: round(float(v), 4) for k, v in result.items()})
    log_experiment("gender_baseline", "female=1, male=0", result)
