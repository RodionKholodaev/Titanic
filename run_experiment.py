import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.cv import evaluate, log_experiment
from src.prep import build


class GenderBaseline(ClassifierMixin, BaseEstimator):
    """Женщины выжили, мужчины нет. Ожидает колонку Sex: 'female'/'male' или 1/0."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        sex = X["Sex"]
        return ((sex == "female") | (sex == 1)).astype(int).to_numpy()


if __name__ == "__main__":
    X, y, _, _ = build()
    result = evaluate(GenderBaseline(), X, y)
    print({k: round(float(v), 4) for k, v in result.items()})
    log_experiment("gender_baseline", "female=1, male=0", result)
