import sys

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.cv import evaluate, log_experiment
from src.prep import GroupSurvival, build


class GenderBaseline(ClassifierMixin, BaseEstimator):
    """Женщины выжили, мужчины нет. Ожидает колонку Sex: 'female'/'male' или 1/0."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        sex = X["Sex"]
        return ((sex == "female") | (sex == 1)).astype(int).to_numpy()





EXPERIMENTS = {
    "gender_baseline": (GenderBaseline(), "female=1, male=0"),
    "logreg": (make_pipeline(GroupSurvival(), StandardScaler(), LogisticRegression(max_iter=1000)),
               "StandardScaler + LogisticRegression, + Sex_Pclass, HasGroup, GroupSurvival"),
    "rf": (make_pipeline(GroupSurvival(),
                         RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42, n_jobs=-1)),
           "RF 500 деревьев, max_depth=6, + Sex_Pclass, HasGroup, GroupSurvival"),
}


if __name__ == "__main__":
    X, y, _, _ = build()
    for name in sys.argv[1:] or EXPERIMENTS:
        model, description = EXPERIMENTS[name]
        result = evaluate(model, X, y)
        print(name, {k: round(float(v), 4) for k, v in result.items()})
        log_experiment(name, description, result)
