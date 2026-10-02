import csv

import pandas as pd

from run_experiment import GenderBaseline
from src.cv import CV, evaluate, log_experiment


def test_cv_is_fixed():
    assert CV.get_n_splits() == 25
    assert CV.random_state == 42


def test_gender_baseline_through_evaluate():
    train = pd.read_csv("data/train.csv")
    res = evaluate(GenderBaseline(), train.drop(columns="Survived"), train["Survived"])
    assert set(res) == {"cv_mean", "cv_std", "train_mean"}
    assert 0.78 < res["cv_mean"] < 0.79


def test_log_experiment_appends(tmp_path):
    path = tmp_path / "exp.csv"
    res = {"cv_mean": 0.5, "cv_std": 0.1, "train_mean": 0.6}
    log_experiment("a", "first", res, path)
    log_experiment("b", "second", res, path)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [r["name"] for r in rows] == ["a", "b"]
    assert rows[0]["cv_mean"] == "0.5"
