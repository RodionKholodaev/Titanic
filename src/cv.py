
# файл для оценки моделей
import csv
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

# делим датасет на 5 фолдов, на 4х обучаем на 5м проверяем, 
# так 5 раз с разнымыми перемешиваниями
CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
LOG_PATH = Path(__file__).resolve().parent.parent / "experiments.csv"
LOG_FIELDS = ["date", "name", "description", "cv_mean", "cv_std", "train_mean"]


def evaluate(model, X, y):
    # обучаем модель 25 раз и записываем результат в res
    res = cross_validate(model, X, y, cv=CV, scoring="accuracy", return_train_score=True)
    
    """
    формат res
    res = {
        "fit_time":       array([...]),   # время обучения на каждом фолде
        "score_time":     array([...]),   # время предсказания на каждом фолде
        "test_score":     array([...]),   # accuracy на тестовой части (5×5 = 25 чисел)
        "train_score":    array([...]),   # accuracy на обучающей части (25 чисел)
    }    
    """
    return {
        "cv_mean": res["test_score"].mean(),
        "cv_std": res["test_score"].std(),
        "train_mean": res["train_score"].mean(),
    }

#Сохранение результатов эксперимента в csv файл
def log_experiment(name, description, result, path=LOG_PATH):
    new = not Path(path).exists() or Path(path).stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_FIELDS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), name, description,
                    *(round(result[k], 4) for k in LOG_FIELDS[3:])])
