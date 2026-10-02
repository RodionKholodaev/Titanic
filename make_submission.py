import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from run_experiment import EXPERIMENTS
from src.prep import build

OUT_DIR = Path(__file__).resolve().parent / "submissions"


if __name__ == "__main__":
    name = sys.argv[1]
    X_train, y_train, X_test, full = build()
    model = clone(EXPERIMENTS[name][0]).fit(X_train, y_train)
    sub = pd.DataFrame({"PassengerId": full.loc[full["is_test"], "PassengerId"],
                        "Survived": model.predict(X_test).astype(int)})
    OUT_DIR.mkdir(exist_ok=True)
    sub.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"saved submissions/{name}.csv, {len(sub)} rows")
