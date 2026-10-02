from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
             "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master"}
CATEGORICAL = ["Title", "Embarked", "Deck", "Sex_Pclass"]
# Ticket остаётся в X: из него GroupSurvival строит признак внутри фолда
DROP = ["PassengerId", "Name", "Cabin", "Surname", "Survived", "is_test"]

# получаем train и test и склеиваем их 
# чтобы признаки которые считались по другим людям были корректны
def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return pd.concat([train.assign(is_test=False), test.assign(is_test=True)], ignore_index=True)

# получаем титул (Mr, ...) из имени, 
# редкие титулы, которые не в TITLE_MAP помечаем как Rare
def extract_title(full):
    title = full["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return full.assign(Title=title.map(TITLE_MAP).fillna("Rare"))

# достаем фамилию людей из датасета
# .str.strip() - уберает пробелы по краям
#  возвращяем новый датасет с колонкой Surname
def extract_surname(full):
    return full.assign(Surname=full["Name"].str.split(",").str[0].str.strip())


def group_features(full):
    ticket_freq = full.groupby("Ticket")["Ticket"].transform("count")
    family_size = full["SibSp"] + full["Parch"] + 1
    return full.assign(TicketFreq=ticket_freq, FamilySize=family_size,
                       IsAlone=(family_size == 1).astype(int),
                       HasGroup=(ticket_freq > 1).astype(int),
                       FarePerPerson=full["Fare"] / ticket_freq)


def fill_missing(full):
    train = full[~full["is_test"]]

    # смотрим медиану только по тестовым данным, чтобы не было утечки данных
    # train.groupby("Title")["Age"].median() - Series где индекс это Title
    # Используем map, чтобы как по словарю сопоставить Title средний возраст
    # Получаем Series в котором у каждого возраста есть значение - медиана по титулу
    age_med = full["Title"].map(train.groupby("Title")["Age"].median())
    # То же самое для Fare
    fare_med = full["Pclass"].map(train.groupby("Pclass")["Fare"].median())
    # Создаем новый датафрейм, в котором заполняем None
    return full.assign(Age=full["Age"].fillna(age_med),
                       Embarked=full["Embarked"].fillna(train["Embarked"].mode()[0]),
                       Fare=full["Fare"].fillna(fare_med))


def basic_features(full):
    return full.assign(Sex_Pclass=full["Sex"] + "_" + full["Pclass"].astype(str),
                       Sex=(full["Sex"] == "female").astype(int),
                       HasCabin=full["Cabin"].notna().astype(int),
                       Deck=full["Cabin"].str[0].fillna("U"),
                       IsChild=(full["Age"] < 12).astype(int))


def build():
    full = load()
    for step in (extract_title, extract_surname, fill_missing, group_features, basic_features):
        full = step(full)
    # one-hot encoding
    X = pd.get_dummies(full.drop(columns=DROP), columns=CATEGORICAL, dtype=int)

    is_test = full["is_test"]

    X_train = X[~is_test]
    y_train = full.loc[~is_test, "Survived"].astype(int)
    X_test = X[is_test]
    
    return X_train, y_train, X_test, full


class GroupSurvival(BaseEstimator, TransformerMixin):
    """Заменяет Ticket на долю выживших среди остальных пассажиров с тем же билетом.

    Статистики берутся только из fit (train-часть фолда); для строк из fit
    собственная метка исключается (leave-one-out). Нет данных -> 0.5.
    """

    def fit(self, X, y):
        self.y_ = pd.Series(y.to_numpy(), index=X.index)
        grouped = self.y_.groupby(X["Ticket"].to_numpy())
        self.sum_, self.count_ = grouped.sum(), grouped.count()
        return self

    def transform(self, X):
        own = self.y_.reindex(X.index)
        total = X["Ticket"].map(self.sum_).fillna(0) - own.fillna(0)
        known = X["Ticket"].map(self.count_).fillna(0) - own.notna()
        return X.drop(columns="Ticket").assign(GroupSurvival=(total / known.where(known > 0)).fillna(0.5))
