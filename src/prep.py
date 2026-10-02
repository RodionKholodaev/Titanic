from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
             "Mr": "Mr", "Mrs": "Mrs", "Miss": "Miss", "Master": "Master"}
CATEGORICAL = ["Title", "Embarked", "Deck"]
DROP = ["PassengerId", "Name", "Ticket", "Cabin", "Surname", "Survived", "is_test"]


def load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return pd.concat([train.assign(is_test=False), test.assign(is_test=True)], ignore_index=True)


def extract_title(full):
    title = full["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return full.assign(Title=title.map(TITLE_MAP).fillna("Rare"))


def extract_surname(full):
    return full.assign(Surname=full["Name"].str.split(",").str[0].str.strip())


def group_features(full):
    ticket_freq = full.groupby("Ticket")["Ticket"].transform("count")
    family_size = full["SibSp"] + full["Parch"] + 1
    return full.assign(TicketFreq=ticket_freq, FamilySize=family_size,
                       IsAlone=(family_size == 1).astype(int),
                       FarePerPerson=full["Fare"] / ticket_freq)


def fill_missing(full):
    train = full[~full["is_test"]]
    age_med = full["Title"].map(train.groupby("Title")["Age"].median())
    fare_med = full["Pclass"].map(train.groupby("Pclass")["Fare"].median())
    return full.assign(Age=full["Age"].fillna(age_med),
                       Embarked=full["Embarked"].fillna(train["Embarked"].mode()[0]),
                       Fare=full["Fare"].fillna(fare_med))


def basic_features(full):
    return full.assign(Sex=(full["Sex"] == "female").astype(int),
                       HasCabin=full["Cabin"].notna().astype(int),
                       Deck=full["Cabin"].str[0].fillna("U"),
                       IsChild=(full["Age"] < 12).astype(int))


def build():
    full = load()
    for step in (extract_title, extract_surname, fill_missing, group_features, basic_features):
        full = step(full)
    X = pd.get_dummies(full.drop(columns=DROP), columns=CATEGORICAL, dtype=int)
    is_test = full["is_test"]
    return X[~is_test], full.loc[~is_test, "Survived"].astype(int), X[is_test], full
