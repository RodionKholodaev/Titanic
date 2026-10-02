import pytest

from src.prep import build, extract_title, load


@pytest.fixture(scope="module")
def built():
    return build()


def test_no_nan(built):
    X_train, _, X_test, _ = built
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()


def test_same_columns(built):
    X_train, _, X_test, _ = built
    assert list(X_train.columns) == list(X_test.columns)


def test_shapes(built):
    X_train, y_train, X_test, _ = built
    assert len(X_train) == len(y_train) == 891
    assert len(X_test) == 418


def test_age_medians_from_train_only(built):
    *_, full = built
    raw = extract_title(load())
    train = raw[~raw["is_test"]]
    expected = {t: train.loc[train["Title"] == t, "Age"].median() for t in train["Title"].unique()}
    missing = raw["Age"].isna()
    filled = full.loc[missing, "Age"]
    assert (filled == raw.loc[missing, "Title"].map(expected)).all()


def test_no_target_in_features(built):
    X_train, _, X_test, _ = built
    assert not any("Survived" in c for c in X_train.columns)
    assert not any("Survived" in c for c in X_test.columns)


def test_titles(built):
    *_, full = built
    assert set(full["Title"]) == {"Mr", "Mrs", "Miss", "Master", "Rare"}


def test_ticket_freq_uses_full(built):
    *_, full = built
    train = full[~full["is_test"]]
    train_freq = train.groupby("Ticket")["Ticket"].transform("count")
    assert (train["TicketFreq"] > train_freq).any()
