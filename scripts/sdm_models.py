"""LR, RF and SVM fits (70/30 hold-out grids) shared by the SDM scripts."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def _grid_search(estimator, grid, X, y):
    return GridSearchCV(estimator, grid, cv=3, scoring="accuracy", n_jobs=-1).fit(X, y).best_estimator_


def fit_lr(X, y):
    return _grid_search(LogisticRegression(class_weight="balanced", max_iter=1000),
                        {"C": [0.01, 0.1, 1, 10, 100]}, X, y)


def fit_rf(X, y, family):
    if family == "Thraupidae":
        clf = RandomForestClassifier(n_estimators=10, max_depth=5, max_features="sqrt", min_samples_leaf=5,
                                     min_samples_split=10, random_state=42, n_jobs=-1, class_weight="balanced")
        return clf.fit(X, y)
    grid = {"n_estimators": [10, 50], "max_depth": [5, 10], "min_samples_split": [5, 10],
            "min_samples_leaf": [3, 5], "max_features": ["sqrt", "log2"]}
    return _grid_search(RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"), grid, X, y)


def fit_svm(X, y):
    grid = [{"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
            {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]}]
    return _grid_search(SVC(random_state=42, probability=True, class_weight="balanced"), grid, X, y)
