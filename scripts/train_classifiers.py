"""Classifier comparison of the manuscript (table:metrics): LR, RF and SVM per family on the 30% hold-out.

  cd article-code
  python -m scripts.train_classifiers path/to/study_matrix.csv

Writes output/metrics.json with the obtained and the published values side by side.
"""
import json
import sys

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from scripts.sdm_common import FAMILIES, OUTPUT, load, prep
from scripts.sdm_models import fit_lr, fit_rf, fit_svm

PUBLISHED = {  # accuracy, recall, ROC-AUC
    "Thraupidae": {"lr": (0.603, 0.568, 0.727), "rf": (0.865, 0.814, 0.922), "svm": (0.813, 0.973, 0.924)},
    "Tyrannidae": {"lr": (0.632, 0.723, 0.631), "rf": (0.947, 0.979, 0.996), "svm": (0.926, 0.979, 0.962)},
}


def scores(clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    return {"accuracy": accuracy_score(y_test, pred), "recall": recall_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba)}


def main():
    raw = load(sys.argv[1])
    out = {}
    for family in FAMILIES:
        X_test, y_test, X_train, y_train, _ = prep(raw, family)
        models = {"lr": fit_lr(X_train, y_train), "rf": fit_rf(X_train, y_train, family),
                  "svm": fit_svm(X_train, y_train)}
        out[family] = {}
        for name, clf in models.items():
            got = scores(clf, X_test, y_test)
            out[family][name] = {
                "obtained": got,
                "published": dict(zip(got.keys(), PUBLISHED[family][name])),
            }
            print(f"{family} {name}: " + ", ".join(
                f"{k} {v:.3f} (published {p:.3f})" for (k, v), p in zip(got.items(), PUBLISHED[family][name])))
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
