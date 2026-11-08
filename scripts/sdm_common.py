"""Data preparation shared by the SDM scripts (trata_dados() of the original SDM_XAI.ipynb).

MinMax scaling is fitted on the whole grid before the 70/30 split, as in the notebook that produced the
published figures; fitting it on the training split only changes the Thraupidae waterfall (f(x) = 0.493
instead of 0.534). Pseudo-absences: ClusterCentroids replaces the background cells of the training split by
K-Means centroids (presence:background = 1:2).
"""
from pathlib import Path

import pandas as pd
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OUTPUT = Path(__file__).resolve().parents[1] / "output"
FEATURES = ["CO (ppm)", "O3 (ppb)", "NOX (ppb)", "Acetonitrile (ppb)", "Isoprene (ppb)", "H2O (%)",
            "CH4 (ppm)", "Min Temperature (°C)", "Max Temperature (°C)"]
FAMILIES = {
    "Tyrannidae": ["Myiozetetes cayanensis", "Pitangus sulphuratus", "Todirostrum maculatum",
                   "Tyrannus melancholicus", "Tyrannus savana"],
    "Thraupidae": ["Ramphocelus carbo", "Sicalis columbiana", "Sporophila castaneiventris",
                   "Thraupis episcopus", "Thraupis palmarum"],
}


def load(csv_path):
    """Study matrix with one 0/1 column per family (presence of any of its five species)."""
    raw = pd.read_csv(csv_path)
    for family, species in FAMILIES.items():
        raw[family] = (raw[[f"Especie_{sp}" for sp in species]].sum(axis=1) > 0).astype(int)
    return raw


def prep(raw, family):
    """Returns X_test, y_test (Series, original index), X_train, y_train (resampled) and the scaled grid."""
    norm_X = MinMaxScaler().fit_transform(raw.iloc[:, 2:11])  # the 9 predictors, in the CSV order
    X_train, X_test, y_train, y_test = train_test_split(norm_X, raw[[family]], random_state=41, test_size=0.3)
    X_train, y_train = ClusterCentroids(random_state=42, sampling_strategy=0.5).fit_resample(X_train, y_train)
    return (pd.DataFrame(X_test, columns=FEATURES), y_test[family], pd.DataFrame(X_train, columns=FEATURES),
            y_train[family].to_numpy(), norm_X)
