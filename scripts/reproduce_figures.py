"""Habitat map, occurrence map and SHAP figures for both families + shap_metadata cache for NLE.

  cd article-code && python -m scripts.reproduce_figures path/to/study_matrix.csv
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar
from pyproj import Transformer

from scripts.sdm_common import FAMILIES, FEATURES, OUTPUT, load, prep
from scripts.sdm_models import fit_rf, fit_svm

FIG = OUTPUT / "figures"
SHAP_DIR = FIG / "shap"
MAP_DIR = FIG / "maps"
CACHE = OUTPUT / "shap_cache"
COLORS = {"Tyrannidae": "#E87722", "Thraupidae": "#2D6A4F"}
MARKERS = ["o", "s", "^", "D", "v"]
STEP_LAT, STEP_LON = 0.002, 0.001
NE_URL = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
DPI = 300
THRAUPIDAE_WATERFALL = 1000


def save(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close("all")


def shap_plots(clf, X_test, i, tag):
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)[:, :, 1]
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    plt.xlabel("mean(|SHAP value|)\n(average impact on model output magnitude)")
    save(SHAP_DIR / f"shap_avg_{tag}.png")
    shap.summary_plot(sv, X_test, show=False)
    save(SHAP_DIR / f"shap_beeswarm_{tag}.png")
    plt.figure()
    exp = shap.Explanation(values=sv[i], base_values=explainer.expected_value[1], data=X_test.iloc[i],
                           feature_names=FEATURES)
    shap.waterfall_plot(exp, max_display=len(FEATURES), show=False)
    plt.title("SHAP Waterfall Plot - Presence")
    plt.tight_layout()
    save(SHAP_DIR / f"shap_waterfall_{tag}.png")
    shap.dependence_plot("NOX (ppb)", sv, X_test, interaction_index="O3 (ppb)", show=False)
    save(SHAP_DIR / f"shap_dependence_{tag}.png")
    waterfall = [{"feature": FEATURES[j], "shap": float(sv[i, j]), "value": float(X_test.iloc[i, j])}
                 for j in range(len(FEATURES))]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    meta = {
        "family": tag,
        "expected_value": float(explainer.expected_value[1]),
        "instance_index": i,
        "instance_proba": float(clf.predict_proba(X_test.iloc[[i]])[0, 1]),
        "feature_ranking": [FEATURES[j] for j in order],
        "mean_abs_shap": {FEATURES[j]: float(mean_abs[j]) for j in range(len(FEATURES))},
        "waterfall": waterfall,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    return meta


def merc(lon, lat):
    return TO_3857.transform(np.asarray(lon), np.asarray(lat))


def grid_of(raw, values):
    lat0, lon0 = raw.Latitude.min(), raw.Longitude.min()
    rows = np.rint((raw.Latitude - lat0) / STEP_LAT).astype(int)
    cols = np.rint((raw.Longitude - lon0) / STEP_LON).astype(int)
    arr = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    arr[rows, cols] = values
    (x0, x1), (y0, y1) = merc([lon0 - STEP_LON / 2, raw.Longitude.max() + STEP_LON / 2],
                              [lat0 - STEP_LAT / 2, raw.Latitude.max() + STEP_LAT / 2])
    return arr, [x0, x1, y0, y1]


def base_axes(raw, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    (x0, x1), (y0, y1) = merc([raw.Longitude.min() - 0.02, raw.Longitude.max() + 0.02],
                              [raw.Latitude.min() - 0.02, raw.Latitude.max() + 0.02])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, crs="EPSG:3857", attribution_size=5)
    ax.add_artist(ScaleBar(np.cos(np.radians(raw.Latitude.mean())), units="m", location="lower right",
                           length_fraction=0.2, box_alpha=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def study_outline(ax, raw, color="black"):
    arr, extent = grid_of(raw, 1.0)
    ax.contour(np.isfinite(arr).astype(float), levels=[0.5], colors=color, linewidths=1.0,
               extent=extent, origin="lower")
    return Line2D([], [], color=color, linewidth=1.0, label="GoAmazon interpolated grid (study area)")


def habitat_map(raw, proba):
    arr, extent = grid_of(raw, proba)
    fig, ax = base_axes(raw, (8, 7.5))
    im = ax.imshow(arr, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest", zorder=2)
    handle = study_outline(ax, raw)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Habitat suitability (SVM predicted probability of presence)")
    ax.legend(handles=[handle], loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title("Thraupidae (family level), Support Vector Machine", fontsize=10)
    save(MAP_DIR / "habitat_suitability_revised.png")


def occurrence_map(raw, instances):
    fig, ax = base_axes(raw, (10, 7.5))
    handles = [study_outline(ax, raw, color="#555555")]
    for family, species in FAMILIES.items():
        n_family = int(raw[family].sum())
        handles.append(Line2D([], [], linestyle="none", label=f"$\\it{{{family}}}$ (N = {n_family})"))
        for sp, marker in zip(species, MARKERS):
            pts = raw[raw[f"Especie_{sp}"] == 1]
            x, y = merc(pts.Longitude, pts.Latitude)
            ax.scatter(x, y, s=28, marker=marker, color=COLORS[family], edgecolor="white", linewidth=0.5, zorder=3)
            genus, epithet = sp.split()
            handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=6,
                                  markerfacecolor=COLORS[family], markeredgecolor="white",
                                  label=f"   $\\it{{{genus[0]}.\\ {epithet}}}$ (N = {len(pts)})"))
    for family, (lat, lon) in instances.items():
        x, y = merc(lon, lat)
        ax.scatter(x, y, s=260, marker="*", color=COLORS[family], edgecolor="black", linewidth=1.2, zorder=4)
        handles.append(Line2D([], [], linestyle="none", marker="*", markersize=14,
                              markerfacecolor=COLORS[family], markeredgecolor="black",
                              label=f"Local SHAP instance (waterfall plot), $\\it{{{family}}}$"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    world = gpd.read_file(NE_URL)
    sa = world[world["CONTINENT"] == "South America"]
    ins = ax.inset_axes([0.0, 0.66, 0.30, 0.34])
    sa.plot(ax=ins, color="#e6e6e6", edgecolor="white", linewidth=0.3)
    sa[sa["ADMIN"] == "Brazil"].plot(ax=ins, color="#bdbdbd", edgecolor="white", linewidth=0.3)
    w, e, s, n = raw.Longitude.min(), raw.Longitude.max(), raw.Latitude.min(), raw.Latitude.max()
    ins.add_patch(Rectangle((w - 1, s - 1), e - w + 2, n - s + 2, fill=False, edgecolor="#c0392b", linewidth=1.5))
    ins.text(-52, -12, "Brazil", fontsize=7, ha="center", color="#333333")
    ins.set_xlim(-82, -34)
    ins.set_ylim(-56, 13)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_facecolor("white")
    for spine in ins.spines.values():
        spine.set_edgecolor("#555555")
    save(MAP_DIR / "occurrence_points_revised.png")


def main():
    raw = load(sys.argv[1])
    instances = {}

    X_test, y_test, X_train, y_train, norm_X = prep(raw, "Thraupidae")
    clf_thr = fit_rf(X_train, y_train, "Thraupidae")
    base = shap.TreeExplainer(clf_thr).expected_value[1]
    fx = clf_thr.predict_proba(X_test.iloc[[THRAUPIDAE_WATERFALL]])[0, 1]
    assert abs(base - 0.504) < 0.005 and abs(fx - 0.534) < 0.005, "Thraupidae waterfall check failed (Fig. 7)"
    lat, lon = raw.loc[y_test.index[THRAUPIDAE_WATERFALL], ["Latitude", "Longitude"]]
    instances["Thraupidae"] = (float(lat), float(lon))
    shap_plots(clf_thr, X_test, THRAUPIDAE_WATERFALL, "thraupidae")
    habitat_map(raw, fit_svm(X_train, y_train).predict_proba(norm_X)[:, 1])

    X_test, y_test, X_train, y_train, _ = prep(raw, "Tyrannidae")
    clf_tyr = fit_rf(X_train, y_train, "Tyrannidae")
    proba = clf_tyr.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    i = int(np.flatnonzero((y == 1) & (proba > 0.5))[0])
    lat, lon = raw.loc[y_test.index[i], ["Latitude", "Longitude"]]
    instances["Tyrannidae"] = (float(lat), float(lon))
    shap_plots(clf_tyr, X_test, i, "tyrannidae")
    occurrence_map(raw, instances)


if __name__ == "__main__":
    main()
