"""
Daily load-shape clustering.

Implements Specification Section 19 (Clustering) and Section 20
(Cluster Evaluation). STATISTICAL method (Section 48).

Primary method: K-means on complete-day profile vectors (one vector per
day, length = intervals_per_day), documented here as the initial choice
because (a) daily profiles are fixed-length numeric vectors so K-means'
Euclidean-distance assumption is directly applicable without a resampling
step, and (b) it is deterministic given a fixed random_state, satisfying
Section 45 reproducibility. Hierarchical/density-based/shape-distance
methods are documented extension points (Section 19) not implemented in
Stage 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _pivot_complete_days(
    daily_profiles: pd.DataFrame, value_col: str, entity_id: str
) -> tuple[pd.DataFrame, list]:
    """Pivot to (date x interval_index) matrix, complete days only."""
    sub = daily_profiles[daily_profiles["entity_id"] == entity_id]
    complete_dates = sub[sub["is_complete_day"]]["date"].unique()
    sub = sub[sub["date"].isin(complete_dates)]
    pivot = sub.pivot_table(index="date", columns="interval_index", values=value_col)
    pivot = pivot.dropna(axis=0, how="any")  # a complete day should have no NaN, but guard
    return pivot, list(pivot.index)


def cluster_daily_profiles(
    daily_profiles: pd.DataFrame,
    entity_id: str,
    value_col: str = "demand_kw",
    n_clusters: int | str = "auto",
    max_k: int = 8,
    random_state: int = 42,
) -> dict:
    """
    Cluster complete-day profiles for one entity.

    Parameters
    ----------
    daily_profiles:
        Output of profiles.construct_daily_profiles /
        normalize_daily_profiles (must contain is_complete_day,
        interval_index, and value_col).
    entity_id:
        Entity to cluster (meter, group, or "Portfolio").
    value_col:
        "demand_kw" (absolute clustering) or "normalized_demand"
        (normalized clustering) -- Section 19 requires both to be
        available; call this function twice with each value_col.
    n_clusters:
        "auto" (select via silhouette score over 2..max_k) or an
        explicit int.
    random_state:
        Fixed for reproducibility (Section 45).

    Returns
    -------
    dict with keys:
        entity_id, value_col, n_clusters, dates (list, in pivot order),
        labels (np.ndarray aligned with dates), silhouette (float or
        NaN if only 1 cluster / <3 points), cluster_summary (DataFrame:
        cluster_id, cluster_size, percentage_of_days,
        representative_peak_kw or representative_peak_normalized,
        within_cluster_variability), representative_profiles (dict
        cluster_id -> np.ndarray of length intervals_per_day, the
        cluster centroid), success (bool).

    Algorithm
    ---------
    1. Pivot complete days into a (n_days x n_intervals) matrix.
    2. If n_clusters == "auto": try k in 2..min(max_k, n_days-1),
       fit KMeans, score silhouette, pick the k maximizing it. If fewer
       than 4 complete days exist, auto-selection is skipped and
       n_clusters is forced to 1 (not enough data to cluster
       meaningfully -- Section 20: "must not create clusters with no
       analytical value").
    3. Fit final KMeans(n_clusters, random_state, n_init=10).

    Assumptions
    -----------
    Requires >= 2 complete days to run at all; returns success=False
    with an empty result otherwise.
    """
    pivot, dates = _pivot_complete_days(daily_profiles, value_col, entity_id)
    n_days = len(pivot)

    if n_days < 2:
        return {
            "entity_id": entity_id, "value_col": value_col, "n_clusters": 0,
            "dates": [], "labels": np.array([]), "silhouette": np.nan,
            "cluster_summary": pd.DataFrame(), "representative_profiles": {},
            "success": False,
        }

    X = pivot.values

    if n_clusters == "auto":
        if n_days < 4:
            k = 1
        else:
            best_k, best_score = 1, -1.0
            for k_try in range(2, min(max_k, n_days - 1) + 1):
                km = KMeans(n_clusters=k_try, random_state=random_state, n_init=10)
                labels_try = km.fit_predict(X)
                if len(set(labels_try)) < 2:
                    continue
                score = silhouette_score(X, labels_try)
                if score > best_score:
                    best_k, best_score = k_try, score
            k = best_k
    else:
        k = int(n_clusters)

    if k <= 1:
        labels = np.zeros(n_days, dtype=int)
        centroids = X.mean(axis=0, keepdims=True)
        silhouette = np.nan
    else:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_
        silhouette = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else np.nan

    summary_records = []
    representative_profiles = {}
    for c in range(k if k > 0 else 1):
        members = labels == c
        size = int(members.sum())
        cluster_days = X[members]
        within_var = float(np.mean(np.std(cluster_days, axis=0))) if size > 0 else np.nan
        summary_records.append(
            {
                "cluster_id": c,
                "cluster_size": size,
                "percentage_of_days": round(100.0 * size / n_days, 1) if n_days else np.nan,
                "representative_peak": float(np.max(centroids[c])) if size > 0 else np.nan,
                "within_cluster_variability": within_var,
            }
        )
        representative_profiles[c] = centroids[c]

    return {
        "entity_id": entity_id,
        "value_col": value_col,
        "n_clusters": k,
        "dates": dates,
        "labels": labels,
        "silhouette": silhouette,
        "cluster_summary": pd.DataFrame.from_records(summary_records),
        "representative_profiles": representative_profiles,
        "success": True,
    }
