import numpy as np
import pandas as pd

import pygeohash as gh


def transform_time_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Extract cyclical hour features from pickup_datetime.
    """

    pickup_datetime = pd.to_datetime(X["pickup_datetime"], utc=True)
    hour = pickup_datetime.dt.hour

    time_features = pd.DataFrame(index=X.index)
    time_features["pickup_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    time_features["pickup_hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return time_features


def haversine_vectorized(
    lon1: pd.Series,
    lat1: pd.Series,
    lon2: pd.Series,
    lat2: pd.Series,
) -> np.ndarray:
    """
    Compute haversine distance in km between pickup and dropoff coordinates.
    """

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2],
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371

    return c * r


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform longitude/latitude columns into distance feature.
    """

    lonlat_features = pd.DataFrame(index=X.index)

    lonlat_features["distance"] = haversine_vectorized(
        X["pickup_longitude"],
        X["pickup_latitude"],
        X["dropoff_longitude"],
        X["dropoff_latitude"],
    )

    return lonlat_features


def compute_geohash(X: pd.DataFrame, precision: int = 5) -> pd.DataFrame:
    """
    Compute a pickup-dropoff geohash pair as one categorical route feature.
    """

    X = X.copy()

    pickup_geohash = X.apply(
        lambda row: gh.encode(
            row["pickup_latitude"],
            row["pickup_longitude"],
            precision=precision,
        ),
        axis=1,
    )

    dropoff_geohash = X.apply(
        lambda row: gh.encode(
            row["dropoff_latitude"],
            row["dropoff_longitude"],
            precision=precision,
        ),
        axis=1,
    )

    geohash_features = pd.DataFrame(index=X.index)
    geohash_features["geohash_pair"] = pickup_geohash + "_" + dropoff_geohash

    return geohash_features
