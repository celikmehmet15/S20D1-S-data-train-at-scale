import numpy as np
import pandas as pd

import pygeohash as gh


def transform_time_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time features from pickup_datetime.
    """

    pickup_datetime = pd.to_datetime(X["pickup_datetime"], utc=True)

    time_features = pd.DataFrame(index=X.index)
    time_features["pickup_hour"] = pickup_datetime.dt.hour
    time_features["pickup_day_of_week"] = pickup_datetime.dt.dayofweek
    time_features["pickup_month"] = pickup_datetime.dt.month
    time_features["pickup_year"] = pickup_datetime.dt.year

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
    Transform longitude/latitude columns into a distance feature.
    """

    lonlat_features = pd.DataFrame(index=X.index)

    lonlat_features["distance"] = haversine_vectorized(
        X["pickup_longitude"],
        X["pickup_latitude"],
        X["dropoff_longitude"],
        X["dropoff_latitude"],
    )

    return lonlat_features


def compute_geohash(X: pd.DataFrame, precision: int = 4) -> pd.DataFrame:
    """
    Compute pickup and dropoff geohashes.
    """

    geohash_features = pd.DataFrame(index=X.index)

    geohash_features["pickup_geohash"] = X.apply(
        lambda row: gh.encode(
            row["pickup_latitude"],
            row["pickup_longitude"],
            precision=precision,
        ),
        axis=1,
    )

    geohash_features["dropoff_geohash"] = X.apply(
        lambda row: gh.encode(
            row["dropoff_latitude"],
            row["dropoff_longitude"],
            precision=precision,
        ),
        axis=1,
    )

    return geohash_features
