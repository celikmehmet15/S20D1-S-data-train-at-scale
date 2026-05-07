import pandas as pd

from colorama import Fore, Style


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw taxifare data by removing invalid fares, invalid passengers,
    invalid coordinates, and unrealistic trips.
    """

    df = df.copy()

    # Remove rows with missing values
    df = df.dropna()

    # Remove invalid fare amounts
    df = df[df["fare_amount"] > 0]

    # Remove invalid passenger counts
    df = df[df["passenger_count"] > 0]
    df = df[df["passenger_count"] <= 8]

    # Keep only plausible NYC longitude/latitude coordinates
    df = df[df["pickup_longitude"].between(-74.3, -73.7)]
    df = df[df["pickup_latitude"].between(40.5, 40.9)]
    df = df[df["dropoff_longitude"].between(-74.3, -73.7)]
    df = df[df["dropoff_latitude"].between(40.5, 40.9)]

    print("✅ data cleaned")

    return df
