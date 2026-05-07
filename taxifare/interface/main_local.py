import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from taxifare.params import *
from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, save_results, load_model
from taxifare.ml_logic.model import compile_model, initialize_model, train_model


def preprocess_and_train(min_date: str = "2009-01-01", max_date: str = "2015-01-01") -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    min_date = parse(min_date).strftime("%Y-%m-%d")
    max_date = parse(max_date).strftime("%Y-%m-%d")

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}`
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """

    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath(
        "raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv"
    )
    data_query_cache_path.parent.mkdir(parents=True, exist_ok=True)

    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")

        data = pd.read_csv(
            data_query_cache_path,
            parse_dates=["pickup_datetime"],
            dtype={
                key: value
                for key, value in DTYPES_RAW.items()
                if key != "pickup_datetime"
            },
        )

    else:
        print("Loading data from Querying Big Query server...")

        client = bigquery.Client(project=GCP_PROJECT, location=BQ_REGION)
        query_job = client.query(query, location=BQ_REGION)
        result = query_job.result()

        data = result.to_dataframe()

        data.to_csv(data_query_cache_path, header=True, index=False)

    data = clean_data(data)

    split_ratio = 0.02
    train_length = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:train_length, :].sample(frac=1).reset_index(drop=True)
    data_val = data.iloc[train_length:, :].sample(frac=1).reset_index(drop=True)

    X_train = data_train.drop("fare_amount", axis=1)
    y_train = data_train[["fare_amount"]]

    X_val = data_val.drop("fare_amount", axis=1)
    y_val = data_val[["fare_amount"]]

    X_processed = preprocess_features(pd.concat([X_train, X_val], axis=0))
    X_train_processed = X_processed[: len(X_train)]
    X_val_processed = X_processed[len(X_train):]

    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    model = initialize_model(input_shape=X_train_processed.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model=model,
        X=X_train_processed,
        y=y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val),
    )

    val_mae = np.min(history.history["val_mae"])

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(
            dict(
                pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz="UTC")],
                pickup_longitude=[-73.950655],
                pickup_latitude=[40.783282],
                dropoff_longitude=[-73.984365],
                dropoff_latitude=[40.769802],
                passenger_count=[1],
            )
        )

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("✅ pred() done")

    return y_pred


if __name__ == "__main__":
    try:
        preprocess_and_train()
        pred()
    except:
        import sys
        import traceback

        import ipdb

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
