from colorama import Fore, Style

from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Normalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def initialize_model(input_shape: tuple) -> Sequential:
    """
    Initialize the Neural Network with random weights.
    """

    model = Sequential()

    model.add(Input(shape=input_shape))
    model.add(Normalization())
    model.add(Dense(100, activation="relu", kernel_regularizer=regularizers.l1_l2(l2=0.005)))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(20, activation="relu"))
    model.add(Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model


def compile_model(model: Sequential, learning_rate: float = 0.0005) -> Sequential:
    """
    Compile the Neural Network.
    """

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mae"],
    )

    print("✅ Model compiled")

    return model


def train_model(
    model: Sequential,
    X=None,
    y=None,
    batch_size: int = 256,
    patience: int = 2,
    validation_data=None,
    validation_split: float = 0.3,
) -> tuple:
    """
    Fit the model and return a tuple: (fitted_model, history).

    The test suite calls this function with X=... and y=...
    """

    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=0 if validation_data is not None else validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )

    print(f"✅ Model trained on {len(X)} rows")

    return model, history
