# ==============================================================
# Stream Water-Level Prediction Using Daily Precipitation
# Predicts the Next 14 Days
#
# Input CSV structure:
# Column A: date
# Column B: wl in m
# Column C: precip in mm
#
# Author: Babak / ChatGPT
# ==============================================================

from pathlib import Path
import html
import json
import os
import struct
import warnings
from urllib.parse import quote, urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==============================================================
# 1. USER SETTINGS
# ==============================================================

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "input_huron.csv"

OUTPUT_DIR = PROJECT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WATERSHED_GEOJSON_FILE = PROJECT_DIR / "Watershed Shapefile" / "Huron River" / "data.geojson"
WATERSHED_SHAPEFILE_DIR = None
OUTPUT_PREFIX = ""
WATERSHED_SLUG = "huron"
WATERSHED_NAME = "Huron River"
WATERSHED_FULL_NAME = "Huron River Watershed, Western Lake Erie Basin, Ohio"
DASHBOARD_TITLE = "Live 14-Day Flood Forecast for Huron River Watershed, Western Lake Erie Basin, Ohio"
WATER_LEVEL_AXIS_LABEL = "Water-level in Huron River (m)"
FUTURE_CLIMATE_FILE = PROJECT_DIR / "Precipitation" / "Huron River" / "huron_future.csv"
HISTORICAL_CLIMATE_FILE = PROJECT_DIR / "Precipitation" / "Huron River" / "huron_historical.csv"


def versioned_output_file(filename):
    """
    Returns the output path for files created by the model.
    """

    if OUTPUT_PREFIX:
        return OUTPUT_DIR / f"{OUTPUT_PREFIX}_{filename}"

    return OUTPUT_DIR / filename

FORECAST_DAYS = 14
FORECAST_START = "today"  # Use "today" or "tomorrow"
STAGE_PLOT_START_DATE = "2025-01-01"
OPEN_METEO_PAST_DAYS = 14
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_TIMEOUT_SECONDS = 60

BASIN_FORECAST_POINTS = [
    (41.2923, -82.6158),
    (41.3031, -82.6845),
    (41.2628, -82.6769),
    (41.2613, -82.7415),
    (41.2484, -82.7284),
    (41.2277, -82.6934),
    (41.1745, -82.8074),
    (41.1337, -82.8445),
    (41.0881, -82.8060),
    (41.2706, -82.5980),
    (41.2277, -82.6591),
    (41.1781, -82.6556),
    (40.9675, -82.7181),
    (41.1042, -82.5492),
    (41.0602, -82.7339),
    (41.1600, -82.5547),
    (41.0804, -82.6659),
    (41.1668, -82.6344),
]

HURON_FORECAST_POINTS = BASIN_FORECAST_POINTS

SANDUSKY_FORECAST_POINTS = [
    (41.2871, -83.1967),
    (41.2282, -83.2544),
    (41.2107, -83.2173),
    (41.1052, -83.3519),
    (41.0286, -83.2420),
    (41.1952, -83.1047),
    (41.0390, -83.1239),
    (40.9965, -82.9262),
    (41.1073, -83.1734),
    (40.9457, -83.3107),
    (40.9457, -83.1459),
    (40.9166, -83.0113),
    (40.8481, -83.2503),
    (40.8200, -83.0978),
    (40.6504, -83.3052),
    (40.8065, -82.7614),
    (40.9062, -83.1418),
]

WATERSHEDS = [
    {
        "slug": "huron",
        "output_prefix": "huron",
        "name": "Huron River",
        "full_name": "Huron River Watershed, Western Lake Erie Basin, Ohio",
        "dashboard_title": "Live 14-Day Flood Forecast for Huron River Watershed, Western Lake Erie Basin, Ohio",
        "axis_label": "Water-level in Huron River (m)",
        "input_file": PROJECT_DIR / "input_huron.csv",
        "geojson_file": PROJECT_DIR / "Watershed Shapefile" / "Huron River" / "data.geojson",
        "shapefile_dir": None,
        "forecast_points": HURON_FORECAST_POINTS,
        "future_climate_file": PROJECT_DIR / "Precipitation" / "Huron River" / "huron_future.csv",
        "historical_climate_file": PROJECT_DIR / "Precipitation" / "Huron River" / "huron_historical.csv",
        "stage_thresholds_m": {
            "action": 4.2672,
            "minor": 5.1816,
            "moderate": 5.6388,
            "major": 6.4008,
        },
    },
    {
        "slug": "sandusky",
        "output_prefix": "sandusky",
        "name": "Sandusky River",
        "full_name": "Sandusky River Watershed, Western Lake Erie Basin, Ohio",
        "dashboard_title": "Live 14-Day Flood Forecast for Sandusky River Watershed, Western Lake Erie Basin, Ohio",
        "axis_label": "Water-level in Sandusky River (m)",
        "input_file": PROJECT_DIR / "input_sandusky.csv",
        "geojson_file": None,
        "shapefile_dir": PROJECT_DIR / "Watershed Shapefile" / "Sandusky River" / "download" / "layers",
        "forecast_points": SANDUSKY_FORECAST_POINTS,
        "future_climate_file": PROJECT_DIR / "Precipitation" / "Sandusky River" / "sandusky_future.csv",
        "historical_climate_file": PROJECT_DIR / "Precipitation" / "Sandusky River" / "sandusky_historical.csv",
        "stage_thresholds_m": {
            "action": 2.1,
            "minor": 3.0,
            "moderate": 4.0,
            "major": 6.4,
        },
    },
]

# Feature settings
MAX_PRECIP_LAG = 14      # precipitation memory, days
MAX_WL_LAG = 7           # water-level memory, days
ROLLING_WINDOWS = [3, 7, 14, 30]

# Chronological split
TEST_FRACTION = 0.20

RANDOM_STATE = 42

# Deep-learning sequence model settings
DEEP_SEQUENCE_LENGTH = 14
DEEP_LEARNING_EPOCHS = 45
DEEP_LEARNING_BATCH_SIZE = 64
DEEP_LEARNING_PATIENCE = 6
DEEP_LEARNING_UNITS = 32

# Water-level stage thresholds, meters
ACTION_STAGE_M = 4.2672
MINOR_FLOOD_STAGE_M = 5.1816
MODERATE_FLOOD_STAGE_M = 5.6388
MAJOR_FLOOD_STAGE_M = 6.4008

FORECAST_SCENARIOS = [
    {
        "name": "10% runoff reduction",
        "column_prefix": "runoff_reduction_10pct",
        "kind": "multiply",
        "value": 0.90,
        "plot_color": "#2e7d32",
        "plot_style": "--",
    },
    {
        "name": "20% runoff reduction",
        "column_prefix": "runoff_reduction_20pct",
        "kind": "multiply",
        "value": 0.80,
        "plot_color": "#00695c",
        "plot_style": "-.",
    },
    {
        "name": "2.5 mm storage",
        "column_prefix": "storage_2_5mm",
        "kind": "storage",
        "value": 2.5,
        "plot_color": "#6a1b9a",
        "plot_style": "--",
    },
    {
        "name": "5 mm storage",
        "column_prefix": "storage_5mm",
        "kind": "storage",
        "value": 5.0,
        "plot_color": "#ef6c00",
        "plot_style": "-.",
    },
]


# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def nash_sutcliffe_efficiency(obs, pred):
    """
    Nash-Sutcliffe Efficiency.
    NSE = 1 is perfect.
    NSE = 0 means model is only as good as the observed mean.
    NSE < 0 means model is worse than using the observed mean.
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((obs - pred) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - numerator / denominator


def clean_input_data(input_file):
    """
    Reads the CSV and standardizes columns:
    date, wl_m, precip_mm
    """

    df = pd.read_csv(input_file)

    if df.shape[1] < 3:
        raise ValueError(
            "The input CSV must have at least three columns: date, water level, precipitation."
        )

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    if {"date", "precip_mm", "daily_water_level_m"}.issubset(normalized_columns):
        df = df[[
            normalized_columns["date"],
            normalized_columns["daily_water_level_m"],
            normalized_columns["precip_mm"],
        ]].copy()
        df.columns = ["date", "wl_m", "precip_mm"]
    else:
        # Use first three columns regardless of their original names
        df = df.iloc[:, :3].copy()
        df.columns = ["date", "wl_m", "precip_mm"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["wl_m"] = pd.to_numeric(df["wl_m"], errors="coerce")
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    # If duplicate dates exist, average water level and sum precipitation
    df = (
        df.groupby("date", as_index=False)
        .agg({
            "wl_m": "mean",
            "precip_mm": "sum"
        })
    )

    # Create continuous daily date range
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_dates)
    df.index.name = "date"
    df = df.reset_index()

    # Fill missing precipitation with 0.
    # If missing precipitation means "missing data" rather than "no rain",
    # you may want to change this later.
    df["precip_mm"] = df["precip_mm"].fillna(0)

    return df


def add_features(df):
    """
    Creates hydrologically meaningful predictors.

    Target variable:
    wl_m on the current day.

    Predictors include:
    - precipitation on current and previous days
    - rolling precipitation totals
    - previous water levels
    - rolling previous water level means
    - seasonal sine/cosine variables
    """

    df = df.copy()

    # Calendar features
    df["dayofyear"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month

    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Precipitation lag features
    # precip_lag_0 = precipitation on the prediction day
    for lag in range(0, MAX_PRECIP_LAG + 1):
        df[f"precip_lag_{lag}"] = df["precip_mm"].shift(lag)

    # Rolling precipitation totals including current day
    for window in ROLLING_WINDOWS:
        df[f"precip_sum_{window}d"] = (
            df["precip_mm"]
            .rolling(window=window, min_periods=window)
            .sum()
        )

    # Water-level lag features
    # wl_lag_1 = yesterday's water level
    # We do NOT use today's water level to predict today's water level.
    for lag in range(1, MAX_WL_LAG + 1):
        df[f"wl_lag_{lag}"] = df["wl_m"].shift(lag)

    # Rolling previous water-level means
    for window in [3, 7, 14]:
        df[f"wl_mean_previous_{window}d"] = (
            df["wl_m"]
            .shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
        )

    # Optional: water-level change from yesterday to day before yesterday
    df["wl_change_1d"] = df["wl_lag_1"] - df["wl_lag_2"]

    return df


def get_feature_columns(df):
    """
    Returns all feature columns used by the model.
    """

    feature_cols = []

    feature_cols += [f"precip_lag_{lag}" for lag in range(0, MAX_PRECIP_LAG + 1)]
    feature_cols += [f"precip_sum_{window}d" for window in ROLLING_WINDOWS]
    feature_cols += [f"wl_lag_{lag}" for lag in range(1, MAX_WL_LAG + 1)]
    feature_cols += [f"wl_mean_previous_{window}d" for window in [3, 7, 14]]

    feature_cols += [
        "wl_change_1d",
        "sin_doy",
        "cos_doy",
        "month",
    ]

    return feature_cols


def train_test_split_time_series(model_df, test_fraction):
    """
    Chronological split.
    No random splitting for time-series hydrology data.
    """

    n = len(model_df)
    test_size = int(np.ceil(n * test_fraction))
    train_size = n - test_size

    train_df = model_df.iloc[:train_size].copy()
    test_df = model_df.iloc[train_size:].copy()

    return train_df, test_df


def evaluate_model(obs, pred):
    """
    Calculates model performance metrics.
    """

    mae = mean_absolute_error(obs, pred)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    r2 = r2_score(obs, pred)
    nse = nash_sutcliffe_efficiency(obs, pred)

    return {
        "MAE_m": mae,
        "RMSE_m": rmse,
        "R2": r2,
        "NSE": nse
    }


def evaluate_prediction_column(test_df, prediction_column):
    """
    Calculates metrics for one prediction column after dropping missing predictions.
    """

    evaluation_df = test_df[["wl_m", prediction_column]].dropna()

    if evaluation_df.empty:
        raise ValueError(f"No complete rows available to evaluate {prediction_column}.")

    return evaluate_model(evaluation_df["wl_m"], evaluation_df[prediction_column])


def load_keras_tools():
    """
    Imports TensorFlow/Keras only when LSTM or GRU models are trained.
    """

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras import callbacks, layers, optimizers
    except ImportError as exc:
        raise ImportError(
            "LSTM and GRU models require TensorFlow. Install the packages in "
            "requirements.txt before running version_9.py."
        ) from exc

    return tf, Sequential, callbacks, layers, optimizers


def make_feature_sequences(feature_values, sequence_length):
    """
    Converts daily feature rows into rolling sequences for recurrent models.
    """

    feature_values = np.asarray(feature_values, dtype=float)

    if len(feature_values) < sequence_length:
        return (
            np.empty((0, sequence_length, feature_values.shape[1])),
            np.array([], dtype=int),
        )

    sequences = []
    row_positions = []

    for end_position in range(sequence_length - 1, len(feature_values)):
        start_position = end_position - sequence_length + 1
        sequences.append(feature_values[start_position:end_position + 1])
        row_positions.append(end_position)

    return np.asarray(sequences), np.asarray(row_positions, dtype=int)


def build_deep_sequence_model(model_kind, input_shape):
    """
    Builds either an LSTM or GRU sequence regressor.
    """

    _, Sequential, _, layers, optimizers = load_keras_tools()
    recurrent_layer = {
        "lstm": layers.LSTM,
        "gru": layers.GRU,
    }.get(model_kind)

    if recurrent_layer is None:
        raise ValueError(f"Unknown deep-learning model type: {model_kind}")

    model = Sequential([
        layers.Input(shape=input_shape),
        recurrent_layer(DEEP_LEARNING_UNITS),
        layers.Dropout(0.15),
        layers.Dense(24, activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )

    return model


def train_deep_sequence_model(model_kind, model_df, feature_cols, train_end_position=None):
    """
    Trains an LSTM or GRU model using chronological sequence data.
    """

    tf, _, callbacks, _, _ = load_keras_tools()
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    if train_end_position is None:
        train_end_position = len(model_df) - 1

    training_df = model_df.iloc[:train_end_position + 1].copy()
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(training_df[feature_cols])
    target_scaler.fit(training_df[["wl_m"]])

    feature_values = feature_scaler.transform(model_df[feature_cols])
    target_values = target_scaler.transform(model_df[["wl_m"]]).ravel()
    X_sequences, row_positions = make_feature_sequences(feature_values, DEEP_SEQUENCE_LENGTH)

    train_mask = row_positions <= train_end_position
    X_train_sequences = X_sequences[train_mask]
    y_train_sequences = target_values[row_positions[train_mask]]

    if len(X_train_sequences) == 0:
        raise ValueError(
            f"Not enough training rows to build {DEEP_SEQUENCE_LENGTH}-day sequences "
            f"for the {model_kind.upper()} model."
        )

    validation_data = None
    if len(X_train_sequences) >= 40:
        validation_size = max(1, int(len(X_train_sequences) * 0.15))
        X_fit = X_train_sequences[:-validation_size]
        y_fit = y_train_sequences[:-validation_size]
        validation_data = (
            X_train_sequences[-validation_size:],
            y_train_sequences[-validation_size:],
        )
    else:
        X_fit = X_train_sequences
        y_fit = y_train_sequences

    model = build_deep_sequence_model(
        model_kind=model_kind,
        input_shape=(DEEP_SEQUENCE_LENGTH, len(feature_cols)),
    )

    callback_list = []
    if validation_data is not None:
        callback_list.append(
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=DEEP_LEARNING_PATIENCE,
                restore_best_weights=True,
            )
        )

    model.fit(
        X_fit,
        y_fit,
        validation_data=validation_data,
        epochs=DEEP_LEARNING_EPOCHS,
        batch_size=DEEP_LEARNING_BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=callback_list,
    )

    return {
        "kind": model_kind,
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sequence_length": DEEP_SEQUENCE_LENGTH,
    }


def predict_deep_sequence_model(model_bundle, model_df, feature_cols):
    """
    Predicts all rows that have enough prior sequence context.
    """

    feature_values = model_bundle["feature_scaler"].transform(model_df[feature_cols])
    X_sequences, row_positions = make_feature_sequences(
        feature_values,
        model_bundle["sequence_length"],
    )

    if len(X_sequences) == 0:
        return pd.Series(dtype=float)

    scaled_predictions = model_bundle["model"].predict(X_sequences, verbose=0).reshape(-1, 1)
    predictions = (
        model_bundle["target_scaler"]
        .inverse_transform(scaled_predictions)
        .ravel()
    )

    return pd.Series(predictions, index=model_df.index[row_positions])


def classify_water_level_stage(water_level_m):
    """
    Converts a predicted water level into the requested flood-stage category.
    """

    if water_level_m < ACTION_STAGE_M:
        return "Action stage"
    if water_level_m < MINOR_FLOOD_STAGE_M:
        return "Minor flood stage"
    if water_level_m < MODERATE_FLOOD_STAGE_M:
        return "Moderate flood stage"
    if water_level_m < MAJOR_FLOOD_STAGE_M:
        return "Major flood stage"

    return "Above major flood stage"


def plot_observed_vs_predicted(test_df, output_file):
    """
    Creates time-series plot for observed vs predicted water level.
    """

    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()
    ax2.bar(
        test_df["date"],
        test_df["precip_mm"],
        color="lightskyblue",
        alpha=0.45,
        label="Precipitation",
        width=0.8
    )
    ax.plot(test_df["date"], test_df["wl_m"], label="Observed", linewidth=2)
    ax.plot(test_df["date"], test_df["predicted_wl_m"], label="Predicted (Random Forest)", linewidth=2)

    if "lstm_predicted_wl_m" in test_df.columns:
        ax.plot(
            test_df["date"],
            test_df["lstm_predicted_wl_m"],
            label="Predicted (LSTM)",
            linewidth=2,
            linestyle=":",
        )

    if "gru_predicted_wl_m" in test_df.columns:
        ax.plot(
            test_df["date"],
            test_df["gru_predicted_wl_m"],
            label="Predicted (GRU)",
            linewidth=2,
            linestyle="-.",
        )

    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_alpha(0)
    ax.set_xlabel("Date")
    ax.set_ylabel(WATER_LEVEL_AXIS_LABEL)
    ax2.set_ylabel("Precipitation (mm)")
    ax2.yaxis.label.set_color("#111111")
    ax2.tick_params(axis="y", colors="#111111")
    ax.set_title("Observed vs Predicted Stream Water Level")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close()


def plot_stage_forecast_summary(
    historical_df,
    test_df,
    forecast_df,
    output_file,
    scenario_forecasts=None
):
    """
    Creates a USGS-style stage plot with observed, estimated, and forecast water levels.
    """

    plot_start = pd.Timestamp(STAGE_PLOT_START_DATE)

    observed_df = historical_df[historical_df["date"] >= plot_start].copy()
    estimated_df = test_df[test_df["date"] >= plot_start].copy()
    forecast_plot_df = forecast_df.copy()
    scenario_forecasts = scenario_forecasts or []

    scenario_max_values = [
        scenario["forecast_df"]["predicted_wl_m"].max()
        for scenario in scenario_forecasts
    ]
    forecast_model_columns = [
        column
        for column in [
            "predicted_wl_m",
            "lstm_predicted_wl_m",
            "gru_predicted_wl_m",
        ]
        if column in forecast_plot_df.columns
    ]
    estimated_model_columns = [
        column
        for column in [
            "predicted_wl_m",
            "lstm_predicted_wl_m",
            "gru_predicted_wl_m",
        ]
        if column in estimated_df.columns
    ]
    forecast_model_max_values = [
        forecast_plot_df[column].max()
        for column in forecast_model_columns
    ]
    estimated_model_max_values = [
        estimated_df[column].max()
        for column in estimated_model_columns
    ]

    series_max = max(
        observed_df["wl_m"].max(),
        *estimated_model_max_values,
        *forecast_model_max_values,
        *scenario_max_values,
        MAJOR_FLOOD_STAGE_M,
    )
    series_min = min(
        observed_df["wl_m"].min(),
        *[
            forecast_plot_df[column].min()
            for column in forecast_model_columns
        ],
    )
    y_min = max(0, series_min - 0.3)
    y_max = series_max + 0.5

    fig, ax = plt.subplots(figsize=(16, 8))

    stage_bands = [
        (y_min, ACTION_STAGE_M, "#ffffff", "Action stage"),
        (ACTION_STAGE_M, MINOR_FLOOD_STAGE_M, "#fff9e8", "Minor flood stage"),
        (MINOR_FLOOD_STAGE_M, MODERATE_FLOOD_STAGE_M, "#fde8e6", "Moderate flood stage"),
        (MODERATE_FLOOD_STAGE_M, MAJOR_FLOOD_STAGE_M, "#f7eafa", "Major flood stage"),
        (MAJOR_FLOOD_STAGE_M, y_max, "#ead3f5", "Above major flood stage"),
    ]

    for lower, upper, color, _ in stage_bands:
        ax.axhspan(lower, upper, color=color, zorder=0)

    stage_lines = [
        (ACTION_STAGE_M, "#f6d84a", "Action stage"),
        (MINOR_FLOOD_STAGE_M, "#f39c12", "Minor flood stage"),
        (MODERATE_FLOOD_STAGE_M, "#ff2f20", "Moderate flood stage"),
        (MAJOR_FLOOD_STAGE_M, "#d05cff", "Major flood stage"),
    ]

    for threshold, color, label in stage_lines:
        ax.axhline(threshold, color=color, linewidth=2, zorder=1)
        ax.text(
            0.01,
            threshold + 0.03,
            f"{threshold:.2f} m - {label}",
            transform=ax.get_yaxis_transform(),
            color="#111111",
            fontsize=10,
            va="bottom",
        )

    ax.plot(
        observed_df["date"],
        observed_df["wl_m"],
        color="#173a8a",
        linewidth=2.0,
        label="Observed",
        zorder=3,
    )
    ax.plot(
        estimated_df["date"],
        estimated_df["predicted_wl_m"],
        color="#8b1e1e",
        linestyle="--",
        linewidth=2.0,
        label="Estimated (Random Forest)",
        zorder=4,
    )
    if "lstm_predicted_wl_m" in estimated_df.columns:
        ax.plot(
            estimated_df["date"],
            estimated_df["lstm_predicted_wl_m"],
            color="#5b2c83",
            linestyle=":",
            linewidth=2.0,
            label="Estimated (LSTM)",
            zorder=4,
        )
    if "gru_predicted_wl_m" in estimated_df.columns:
        ax.plot(
            estimated_df["date"],
            estimated_df["gru_predicted_wl_m"],
            color="#006d77",
            linestyle="-.",
            linewidth=2.0,
            label="Estimated (GRU)",
            zorder=4,
        )
    ax.plot(
        forecast_plot_df["date"],
        forecast_plot_df["predicted_wl_m"],
        color="#0b5d1e",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Forecast water-level",
        zorder=5,
    )
    if "lstm_predicted_wl_m" in forecast_plot_df.columns:
        ax.plot(
            forecast_plot_df["date"],
            forecast_plot_df["lstm_predicted_wl_m"],
            color="#5b2c83",
            linestyle=":",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="LSTM forecast water-level",
            zorder=5,
        )
    if "gru_predicted_wl_m" in forecast_plot_df.columns:
        ax.plot(
            forecast_plot_df["date"],
            forecast_plot_df["gru_predicted_wl_m"],
            color="#006d77",
            linestyle="-.",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="GRU forecast water-level",
            zorder=5,
        )

    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"]
        ax.plot(
            scenario_df["date"],
            scenario_df["predicted_wl_m"],
            color=scenario["plot_color"],
            linestyle=scenario["plot_style"],
            linewidth=2.0,
            marker="o",
            markersize=3,
            label=scenario["name"],
            zorder=5,
        )

    ax.axvline(
        forecast_plot_df["date"].min(),
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_title("Observed, Estimated, and Forecast Stream Water Level", fontsize=16, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(WATER_LEVEL_AXIS_LABEL)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, color="#d7d7d7", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def plot_scatter_observed_vs_predicted(test_df, output_file):
    """
    Creates scatter plot for observed vs predicted water level.
    """

    obs = test_df["wl_m"]
    pred = test_df["predicted_wl_m"]

    prediction_columns = [
        column
        for column in [
            "predicted_wl_m",
            "lstm_predicted_wl_m",
            "gru_predicted_wl_m",
        ]
        if column in test_df.columns
    ]
    prediction_values = pd.concat([test_df[column] for column in prediction_columns])

    min_val = min(obs.min(), prediction_values.min())
    max_val = max(obs.max(), prediction_values.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(obs, pred, alpha=0.6, label="Random Forest")
    if "lstm_predicted_wl_m" in test_df.columns:
        ax.scatter(
            obs,
            test_df["lstm_predicted_wl_m"],
            alpha=0.55,
            label="LSTM",
        )
    if "gru_predicted_wl_m" in test_df.columns:
        ax.scatter(
            obs,
            test_df["gru_predicted_wl_m"],
            alpha=0.55,
            label="GRU",
        )
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    ax.set_xlabel("Observed water level (m)")
    ax.set_ylabel("Predicted water level (m)")
    ax.set_title("Observed vs Predicted Water Level")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    plt.close()


def display_table_column_name(column):
    """
    Converts output dataframe column names into dashboard table labels.
    """

    labels = {
        "date": "Date",
        "forecast_day": "Forecast Day",
        "forecast_precip_mm": "Forecast Precip (mm)",
        "predicted_wl_m": "Predicted WL (m)",
        "water_level_stage": "Water Level Stage",
        "lstm_predicted_wl_m": "LSTM Predicted WL (m)",
        "lstm_water_level_stage": "LSTM Water Level Stage",
        "gru_predicted_wl_m": "GRU Predicted WL (m)",
        "gru_water_level_stage": "GRU Water Level Stage",
        "runoff_reduction_10pct_predicted_wl_m": "Runoff Reduction 10% Predicted WL (m)",
        "runoff_reduction_10pct_water_level_stage": "Runoff Reduction 10% Water Level Stage",
        "runoff_reduction_20pct_predicted_wl_m": "Runoff Reduction 20% Predicted WL (m)",
        "runoff_reduction_20pct_water_level_stage": "Runoff Reduction 20% Water Level Stage",
        "storage_2_5mm_predicted_wl_m": "Storage 2.5 mm Predicted WL (m)",
        "storage_2_5mm_water_level_stage": "Storage 2.5 mm Water Level Stage",
        "storage_5mm_predicted_wl_m": "Storage 5 mm Predicted WL (m)",
        "storage_5mm_water_level_stage": "Storage 5 mm Water Level Stage",
    }

    return labels.get(column, column.replace("_", " ").title())


def round_numeric_columns(df, decimals=2):
    """
    Rounds numeric output columns while preserving dates and text categories.
    """

    rounded_df = df.copy()
    numeric_columns = rounded_df.select_dtypes(include=[np.number]).columns
    rounded_df[numeric_columns] = rounded_df[numeric_columns].round(decimals)
    return rounded_df


def dataframe_to_html_table(df, columns):
    """
    Builds an HTML table body from selected dataframe columns.
    """

    header_cells = "".join(
        f"<th>{html.escape(display_table_column_name(column))}</th>"
        for column in columns
    )
    rows = []

    for _, row in df.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                display_value = ""
            elif isinstance(value, pd.Timestamp):
                display_value = value.strftime("%Y-%m-%d")
            elif "date" in column:
                display_value = pd.Timestamp(value).strftime("%Y-%m-%d")
            elif isinstance(value, (float, np.floating)):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            cells.append(f"<td>{html.escape(display_value)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"<thead><tr>{header_cells}</tr></thead><tbody>{''.join(rows)}</tbody>"


def series_for_interactive_chart(name, dates, values, color, dash="solid"):
    """
    Creates a Plotly trace dictionary with consistent hover text.
    """

    return {
        "type": "scatter",
        "mode": "lines+markers",
        "name": name,
        "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in dates],
        "y": [None if pd.isna(value) else round(float(value), 2) for value in values],
        "line": {
            "color": color,
            "width": 2.5,
            "dash": dash,
        },
        "marker": {
            "size": 5,
        },
        "hovertemplate": (
            "<b>%{fullData.name}</b><br>"
            "Date: %{x}<br>"
            "Water level: %{y:.2f} m"
            "<extra></extra>"
        ),
    }


def long_date_label(date):
    """
    Formats dates for dashboard axes without a leading zero on the day.
    """

    timestamp = pd.Timestamp(date)
    return f"{timestamp.strftime('%B')} {timestamp.day}, {timestamp.year}"


def load_watershed_geojson():
    """
    Loads the watershed shapefile export used by the interactive map.
    """

    if WATERSHED_GEOJSON_FILE is None or not WATERSHED_GEOJSON_FILE.exists():
        if WATERSHED_SHAPEFILE_DIR is None:
            return None
        return load_watershed_shapefile_geojson(WATERSHED_SHAPEFILE_DIR)

    with WATERSHED_GEOJSON_FILE.open("r", encoding="utf-8") as geojson_file:
        return json.load(geojson_file)


def read_shapefile_features(shp_file):
    """
    Reads simple point and polygon shapefiles into GeoJSON features.
    """

    features = []
    if not shp_file or not shp_file.exists():
        return features

    data = shp_file.read_bytes()
    offset = 100

    while offset + 8 <= len(data):
        _, record_length_words = struct.unpack(">2i", data[offset:offset + 8])
        offset += 8
        record_length = record_length_words * 2
        record = data[offset:offset + record_length]
        offset += record_length

        if len(record) < 4:
            continue

        shape_type = struct.unpack("<i", record[:4])[0]

        if shape_type == 0:
            continue

        if shape_type == 1 and len(record) >= 20:
            x, y = struct.unpack("<2d", record[4:20])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": {},
            })

        if shape_type == 5 and len(record) >= 44:
            num_parts, num_points = struct.unpack("<2i", record[36:44])
            parts_start = 44
            parts_end = parts_start + (num_parts * 4)
            points_start = parts_end

            if len(record) < points_start + (num_points * 16):
                continue

            parts = list(struct.unpack(f"<{num_parts}i", record[parts_start:parts_end]))
            points = []
            for point_index in range(num_points):
                start = points_start + (point_index * 16)
                points.append(list(struct.unpack("<2d", record[start:start + 16])))

            rings = []
            for part_index, part_start in enumerate(parts):
                part_end = parts[part_index + 1] if part_index + 1 < len(parts) else num_points
                ring = points[part_start:part_end]
                if ring and ring[0] != ring[-1]:
                    ring.append(ring[0])
                if len(ring) >= 4:
                    rings.append(ring)

            if rings:
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": {},
                })

    return features


def load_watershed_shapefile_geojson(shapefile_dir):
    """
    Builds GeoJSON from watershed polygon and outlet point shapefiles.
    """

    features = []
    features.extend(read_shapefile_features(shapefile_dir / "globalwatershed.shp"))
    features.extend(read_shapefile_features(shapefile_dir / "globalwatershedpoint.shp"))

    if not features:
        return None

    coordinates = []
    for feature in features:
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Point":
            coordinates.append(geometry["coordinates"])
        elif geometry.get("type") == "Polygon":
            for ring in geometry.get("coordinates", []):
                coordinates.extend(ring)

    geojson = {"type": "FeatureCollection", "features": features}
    if coordinates:
        xs = [point[0] for point in coordinates]
        ys = [point[1] for point in coordinates]
        geojson["bbox"] = [min(xs), min(ys), max(xs), max(ys)]

    return geojson


def load_future_climate_summary():
    """
    Aggregates future daily precipitation and mean temperature into annual values.
    """

    if FUTURE_CLIMATE_FILE is None or not FUTURE_CLIMATE_FILE.exists():
        return pd.DataFrame(columns=["year", "annual_precip_mm", "avg_tmean_c"])

    climate_df = pd.read_csv(FUTURE_CLIMATE_FILE)

    required_columns = {"date", "precip_mm_day", "tmean_C"}
    missing_columns = required_columns - set(climate_df.columns)
    if missing_columns:
        raise ValueError(
            f"Future climate file {FUTURE_CLIMATE_FILE} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    climate_df = climate_df[["date", "precip_mm_day", "tmean_C"]].copy()
    climate_df["date"] = pd.to_datetime(climate_df["date"], errors="coerce")
    climate_df["precip_mm_day"] = pd.to_numeric(climate_df["precip_mm_day"], errors="coerce")
    climate_df["tmean_C"] = pd.to_numeric(climate_df["tmean_C"], errors="coerce")
    climate_df = climate_df.dropna(subset=["date"])

    daily_df = (
        climate_df
        .groupby("date", as_index=False)
        .agg(
            precip_mm_day=("precip_mm_day", "mean"),
            tmean_C=("tmean_C", "mean"),
        )
    )
    daily_df["year"] = daily_df["date"].dt.year

    annual_df = (
        daily_df
        .groupby("year", as_index=False)
        .agg(
            annual_precip_mm=("precip_mm_day", "sum"),
            avg_tmean_c=("tmean_C", "mean"),
        )
        .sort_values("year")
    )

    round_numeric_columns(annual_df).to_csv(
        versioned_output_file("future_annual_precip_temperature.csv"),
        index=False,
        float_format="%.2f"
    )

    return annual_df


def build_future_climate_chart():
    """
    Returns Plotly traces and layout for annual future precipitation and temperature.
    """

    annual_df = load_future_climate_summary()

    if annual_df.empty:
        return [], {}

    annual_df = annual_df[
        (annual_df["year"] >= 2026) &
        (annual_df["year"] <= 2100)
    ].copy()

    if annual_df.empty:
        return [], {}

    years = [int(year) for year in annual_df["year"]]
    annual_precip = [
        None if pd.isna(value) else round(float(value), 2)
        for value in annual_df["annual_precip_mm"]
    ]
    avg_tmean = [
        None if pd.isna(value) else round(float(value), 2)
        for value in annual_df["avg_tmean_c"]
    ]

    traces = [
        {
            "type": "bar",
            "name": "Annual precipitation",
            "x": years,
            "y": annual_precip,
            "marker": {"color": "rgba(31, 78, 121, 0.55)"},
            "hovertemplate": (
                "<b>Annual precipitation</b><br>"
                "Year: %{x}<br>"
                "Precipitation: %{y:.2f} mm"
                "<extra></extra>"
            ),
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Average mean temperature",
            "x": years,
            "y": avg_tmean,
            "yaxis": "y2",
            "line": {"color": "#8b1e1e", "width": 3},
            "marker": {"size": 6, "color": "#8b1e1e"},
            "hovertemplate": (
                "<b>Average mean temperature</b><br>"
                "Year: %{x}<br>"
                "Temperature: %{y:.2f} C"
                "<extra></extra>"
            ),
        },
    ]

    layout = {
        "margin": {"l": 96, "r": 96, "t": 82, "b": 86},
        "height": 600,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 15},
        },
        "xaxis": {
            "title": "",
            "range": [2025.5, 2100.5],
            "dtick": 10,
            "tickfont": {"size": 15, "color": "#111111"},
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": {
                "text": "Annual precipitation (mm)",
                "font": {"size": 18, "color": "#111111"},
            },
            "tickfont": {"size": 15, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis2": {
            "title": {
                "text": "Average mean temperature (C)",
                "font": {"size": 18, "color": "#111111"},
            },
            "overlaying": "y",
            "side": "right",
            "tickfont": {"size": 15, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": False,
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "zeroline": False,
        },
    }

    return traces, layout


def load_daily_climate_file(climate_file):
    """
    Loads daily precipitation and mean temperature climate data.
    """

    if climate_file is None or not climate_file.exists():
        return pd.DataFrame(columns=["date", "precip_mm_day", "tmean_C"])

    climate_df = pd.read_csv(climate_file)
    required_columns = {"date", "precip_mm_day", "tmean_C"}
    missing_columns = required_columns - set(climate_df.columns)
    if missing_columns:
        raise ValueError(
            f"Climate file {climate_file} is missing columns: {sorted(missing_columns)}"
        )

    climate_df = climate_df[["date", "precip_mm_day", "tmean_C"]].copy()
    climate_df["date"] = pd.to_datetime(climate_df["date"], errors="coerce")
    climate_df["precip_mm_day"] = pd.to_numeric(climate_df["precip_mm_day"], errors="coerce")
    climate_df["tmean_C"] = pd.to_numeric(climate_df["tmean_C"], errors="coerce")
    climate_df = climate_df.dropna(subset=["date"])

    return (
        climate_df
        .groupby("date", as_index=False)
        .agg(
            precip_mm_day=("precip_mm_day", "mean"),
            tmean_C=("tmean_C", "mean"),
        )
    )


def summarize_monthly_climate(climate_df):
    """
    Creates 12 monthly mean/min/max values from month-year averages.
    """

    if climate_df.empty:
        return pd.DataFrame(columns=[
            "month",
            "month_name",
            "precip_mean",
            "precip_min",
            "precip_max",
            "tmean_mean",
            "tmean_min",
            "tmean_max",
        ])

    working_df = climate_df.copy()
    working_df["year"] = working_df["date"].dt.year
    working_df["month"] = working_df["date"].dt.month

    month_year_df = (
        working_df
        .groupby(["year", "month"], as_index=False)
        .agg(
            precip_monthly_mean=("precip_mm_day", "mean"),
            tmean_monthly_mean=("tmean_C", "mean"),
        )
    )

    monthly_summary = (
        month_year_df
        .groupby("month", as_index=False)
        .agg(
            precip_mean=("precip_monthly_mean", "mean"),
            precip_min=("precip_monthly_mean", "min"),
            precip_max=("precip_monthly_mean", "max"),
            tmean_mean=("tmean_monthly_mean", "mean"),
            tmean_min=("tmean_monthly_mean", "min"),
            tmean_max=("tmean_monthly_mean", "max"),
        )
        .sort_values("month")
    )
    monthly_summary["month_name"] = monthly_summary["month"].apply(
        lambda month: pd.Timestamp(year=2000, month=int(month), day=1).strftime("%B")
    )

    return monthly_summary


def build_monthly_climate_comparison_charts():
    """
    Builds monthly historical-vs-future precipitation and temperature charts.
    """

    historical_daily_df = load_daily_climate_file(HISTORICAL_CLIMATE_FILE)
    future_daily_df = load_daily_climate_file(FUTURE_CLIMATE_FILE)
    historical_daily_df = historical_daily_df[
        (historical_daily_df["date"].dt.year >= 1970) &
        (historical_daily_df["date"].dt.year <= 2010)
    ].copy()
    future_daily_df = future_daily_df[
        (future_daily_df["date"].dt.year >= 2026) &
        (future_daily_df["date"].dt.year <= 2100)
    ].copy()

    historical_summary = summarize_monthly_climate(historical_daily_df)
    future_summary = summarize_monthly_climate(future_daily_df)

    if historical_summary.empty or future_summary.empty:
        return [], {}, [], {}

    combined_summary = historical_summary.merge(
        future_summary,
        on=["month", "month_name"],
        how="outer",
        suffixes=("_historical", "_future"),
    ).sort_values("month")

    round_numeric_columns(combined_summary).to_csv(
        versioned_output_file("monthly_historical_future_climate_summary.csv"),
        index=False,
        float_format="%.2f"
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    combined_summary["month_name"] = pd.Categorical(
        combined_summary["month_name"],
        categories=month_order,
        ordered=True,
    )
    combined_summary = combined_summary.sort_values("month_name")
    months = combined_summary["month_name"].astype(str).tolist()

    def values(column):
        return [
            None if pd.isna(value) else round(float(value), 2)
            for value in combined_summary[column]
        ]

    precipitation_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Historical range (1970-2010)",
            "x": months,
            "y": values("precip_min_historical"),
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Historical variation range (1970-2010)",
            "x": months,
            "y": values("precip_max_historical"),
            "fill": "tonexty",
            "fillcolor": "rgba(198, 18, 24, 0.18)",
            "line": {"width": 0},
            "hovertemplate": "Historical max (1970-2010): %{y:.2f} mm/day<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Future range (2026-2100)",
            "x": months,
            "y": values("precip_min_future"),
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Future variation range (2026-2100)",
            "x": months,
            "y": values("precip_max_future"),
            "fill": "tonexty",
            "fillcolor": "rgba(42, 124, 190, 0.22)",
            "line": {"width": 0},
            "hovertemplate": "Future max (2026-2100): %{y:.2f} mm/day<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Historical (1970-2010)",
            "x": months,
            "y": values("precip_mean_historical"),
            "line": {"color": "#c61218", "width": 4},
            "marker": {"size": 7, "color": "#c61218"},
            "hovertemplate": "Historical (1970-2010): %{y:.2f} mm/day<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Future (2026-2100)",
            "x": months,
            "y": values("precip_mean_future"),
            "line": {"color": "#2a7cbe", "width": 4},
            "marker": {"size": 7, "color": "#2a7cbe"},
            "hovertemplate": "Future (2026-2100): %{y:.2f} mm/day<extra></extra>",
        },
    ]

    temperature_traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Historical range (1970-2010)",
            "x": months,
            "y": values("tmean_min_historical"),
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Historical variation range (1970-2010)",
            "x": months,
            "y": values("tmean_max_historical"),
            "fill": "tonexty",
            "fillcolor": "rgba(198, 18, 24, 0.18)",
            "line": {"width": 0},
            "hovertemplate": "Historical max (1970-2010): %{y:.2f} C<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Future range (2026-2100)",
            "x": months,
            "y": values("tmean_min_future"),
            "line": {"width": 0},
            "showlegend": False,
            "hoverinfo": "skip",
        },
        {
            "type": "scatter",
            "mode": "lines",
            "name": "Future variation range (2026-2100)",
            "x": months,
            "y": values("tmean_max_future"),
            "fill": "tonexty",
            "fillcolor": "rgba(42, 124, 190, 0.22)",
            "line": {"width": 0},
            "hovertemplate": "Future max (2026-2100): %{y:.2f} C<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Historical (1970-2010)",
            "x": months,
            "y": values("tmean_mean_historical"),
            "line": {"color": "#c61218", "width": 4},
            "marker": {"size": 7, "color": "#c61218"},
            "hovertemplate": "Historical (1970-2010): %{y:.2f} C<extra></extra>",
        },
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": "Future (2026-2100)",
            "x": months,
            "y": values("tmean_mean_future"),
            "line": {"color": "#2a7cbe", "width": 4},
            "marker": {"size": 7, "color": "#2a7cbe"},
            "hovertemplate": "Future (2026-2100): %{y:.2f} C<extra></extra>",
        },
    ]

    base_layout = {
        "margin": {"l": 96, "r": 42, "t": 88, "b": 130},
        "height": 600,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.10,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 15},
        },
        "xaxis": {
            "title": "",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": month_order,
            "range": [0, 11],
            "tickangle": -45,
            "tickfont": {"size": 15, "color": "#111111"},
            "showgrid": True,
            "gridcolor": "#edf1f6",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis": {
            "tickfont": {"size": 15, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": True,
            "gridcolor": "#edf1f6",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
    }

    precipitation_layout = {
        **base_layout,
        "yaxis": {
            **base_layout["yaxis"],
            "title": {
                "text": "Average monthly precipitation (mm/day)",
                "font": {"size": 18, "color": "#111111"},
            },
        },
    }

    temperature_layout = {
        **base_layout,
        "yaxis": {
            **base_layout["yaxis"],
            "title": {
                "text": "Average monthly mean temperature (C)",
                "font": {"size": 18, "color": "#111111"},
            },
        },
    }

    return precipitation_traces, precipitation_layout, temperature_traces, temperature_layout


def write_interactive_index(
    historical_df,
    test_df,
    forecast_df,
    forecast_output_df,
    scenario_forecasts,
    metrics,
    output_file
):
    """
    Writes an interactive HTML dashboard with hoverable chart and forecast table.
    """

    plot_start = pd.Timestamp(STAGE_PLOT_START_DATE)
    observed_df = historical_df[historical_df["date"] >= plot_start].copy()
    estimated_df = test_df[test_df["date"] >= plot_start].copy()

    model_test_traces = [
        {
            "type": "bar",
            "name": "Precipitation",
            "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in observed_df["date"]],
            "y": [None if pd.isna(value) else round(float(value), 2) for value in observed_df["precip_mm"]],
            "yaxis": "y2",
            "marker": {"color": "rgba(135, 206, 250, 0.58)"},
            "opacity": 0.58,
            "hovertemplate": (
                "<b>Precipitation</b><br>"
                "Date: %{x}<br>"
                "Precipitation: %{y:.2f} mm"
                "<extra></extra>"
            ),
        },
        series_for_interactive_chart(
            "Observed",
            observed_df["date"],
            observed_df["wl_m"],
            "#173a8a",
            "solid"
        ),
        series_for_interactive_chart(
            "Estimated (Random Forest)",
            estimated_df["date"],
            estimated_df["predicted_wl_m"],
            "#8b1e1e",
            "dash"
        ),
    ]

    if "lstm_predicted_wl_m" in estimated_df.columns:
        model_test_traces.append(
            series_for_interactive_chart(
                "Estimated (LSTM)",
                estimated_df["date"],
                estimated_df["lstm_predicted_wl_m"],
                "#5b2c83",
                "dot"
            )
        )

    if "gru_predicted_wl_m" in estimated_df.columns:
        model_test_traces.append(
            series_for_interactive_chart(
                "Estimated (GRU)",
                estimated_df["date"],
                estimated_df["gru_predicted_wl_m"],
                "#006d77",
                "dashdot"
            )
        )

    forecast_traces = [
        {
            "type": "bar",
            "name": "Forecast precipitation",
            "x": [pd.Timestamp(date).strftime("%Y-%m-%d") for date in forecast_df["date"]],
            "y": [None if pd.isna(value) else round(float(value), 2) for value in forecast_df["forecast_precip_mm"]],
            "yaxis": "y2",
            "marker": {"color": "rgba(135, 206, 250, 0.58)"},
            "opacity": 0.58,
            "hovertemplate": (
                "<b>Forecast precipitation</b><br>"
                "Date: %{x}<br>"
                "Precipitation: %{y:.2f} mm"
                "<extra></extra>"
            ),
        },
        series_for_interactive_chart(
            "Forecast water-level",
            forecast_df["date"],
            forecast_df["predicted_wl_m"],
            "#0b5d1e",
            "solid"
        ),
    ]

    if "lstm_predicted_wl_m" in forecast_df.columns:
        forecast_traces.append(
            series_for_interactive_chart(
                "LSTM forecast water-level",
                forecast_df["date"],
                forecast_df["lstm_predicted_wl_m"],
                "#5b2c83",
                "dot"
            )
        )

    if "gru_predicted_wl_m" in forecast_df.columns:
        forecast_traces.append(
            series_for_interactive_chart(
                "GRU forecast water-level",
                forecast_df["date"],
                forecast_df["gru_predicted_wl_m"],
                "#006d77",
                "dashdot"
            )
        )

    for scenario in scenario_forecasts:
        forecast_traces.append(
            series_for_interactive_chart(
                scenario["name"],
                scenario["forecast_df"]["date"],
                scenario["forecast_df"]["predicted_wl_m"],
                scenario["plot_color"],
                "dashdot" if scenario["plot_style"] == "-." else "dash"
            )
        )

    y_values = []
    for trace in forecast_traces + model_test_traces:
        if trace.get("yaxis") == "y2":
            continue
        y_values.extend(value for value in trace["y"] if value is not None)

    y_min = max(0, min(y_values) - 0.3)
    y_max = max(max(y_values), MAJOR_FLOOD_STAGE_M) + 0.5
    precipitation_values = forecast_df["forecast_precip_mm"].dropna()
    precipitation_y_max = max(1.0, float(precipitation_values.max()) * 1.2) if not precipitation_values.empty else 1.0
    model_precipitation_values = observed_df["precip_mm"].dropna()
    model_precipitation_y_max = (
        max(1.0, float(model_precipitation_values.max()) * 1.2)
        if not model_precipitation_values.empty
        else 1.0
    )
    forecast_tick_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in forecast_df["date"]
    ]
    forecast_tick_labels = [
        long_date_label(date)
        for date in forecast_df["date"]
    ]
    forecast_x_range = [
        pd.Timestamp(forecast_df["date"].min()).strftime("%Y-%m-%d"),
        pd.Timestamp(forecast_df["date"].max()).strftime("%Y-%m-%d"),
    ]
    model_test_x_range = [
        min(observed_df["date"].min(), estimated_df["date"].min()).strftime("%Y-%m-%d"),
        max(observed_df["date"].max(), estimated_df["date"].max()).strftime("%Y-%m-%d"),
    ]
    model_test_months = pd.date_range(
        start=min(observed_df["date"].min(), estimated_df["date"].min()).to_period("M").to_timestamp(),
        end=max(observed_df["date"].max(), estimated_df["date"].max()).to_period("M").to_timestamp(),
        freq="MS",
    )
    model_test_tick_dates = [
        pd.Timestamp(date).strftime("%Y-%m-%d")
        for date in model_test_months
    ]
    model_test_tick_labels = [
        pd.Timestamp(date).strftime("%b %Y")
        for date in model_test_months
    ]

    stage_shapes = [
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": y_min,
            "y1": ACTION_STAGE_M,
            "fillcolor": "#ffffff",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": ACTION_STAGE_M,
            "y1": MINOR_FLOOD_STAGE_M,
            "fillcolor": "#fff9e8",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MINOR_FLOOD_STAGE_M,
            "y1": MODERATE_FLOOD_STAGE_M,
            "fillcolor": "#fde8e6",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MODERATE_FLOOD_STAGE_M,
            "y1": MAJOR_FLOOD_STAGE_M,
            "fillcolor": "#f7eafa",
            "line": {"width": 0},
            "layer": "below",
        },
        {
            "type": "rect",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": MAJOR_FLOOD_STAGE_M,
            "y1": y_max,
            "fillcolor": "#ead3f5",
            "line": {"width": 0},
            "layer": "below",
        },
    ]

    stage_lines = [
        (ACTION_STAGE_M, "#f6d84a", "Action stage"),
        (MINOR_FLOOD_STAGE_M, "#f39c12", "Minor flood stage"),
        (MODERATE_FLOOD_STAGE_M, "#ff2f20", "Moderate flood stage"),
        (MAJOR_FLOOD_STAGE_M, "#d05cff", "Major flood stage"),
    ]

    for threshold, color, _ in stage_lines:
        stage_shapes.append({
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": threshold,
            "y1": threshold,
            "line": {"color": color, "width": 2},
        })

    stage_annotations = [
        {
            "xref": "paper",
            "x": 0.01,
            "yref": "y",
            "y": threshold + 0.03,
            "text": f"{threshold:.2f} m - {label}",
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 12, "color": "#1f2933"},
            "bgcolor": "rgba(255,255,255,0.55)",
        }
        for threshold, _, label in stage_lines
    ]

    base_chart_layout = {
        "margin": {"l": 128, "r": 96, "t": 118, "b": 96},
        "height": 660,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "hovermode": "x unified",
        "legend": {
            "orientation": "h",
            "x": 0,
            "y": 1.16,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 15},
        },
        "xaxis": {
            "title": "",
            "type": "date",
            "tickformat": "%B %-d, %Y",
            "automargin": True,
            "tickfont": {"size": 15, "color": "#111111"},
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis": {
            "title": {
                "text": WATER_LEVEL_AXIS_LABEL,
                "font": {"size": 19, "color": "#111111"},
            },
            "range": [y_min, y_max],
            "automargin": True,
            "tickfont": {"size": 15, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": True,
            "gridcolor": "#d7dee8",
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "mirror": True,
            "zeroline": False,
        },
        "yaxis2": {
            "title": {
                "text": "Forecast precipitation (mm)",
                "font": {"size": 17, "color": "#111111"},
            },
            "range": [0, precipitation_y_max],
            "overlaying": "y",
            "side": "right",
            "automargin": True,
            "tickfont": {"size": 14, "color": "#111111"},
            "tickformat": ".2f",
            "showgrid": False,
            "showline": True,
            "linecolor": "#000000",
            "linewidth": 2,
            "zeroline": False,
        },
        "shapes": stage_shapes,
        "annotations": stage_annotations,
    }

    forecast_chart_layout = {
        **base_chart_layout,
        "xaxis": {
            **base_chart_layout["xaxis"],
            "range": forecast_x_range,
            "tickmode": "array",
            "tickvals": forecast_tick_dates,
            "ticktext": forecast_tick_labels,
            "tickangle": -35,
        },
    }

    model_test_chart_layout = {
        **base_chart_layout,
        "yaxis2": {
            **base_chart_layout["yaxis2"],
            "title": {
                "text": "Precipitation (mm)",
                "font": {"size": 17, "color": "#111111"},
            },
            "range": [0, model_precipitation_y_max],
        },
        "xaxis": {
            **base_chart_layout["xaxis"],
            "range": model_test_x_range,
            "tickmode": "array",
            "tickvals": model_test_tick_dates,
            "ticktext": model_test_tick_labels,
            "tickangle": -45,
        },
    }

    table_columns = forecast_output_df.columns.tolist()
    forecast_table_html = dataframe_to_html_table(forecast_output_df, table_columns)

    forecast_start = pd.Timestamp(forecast_df["date"].min()).strftime("%Y-%m-%d")
    forecast_end = pd.Timestamp(forecast_df["date"].max()).strftime("%Y-%m-%d")
    current_timestamp = pd.Timestamp.now(tz="America/New_York")
    today_date = current_timestamp.strftime("%Y-%m-%d")
    updated_at = current_timestamp.strftime("%Y-%m-%d %I:%M %p %Z")
    calibration_years = (
        (historical_df["date"].max() - historical_df["date"].min()).days / 365.25
        if not historical_df.empty
        else 0
    )
    watershed_geojson = load_watershed_geojson()
    watershed_geojson_json = json.dumps(watershed_geojson)
    future_climate_traces, future_climate_layout = build_future_climate_chart()
    monthly_precip_traces, monthly_precip_layout, monthly_temp_traces, monthly_temp_layout = (
        build_monthly_climate_comparison_charts()
    )
    watershed_map_html = ""
    if watershed_geojson is not None:
        watershed_map_html = f"""
        <aside class="watershed-map-card" aria-label="Interactive {html.escape(WATERSHED_NAME)} watershed map">
          <h2>{html.escape(WATERSHED_FULL_NAME)}</h2>
          <div id="watershedMap"></div>
        </aside>
        """

    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stream Water-Level Forecast Dashboard</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
      :root {{
        --ink: #1f2933;
        --muted: #64748b;
        --line: #d7dee8;
        --panel: #ffffff;
        --surface: #f3f7fb;
        --brand: #1f4e79;
        --brand-2: #2d7a78;
        --shadow: 0 18px 45px rgba(31, 78, 121, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-width: 320px;
        color: var(--ink);
        background: linear-gradient(180deg, rgba(31, 78, 121, 0.10), rgba(45, 122, 120, 0.04) 320px), var(--surface);
        font-family: Arial, Helvetica, sans-serif;
      }}
      .dashboard {{
        width: min(1900px, calc(100% - 32px));
        margin: 0 auto;
        padding: 28px 0 44px;
      }}
      .topbar {{
        display: block;
        margin-bottom: 20px;
      }}
      .overview-block {{
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 20px;
      }}
      h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
      h1 {{
        max-width: 100%;
        font-size: clamp(24px, 2vw, 34px);
        line-height: 1.12;
        white-space: nowrap;
      }}
      h2 {{ font-size: 20px; line-height: 1.25; }}
      h3 {{ margin: 22px 0 10px; color: var(--brand); font-size: 16px; }}
      .section-title {{
        display: block;
        margin: 28px 0 14px;
        padding: 13px 16px;
        border: 1px solid #9fb6d1;
        border-radius: 6px;
        background: var(--brand);
        color: #ffffff;
        font-size: 21px;
        font-weight: 700;
        line-height: 1.25;
      }}
      .eyebrow {{
        margin: 0 0 6px;
        color: var(--brand-2);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
      }}
      .status-strip {{
        display: grid;
        grid-template-columns: repeat(3, minmax(120px, 1fr));
        gap: 8px;
      }}
      .status-strip span {{
        padding: 11px 12px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.82);
        border-radius: 8px;
        color: var(--muted);
        font-size: 13px;
      }}
      .status-strip b {{
        display: block;
        color: var(--ink);
        font-size: 15px;
        margin-bottom: 3px;
      }}
      .watershed-map-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .watershed-map-card h2 {{
        padding: 14px 16px;
        border-bottom: 1px solid var(--line);
        color: var(--brand);
        font-size: 18px;
      }}
      #watershedMap {{
        width: 100%;
        height: 660px;
        background: #f8fbfd;
      }}
      .preview-panel {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .preview-toolbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        padding: 18px 20px;
        border-bottom: 1px solid var(--line);
      }}
      .email-preview {{ padding: 20px; }}
      .forecast-layout {{
        display: grid;
        grid-template-columns: minmax(260px, 340px) minmax(0, 1fr) minmax(280px, 380px);
        align-items: stretch;
        gap: 18px;
      }}
      .model-info-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 16px;
        box-shadow: var(--shadow);
        color: var(--ink);
        font-size: 14px;
        line-height: 1.45;
      }}
      .model-info-card p {{
        margin: 0 0 12px;
      }}
      .model-info-card h4 {{
        margin: 16px 0 7px;
        color: var(--brand);
        font-size: 14px;
      }}
      .model-info-card ul {{
        margin: 0 0 12px 18px;
        padding: 0;
      }}
      .model-info-card li {{
        margin: 3px 0;
      }}
      .chart-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        padding: 16px;
      }}
      #forecastChart,
      #modelTestChart {{ width: 100%; min-height: 660px; }}
      #futureClimateChart {{ width: 100%; min-height: 600px; }}
      #monthlyPrecipChart,
      #monthlyTempChart {{ width: 100%; min-height: 600px; }}
      .climate-comparison-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
      }}
      .forecast-insights {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 16px 0 8px;
      }}
      .insight-card {{
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        background: #f8fbfd;
      }}
      .insight-card h3 {{
        margin: 0;
        color: var(--ink);
        font-size: 18px;
      }}
      .insight-card p:last-child {{
        margin: 6px 0 0;
        color: var(--muted);
        font-size: 13px;
      }}
      .table-scroll {{
        overflow-x: auto;
        border: 1px solid var(--line);
        border-radius: 8px;
      }}
      .data-table {{
        width: 100%;
        border-collapse: collapse;
        min-width: 1500px;
        background: #fff;
      }}
      .data-table th,
      .data-table td {{
        padding: 10px 11px;
        border-bottom: 1px solid var(--line);
        border-right: 1px solid #edf1f6;
        text-align: left;
        white-space: nowrap;
        font-size: 13px;
      }}
      .data-table th {{
        position: sticky;
        top: 0;
        z-index: 1;
        color: #fff;
        background: var(--brand);
      }}
      .data-table tr:nth-child(even) td {{ background: #f8fbfd; }}
      @media (max-width: 900px) {{
        .status-strip {{
          grid-template-columns: 1fr;
          margin-top: 16px;
        }}
        h1 {{ white-space: normal; }}
        .forecast-layout {{ grid-template-columns: 1fr; }}
        #watershedMap {{ height: 460px; }}
        .forecast-insights {{ grid-template-columns: 1fr; }}
        .climate-comparison-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="dashboard">
      <section class="topbar" aria-label="Dashboard overview">
        <div class="overview-block">
          <div>
            <p class="eyebrow">Flood Prediction</p>
            <h1>{html.escape(DASHBOARD_TITLE)}</h1>
          </div>
          <div class="status-strip" aria-label="Forecast status">
            <span><b>Today's Date</b>{html.escape(today_date)}</span>
            <span><b>Updated</b>{html.escape(updated_at)}</span>
            <span><b>Forecast Window</b>{html.escape(forecast_start)} to {html.escape(forecast_end)}</span>
          </div>
        </div>
      </section>

      <section class="preview-panel" aria-label="Interactive forecast preview">
        <div class="preview-toolbar">
          <div>
            <p class="eyebrow">Interactive Dashboard</p>
          </div>
        </div>

        <article class="email-preview">
          <h2 class="section-title">14-Day Stream Water-Level Forecast, Runoff/Storage Scenarios, and Flood Stage Classification</h2>
          <div class="forecast-layout">
            <aside class="model-info-card" aria-label="Model information">
              <h4>Machine learning algorithm</h4>
              <p>Random Forest, LSTM, and GRU for baseline water-level forecasts</p>

              <h4>Model Development</h4>
              <p>Calibrated, validated, and tested using ~{calibration_years:.1f} years of historical daily data (2007-2026) obtained from USGS and ERA5</p>

              <h4>Features</h4>
              <p>Uses hydrologic lag, rolling precipitation, antecedent wetness, and seasonal features</p>

              <p>Future precipitation obtained from basin-average Open-Meteo forecasts</p>

              <h4>Flood-stage classification</h4>
              <ul>
                <li>Action</li>
                <li>Minor flood</li>
                <li>Moderate flood</li>
                <li>Major flood</li>
              </ul>

              <h4>Preventative-action scenario testing</h4>
              <ul>
                <li>10% runoff reduction</li>
                <li>20% runoff reduction</li>
                <li>2.5 mm storage</li>
                <li>5 mm storage</li>
              </ul>

              <p>Runoff reduction and increased storage could be achieved through detention basins, wetlands, floodplain reconnection, two-stage ditches, controlled drainage, saturated buffers, cover crops, conservation tillage, grassed waterways, and urban green infrastructure.</p>
            </aside>
            <section class="chart-card">
              <div id="forecastChart" aria-label="Interactive water-level forecast chart"></div>
            </section>
            {watershed_map_html}
          </div>

          <h2 class="section-title">Forecast Stream Water-Level (WL) and Flood Stage Classification</h2>
          <div class="table-scroll">
            <table class="data-table">
              {forecast_table_html}
            </table>
          </div>

          <h2 class="section-title">Machine Learning Forecast Model Performance Evaluation: Observed and Estimated Water-Levels During Model Testing (2025-2026)</h2>
          <section class="chart-card">
            <div id="modelTestChart" aria-label="Interactive model test chart"></div>
          </section>

          <section class="forecast-insights" aria-label="Model summary">
            <article class="insight-card">
              <p class="eyebrow">MAE</p>
              <h3>{metrics["MAE_m"]:.2f} m</h3>
              <p>Testing period error</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">RMSE</p>
              <h3>{metrics["RMSE_m"]:.2f} m</h3>
              <p>Testing period error</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">R2</p>
              <h3>{metrics["R2"]:.2f}</h3>
              <p>Testing period score</p>
            </article>
            <article class="insight-card">
              <p class="eyebrow">NSE</p>
              <h3>{metrics["NSE"]:.2f}</h3>
              <p>Hydrologic efficiency</p>
            </article>
          </section>

          <h2 class="section-title">Watershed Historical (1970-2010) and Future (2026-2100) Monthly Climate Comparison | NEX-GDDP CMIP6 | GCM: MIROC6 | SSP245</h2>
          <section class="climate-comparison-grid" aria-label="Historical and future monthly climate charts">
            <article class="chart-card">
              <h3>Monthly Average Precipitation: Historical 1970-2010 and Future 2026-2100</h3>
              <div id="monthlyPrecipChart" aria-label="Historical and future monthly precipitation chart"></div>
            </article>
            <article class="chart-card">
              <h3>Monthly Mean Temperature: Historical 1970-2010 and Future 2026-2100</h3>
              <div id="monthlyTempChart" aria-label="Historical and future monthly temperature chart"></div>
            </article>
          </section>

          <h2 class="section-title">Watershed Future (2026-2100) Annual Precipitation and Mean Temperature | NEX-GDDP CMIP6 | GCM: MIROC6 | SSP245</h2>
          <section class="chart-card">
            <div id="futureClimateChart" aria-label="Future annual precipitation and temperature chart"></div>
          </section>
        </article>
      </section>
    </main>

    <script>
      const forecastChartData = {json.dumps(forecast_traces)};
      const forecastChartLayout = {json.dumps(forecast_chart_layout)};
      const modelTestChartData = {json.dumps(model_test_traces)};
      const modelTestChartLayout = {json.dumps(model_test_chart_layout)};
      const futureClimateChartData = {json.dumps(future_climate_traces)};
      const futureClimateChartLayout = {json.dumps(future_climate_layout)};
      const monthlyPrecipChartData = {json.dumps(monthly_precip_traces)};
      const monthlyPrecipChartLayout = {json.dumps(monthly_precip_layout)};
      const monthlyTempChartData = {json.dumps(monthly_temp_traces)};
      const monthlyTempChartLayout = {json.dumps(monthly_temp_layout)};
      const watershedGeojson = {watershed_geojson_json};
      Plotly.newPlot("forecastChart", forecastChartData, forecastChartLayout, {{responsive: true, displaylogo: false}});
      Plotly.newPlot("modelTestChart", modelTestChartData, modelTestChartLayout, {{responsive: true, displaylogo: false}});
      if (futureClimateChartData.length > 0) {{
        Plotly.newPlot("futureClimateChart", futureClimateChartData, futureClimateChartLayout, {{responsive: true, displaylogo: false}});
      }}
      if (monthlyPrecipChartData.length > 0) {{
        Plotly.newPlot("monthlyPrecipChart", monthlyPrecipChartData, monthlyPrecipChartLayout, {{responsive: true, displaylogo: false}});
      }}
      if (monthlyTempChartData.length > 0) {{
        Plotly.newPlot("monthlyTempChart", monthlyTempChartData, monthlyTempChartLayout, {{responsive: true, displaylogo: false}});
      }}
      const watershedMapElement = document.getElementById("watershedMap");
      if (watershedMapElement && watershedGeojson && window.L) {{
        const watershedMap = L.map("watershedMap", {{
          scrollWheelZoom: false,
          zoomControl: true
        }});
        L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
          maxZoom: 19,
          attribution: "&copy; OpenStreetMap contributors"
        }}).addTo(watershedMap);
        const watershedLayer = L.geoJSON(watershedGeojson, {{
          style: {{
            color: "#1f4e79",
            weight: 3,
            fillColor: "#2d7a78",
            fillOpacity: 0.20
          }},
          pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
            radius: 5,
            color: "#8b1e1e",
            weight: 2,
            fillColor: "#ffffff",
            fillOpacity: 1
          }})
        }}).addTo(watershedMap);
        watershedMap.fitBounds(watershedLayer.getBounds(), {{padding: [18, 18]}});
        setTimeout(() => watershedMap.invalidateSize(), 0);
        setTimeout(() => watershedMap.invalidateSize(), 350);
        setTimeout(() => watershedMap.invalidateSize(), 1000);
      }}
    </script>
  </body>
</html>
"""

    output_file.write_text(document, encoding="utf-8")


def plot_feature_importance(model, feature_cols, output_file):
    """
    Saves feature importance plot.
    """

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(versioned_output_file("feature_importance.csv"), index=False)

    top_n = min(20, len(importance_df))
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)

    plt.figure(figsize=(9, 8))
    plt.barh(top_df["feature"], top_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances")
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def get_forecast_start_date(last_observed_date):
    """
    Returns the first date that should appear in the saved forecast.
    """

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    if FORECAST_START.lower() == "today":
        requested_start = today
    elif FORECAST_START.lower() == "tomorrow":
        requested_start = today + pd.Timedelta(days=1)
    else:
        raise ValueError('FORECAST_START must be either "today" or "tomorrow".')

    first_unobserved_date = pd.Timestamp(last_observed_date) + pd.Timedelta(days=1)

    return max(requested_start, first_unobserved_date)


def download_open_meteo_basin_average(url, params):
    """
    Downloads daily precipitation from Open-Meteo and averages basin points.
    """

    request_url = f"{url}?{urlencode(params)}"

    with urlopen(request_url, timeout=OPEN_METEO_TIMEOUT_SECONDS) as response:
        payload = response.read().decode("utf-8")

    meteo_data = json.loads(payload)

    if isinstance(meteo_data, list):
        point_payloads = [point["daily"] for point in meteo_data]
    elif isinstance(meteo_data, dict) and "daily" in meteo_data:
        point_payloads = [meteo_data["daily"]]
    else:
        raise ValueError("Unexpected Open-Meteo response format.")

    point_frames = []
    for point_index, daily in enumerate(point_payloads, start=1):
        point_df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "point_id": point_index,
            "precip_mm": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
        })
        point_frames.append(point_df)

    all_points_df = pd.concat(point_frames, ignore_index=True)

    basin_precip_df = (
        all_points_df
        .groupby("date", as_index=False)
        .agg(
            precip_mm=("precip_mm", "mean"),
            point_count=("precip_mm", "count"),
        )
        .sort_values("date")
    )

    return basin_precip_df


def fetch_open_meteo_basin_precipitation(forecast_days, last_observed_date, forecast_start_date):
    """
    Downloads daily precipitation from Open-Meteo for the basin points and
    averages the point forecasts by date.
    """

    first_needed_date = pd.Timestamp(last_observed_date) + pd.Timedelta(days=1)
    forecast_end_date = pd.Timestamp(forecast_start_date) + pd.Timedelta(days=forecast_days - 1)
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    latitudes = ",".join(f"{lat:.4f}" for lat, _ in BASIN_FORECAST_POINTS)
    longitudes = ",".join(f"{lon:.4f}" for _, lon in BASIN_FORECAST_POINTS)

    print("\nDownloading Open-Meteo precipitation forecast for basin points...")
    print(f"Number of forecast points: {len(BASIN_FORECAST_POINTS)}")

    basin_precip_frames = []

    archive_end_date = min(today - pd.Timedelta(days=OPEN_METEO_PAST_DAYS + 1), forecast_end_date)
    if first_needed_date <= archive_end_date:
        archive_params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "daily": "precipitation_sum",
            "timezone": "America/New_York",
            "start_date": first_needed_date.strftime("%Y-%m-%d"),
            "end_date": archive_end_date.strftime("%Y-%m-%d"),
        }
        print(
            "Downloading Open-Meteo archive precipitation bridge: "
            f"{first_needed_date.date()} to {archive_end_date.date()}"
        )
        basin_precip_frames.append(
            download_open_meteo_basin_average(OPEN_METEO_ARCHIVE_URL, archive_params)
        )

    forecast_request_start = max(first_needed_date, today - pd.Timedelta(days=OPEN_METEO_PAST_DAYS))
    request_past_days = max(0, (today - forecast_request_start).days)
    request_forecast_days = max(1, (forecast_end_date - today).days + 1)

    forecast_params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "daily": "precipitation_sum",
        "timezone": "America/New_York",
        "forecast_days": request_forecast_days,
    }
    if request_past_days > 0:
        forecast_params["past_days"] = request_past_days

    print(
        "Downloading Open-Meteo forecast/recent precipitation bridge: "
        f"{forecast_request_start.date()} to {forecast_end_date.date()}"
    )
    basin_precip_frames.append(
        download_open_meteo_basin_average(OPEN_METEO_URL, forecast_params)
    )

    basin_precip_df = (
        pd.concat(basin_precip_frames, ignore_index=True)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )

    round_numeric_columns(basin_precip_df).to_csv(
        versioned_output_file("open_meteo_basin_average_precipitation.csv"),
        index=False,
        float_format="%.2f"
    )

    future_precip_df = basin_precip_df[
        (basin_precip_df["date"] >= first_needed_date)
        & (basin_precip_df["date"] <= forecast_end_date)
    ].copy()

    needed_days = (forecast_end_date - first_needed_date).days + 1

    if len(future_precip_df) < needed_days:
        available_start = basin_precip_df["date"].min().date()
        available_end = basin_precip_df["date"].max().date()
        raise ValueError(
            "Open-Meteo did not return enough precipitation days to bridge from "
            f"the last observed water-level date ({pd.Timestamp(last_observed_date).date()}) "
            f"to the forecast end date ({forecast_end_date.date()}). "
            f"Available precipitation dates: {available_start} to {available_end}."
        )

    if future_precip_df["point_count"].min() < len(BASIN_FORECAST_POINTS):
        warnings.warn(
            "At least one Open-Meteo forecast day has fewer point values than expected. "
            "The basin average was calculated from the available point values."
        )

    future_precip_df = future_precip_df[["date", "precip_mm"]].copy()

    print("Open-Meteo basin-average precipitation selected for recursive forecasting:")
    print(future_precip_df)

    return future_precip_df


def prepare_future_precipitation(df, forecast_days):
    """
    Checks whether the input CSV already contains future rows with precipitation.

    Future rows should look like this:
    date, wl_m, precip_mm
    2026-05-07, , 12.4
    2026-05-08, , 2.1
    ...

    If future precipitation is not found, this function downloads Open-Meteo
    forecasts for the basin points and uses the daily basin-average precipitation.
    """

    historical_df = df.dropna(subset=["wl_m"]).copy()

    if historical_df.empty:
        raise ValueError("No historical water-level data found in Column B.")

    last_observed_date = historical_df["date"].max()
    forecast_start_date = get_forecast_start_date(last_observed_date)
    forecast_end_date = forecast_start_date + pd.Timedelta(days=forecast_days - 1)

    future_rows = df[
        (df["date"] > last_observed_date)
    ].copy()

    needed_future_rows = future_rows[
        (future_rows["date"] >= last_observed_date + pd.Timedelta(days=1))
        & (future_rows["date"] <= forecast_end_date)
    ].copy()

    needed_days = (forecast_end_date - last_observed_date).days

    if len(needed_future_rows) >= needed_days:
        future_precip_df = needed_future_rows.head(needed_days)[["date", "precip_mm"]].copy()
        future_precip_df["precip_mm"] = future_precip_df["precip_mm"].fillna(0)

        print("\nFuture precipitation was found in the input CSV.")
        print("The model will use those values for recursive forecasting.")

    else:
        print(
            "\nNo complete future precipitation was found in the input CSV. "
            "The model will use Open-Meteo basin-average precipitation instead."
        )

        future_precip_df = fetch_open_meteo_basin_precipitation(
            forecast_days=forecast_days,
            last_observed_date=last_observed_date,
            forecast_start_date=forecast_start_date
        )

    return historical_df, future_precip_df, forecast_start_date


def recursive_forecast_next_14_days(
    model,
    historical_df,
    future_precip_df,
    feature_cols,
    forecast_start_date
):
    """
    Predicts water level recursively.

    For each future day:
    - precipitation comes from future_precip_df
    - previous water levels come from observed history and previous predictions
    - dates before forecast_start_date are used only to bridge from the last
      observed water level to the requested forecast window
    """

    working_df = historical_df[["date", "wl_m", "precip_mm"]].copy()
    forecast_records = []
    forecast_start_date = pd.Timestamp(forecast_start_date)
    forecast_day = 0

    for i in range(len(future_precip_df)):

        next_date = future_precip_df.iloc[i]["date"]
        next_precip = future_precip_df.iloc[i]["precip_mm"]

        new_row = pd.DataFrame({
            "date": [next_date],
            "wl_m": [np.nan],
            "precip_mm": [next_precip]
        })

        temp_df = pd.concat([working_df, new_row], ignore_index=True)
        temp_features = add_features(temp_df)

        row_to_predict = temp_features.iloc[[-1]].copy()

        X_future = row_to_predict[feature_cols]

        if X_future.isna().any(axis=None):
            missing_cols = X_future.columns[X_future.isna().any()].tolist()
            raise ValueError(
                f"Missing values found in forecast features for {next_date.date()}. "
                f"Missing columns: {missing_cols}"
            )

        predicted_wl = model.predict(X_future)[0]

        # Replace missing wl with predicted value so it can be used as lag
        working_df = pd.concat([
            working_df,
            pd.DataFrame({
                "date": [next_date],
                "wl_m": [predicted_wl],
                "precip_mm": [next_precip]
            })
        ], ignore_index=True)

        if next_date >= forecast_start_date:
            forecast_day += 1
            forecast_records.append({
                "date": next_date,
                "forecast_day": forecast_day,
                "forecast_precip_mm": next_precip,
                "predicted_wl_m": predicted_wl,
                "water_level_stage": classify_water_level_stage(predicted_wl)
            })

    forecast_df = pd.DataFrame(forecast_records)

    return forecast_df


def recursive_forecast_next_14_days_deep(
    model_bundle,
    historical_df,
    future_precip_df,
    feature_cols,
    forecast_start_date
):
    """
    Predicts water level recursively with an LSTM or GRU sequence model.
    """

    working_df = historical_df[["date", "wl_m", "precip_mm"]].copy()
    forecast_records = []
    forecast_start_date = pd.Timestamp(forecast_start_date)
    forecast_day = 0
    sequence_length = model_bundle["sequence_length"]

    for i in range(len(future_precip_df)):

        next_date = future_precip_df.iloc[i]["date"]
        next_precip = future_precip_df.iloc[i]["precip_mm"]

        new_row = pd.DataFrame({
            "date": [next_date],
            "wl_m": [np.nan],
            "precip_mm": [next_precip]
        })

        temp_df = pd.concat([working_df, new_row], ignore_index=True)
        temp_features = add_features(temp_df)
        feature_sequence = temp_features[feature_cols].tail(sequence_length)

        if len(feature_sequence) < sequence_length:
            raise ValueError(
                f"Not enough rows to build a {sequence_length}-day sequence "
                f"for {next_date.date()}."
            )

        if feature_sequence.isna().any(axis=None):
            missing_cols = feature_sequence.columns[feature_sequence.isna().any()].tolist()
            raise ValueError(
                f"Missing values found in deep-learning forecast features for {next_date.date()}. "
                f"Missing columns: {missing_cols}"
            )

        X_future = model_bundle["feature_scaler"].transform(feature_sequence)
        X_future = X_future.reshape(1, sequence_length, len(feature_cols))
        scaled_prediction = model_bundle["model"].predict(X_future, verbose=0).reshape(-1, 1)
        predicted_wl = (
            model_bundle["target_scaler"]
            .inverse_transform(scaled_prediction)
            .ravel()[0]
        )

        # Replace missing wl with predicted value so it can be used as lag
        working_df = pd.concat([
            working_df,
            pd.DataFrame({
                "date": [next_date],
                "wl_m": [predicted_wl],
                "precip_mm": [next_precip]
            })
        ], ignore_index=True)

        if next_date >= forecast_start_date:
            forecast_day += 1
            forecast_records.append({
                "date": next_date,
                "forecast_day": forecast_day,
                "forecast_precip_mm": next_precip,
                "predicted_wl_m": predicted_wl,
                "water_level_stage": classify_water_level_stage(predicted_wl)
            })

    return pd.DataFrame(forecast_records)


def add_model_forecast_to_baseline(forecast_df, model_forecast_df, column_prefix):
    """
    Adds LSTM or GRU baseline forecast columns without changing scenario columns.
    """

    model_columns = model_forecast_df[[
        "date",
        "predicted_wl_m",
        "water_level_stage",
    ]].copy()
    model_columns = model_columns.rename(columns={
        "predicted_wl_m": f"{column_prefix}_predicted_wl_m",
        "water_level_stage": f"{column_prefix}_water_level_stage",
    })

    return forecast_df.merge(model_columns, on="date", how="left")


def apply_forecast_scenario(future_precip_df, forecast_start_date, scenario):
    """
    Applies a runoff-reduction or storage scenario to forecast-window precipitation.
    Bridge days before forecast_start_date are kept unchanged.
    """

    scenario_precip_df = future_precip_df.copy()
    forecast_mask = scenario_precip_df["date"] >= forecast_start_date

    if scenario["kind"] == "multiply":
        scenario_precip_df.loc[forecast_mask, "precip_mm"] = (
            scenario_precip_df.loc[forecast_mask, "precip_mm"] * scenario["value"]
        )
    elif scenario["kind"] == "storage":
        scenario_precip_df.loc[forecast_mask, "precip_mm"] = (
            scenario_precip_df.loc[forecast_mask, "precip_mm"] - scenario["value"]
        ).clip(lower=0)
    else:
        raise ValueError(f"Unknown forecast scenario type: {scenario['kind']}")

    return scenario_precip_df


def build_scenario_forecasts(model, historical_df, future_precip_df, feature_cols, forecast_start_date):
    """
    Runs all forecast scenarios and returns their forecast data frames with metadata.
    """

    scenario_forecasts = []

    for scenario in FORECAST_SCENARIOS:
        scenario_precip_df = apply_forecast_scenario(
            future_precip_df=future_precip_df,
            forecast_start_date=forecast_start_date,
            scenario=scenario
        )
        scenario_forecast_df = recursive_forecast_next_14_days(
            model=model,
            historical_df=historical_df,
            future_precip_df=scenario_precip_df,
            feature_cols=feature_cols,
            forecast_start_date=forecast_start_date
        )

        scenario_forecasts.append({
            **scenario,
            "forecast_df": scenario_forecast_df
        })

    return scenario_forecasts


def add_scenarios_to_forecast_table(forecast_df, scenario_forecasts):
    """
    Adds each scenario water level and stage category as extra columns.
    """

    output_df = forecast_df.copy()

    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"][[
            "date",
            "predicted_wl_m",
            "water_level_stage"
        ]].copy()
        scenario_df = scenario_df.rename(columns={
            "predicted_wl_m": f"{scenario['column_prefix']}_predicted_wl_m",
            "water_level_stage": f"{scenario['column_prefix']}_water_level_stage",
        })

        output_df = output_df.merge(scenario_df, on="date", how="left")

    return output_df


# ==============================================================
# 3. MAIN WORKFLOW
# ==============================================================

def apply_watershed_config(config):
    """
    Sets the active watershed configuration used by the workflow.
    """

    global INPUT_FILE
    global WATERSHED_GEOJSON_FILE
    global WATERSHED_SHAPEFILE_DIR
    global OUTPUT_PREFIX
    global WATERSHED_SLUG
    global WATERSHED_NAME
    global WATERSHED_FULL_NAME
    global DASHBOARD_TITLE
    global WATER_LEVEL_AXIS_LABEL
    global FUTURE_CLIMATE_FILE
    global HISTORICAL_CLIMATE_FILE
    global BASIN_FORECAST_POINTS
    global ACTION_STAGE_M
    global MINOR_FLOOD_STAGE_M
    global MODERATE_FLOOD_STAGE_M
    global MAJOR_FLOOD_STAGE_M

    INPUT_FILE = config["input_file"]
    WATERSHED_GEOJSON_FILE = config["geojson_file"]
    WATERSHED_SHAPEFILE_DIR = config["shapefile_dir"]
    OUTPUT_PREFIX = config["output_prefix"]
    WATERSHED_SLUG = config["slug"]
    WATERSHED_NAME = config["name"]
    WATERSHED_FULL_NAME = config["full_name"]
    DASHBOARD_TITLE = config["dashboard_title"]
    WATER_LEVEL_AXIS_LABEL = config["axis_label"]
    FUTURE_CLIMATE_FILE = config["future_climate_file"]
    HISTORICAL_CLIMATE_FILE = config["historical_climate_file"]
    BASIN_FORECAST_POINTS = config["forecast_points"]
    stage_thresholds = config["stage_thresholds_m"]
    ACTION_STAGE_M = stage_thresholds["action"]
    MINOR_FLOOD_STAGE_M = stage_thresholds["minor"]
    MODERATE_FLOOD_STAGE_M = stage_thresholds["moderate"]
    MAJOR_FLOOD_STAGE_M = stage_thresholds["major"]


def run_current_watershed():

    print("\n====================================================")
    print(f"Stream Water-Level ML Prediction - {WATERSHED_NAME}")
    print("====================================================")

    print(f"\nReading input file:\n{INPUT_FILE}")

    df = clean_input_data(INPUT_FILE)

    print("\nData summary after cleaning:")
    print(df.head())
    print(df.tail())
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Number of daily records: {len(df)}")

    # Separate historical observed data and future precipitation rows if available
    historical_df, future_precip_df, forecast_start_date = prepare_future_precipitation(
        df,
        FORECAST_DAYS
    )

    print(f"\nLast observed water-level date: {historical_df['date'].max().date()}")
    print(f"Saved forecast starts on: {forecast_start_date.date()}")

    # Feature engineering using only historical observed water-level data
    feature_df = add_features(historical_df)

    feature_cols = get_feature_columns(feature_df)

    # Model-ready data
    model_df = feature_df.dropna(subset=feature_cols + ["wl_m"]).copy()
    model_df = model_df.reset_index(drop=True)

    if model_df.empty:
        raise ValueError(
            "After feature engineering, no complete rows are available for training. "
            "Check missing water-level or precipitation data."
        )

    print(f"\nNumber of rows available for model training/testing: {len(model_df)}")

    # Chronological train/test split
    train_df, test_df = train_test_split_time_series(model_df, TEST_FRACTION)

    X_train = train_df[feature_cols]
    y_train = train_df["wl_m"]

    X_test = test_df[feature_cols]

    print("\nTraining period:")
    print(f"{train_df['date'].min().date()} to {train_df['date'].max().date()}")

    print("\nTesting period:")
    print(f"{test_df['date'].min().date()} to {test_df['date'].max().date()}")

    # ==============================================================
    # 4. TRAIN MODEL
    # ==============================================================

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    print("\nTraining LSTM model...")
    lstm_model = train_deep_sequence_model(
        model_kind="lstm",
        model_df=model_df,
        feature_cols=feature_cols,
        train_end_position=len(train_df) - 1,
    )

    print("Training GRU model...")
    gru_model = train_deep_sequence_model(
        model_kind="gru",
        model_df=model_df,
        feature_cols=feature_cols,
        train_end_position=len(train_df) - 1,
    )

    # ==============================================================
    # 5. TEST MODEL
    # ==============================================================

    test_df["predicted_wl_m"] = model.predict(X_test)
    lstm_prediction_series = predict_deep_sequence_model(lstm_model, model_df, feature_cols)
    gru_prediction_series = predict_deep_sequence_model(gru_model, model_df, feature_cols)
    test_df["lstm_predicted_wl_m"] = lstm_prediction_series.reindex(test_df.index).to_numpy()
    test_df["gru_predicted_wl_m"] = gru_prediction_series.reindex(test_df.index).to_numpy()

    metrics = evaluate_prediction_column(test_df, "predicted_wl_m")
    lstm_metrics = evaluate_prediction_column(test_df, "lstm_predicted_wl_m")
    gru_metrics = evaluate_prediction_column(test_df, "gru_predicted_wl_m")
    metrics_by_algorithm = [
        {"algorithm": "Random Forest", **metrics},
        {"algorithm": "LSTM", **lstm_metrics},
        {"algorithm": "GRU", **gru_metrics},
    ]

    print("\nModel performance on testing period:")
    for algorithm_metrics in metrics_by_algorithm:
        print(f"\n{algorithm_metrics['algorithm']}:")
        for key, value in algorithm_metrics.items():
            if key == "algorithm":
                continue
            print(f"{key}: {value:.2f}")

    # Save metrics
    metrics_df = round_numeric_columns(pd.DataFrame(metrics_by_algorithm))
    metrics_df.to_csv(
        versioned_output_file("model_performance_metrics.csv"),
        index=False,
        float_format="%.2f"
    )

    # Save test predictions
    test_output = test_df[[
        "date",
        "wl_m",
        "predicted_wl_m",
        "lstm_predicted_wl_m",
        "gru_predicted_wl_m",
        "precip_mm",
    ]].copy()
    test_output = round_numeric_columns(test_output)
    test_output.to_csv(
        versioned_output_file("test_observed_vs_predicted.csv"),
        index=False,
        float_format="%.2f"
    )

    # Plots
    plot_observed_vs_predicted(
        test_df,
        versioned_output_file("observed_vs_predicted_timeseries.png")
    )

    plot_scatter_observed_vs_predicted(
        test_df,
        versioned_output_file("observed_vs_predicted_scatter.png")
    )

    plot_feature_importance(
        model,
        feature_cols,
        versioned_output_file("feature_importance.png")
    )

    # ==============================================================
    # 6. RETRAIN MODEL ON ALL HISTORICAL DATA
    # ==============================================================

    print("\nRetraining model on all historical data before forecasting...")

    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    X_all = model_df[feature_cols]
    y_all = model_df["wl_m"]

    final_model.fit(X_all, y_all)

    print("Retraining LSTM model on all historical data...")
    final_lstm_model = train_deep_sequence_model(
        model_kind="lstm",
        model_df=model_df,
        feature_cols=feature_cols,
    )

    print("Retraining GRU model on all historical data...")
    final_gru_model = train_deep_sequence_model(
        model_kind="gru",
        model_df=model_df,
        feature_cols=feature_cols,
    )

    # ==============================================================
    # 7. FORECAST NEXT 14 DAYS
    # ==============================================================

    print("\nForecasting next 14 days...")

    forecast_df = recursive_forecast_next_14_days(
        model=final_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )
    lstm_forecast_df = recursive_forecast_next_14_days_deep(
        model_bundle=final_lstm_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )
    gru_forecast_df = recursive_forecast_next_14_days_deep(
        model_bundle=final_gru_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )
    forecast_df = add_model_forecast_to_baseline(
        forecast_df=forecast_df,
        model_forecast_df=lstm_forecast_df,
        column_prefix="lstm",
    )
    forecast_df = add_model_forecast_to_baseline(
        forecast_df=forecast_df,
        model_forecast_df=gru_forecast_df,
        column_prefix="gru",
    )

    scenario_forecasts = build_scenario_forecasts(
        model=final_model,
        historical_df=historical_df,
        future_precip_df=future_precip_df,
        feature_cols=feature_cols,
        forecast_start_date=forecast_start_date
    )

    forecast_output_df = add_scenarios_to_forecast_table(
        forecast_df=forecast_df,
        scenario_forecasts=scenario_forecasts
    )

    forecast_output_df = round_numeric_columns(forecast_output_df)
    forecast_output_df.to_csv(
        versioned_output_file("next_14_day_water_level_forecast.csv"),
        index=False,
        float_format="%.2f"
    )

    print("\nNext 14-day forecast:")
    print(forecast_output_df)

    # Plot forecast
    recent_history = historical_df.tail(60)[["date", "wl_m", "precip_mm"]].copy()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax2.bar(
        forecast_df["date"],
        forecast_df["forecast_precip_mm"],
        color="lightskyblue",
        alpha=0.45,
        label="Forecast precipitation",
        width=0.8
    )
    ax1.plot(
        recent_history["date"],
        recent_history["wl_m"],
        label="Observed water level",
        linewidth=2
    )
    ax1.plot(
        forecast_df["date"],
        forecast_df["predicted_wl_m"],
        marker="o",
        label="Forecast water-level",
        linewidth=2
    )
    if "lstm_predicted_wl_m" in forecast_df.columns:
        ax1.plot(
            forecast_df["date"],
            forecast_df["lstm_predicted_wl_m"],
            marker="o",
            linestyle=":",
            color="#5b2c83",
            label="LSTM forecast water-level",
            linewidth=2
        )
    if "gru_predicted_wl_m" in forecast_df.columns:
        ax1.plot(
            forecast_df["date"],
            forecast_df["gru_predicted_wl_m"],
            marker="o",
            linestyle="-.",
            color="#006d77",
            label="GRU forecast water-level",
            linewidth=2
        )
    for scenario in scenario_forecasts:
        scenario_df = scenario["forecast_df"]
        ax1.plot(
            scenario_df["date"],
            scenario_df["predicted_wl_m"],
            marker="o",
            linestyle=scenario["plot_style"],
            color=scenario["plot_color"],
            label=scenario["name"],
            linewidth=2
        )

    ax1.set_zorder(2)
    ax2.set_zorder(1)
    ax1.patch.set_alpha(0)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(WATER_LEVEL_AXIS_LABEL)
    ax2.set_ylabel("Forecast precipitation (mm)")
    ax2.yaxis.label.set_color("#111111")
    ax2.tick_params(axis="y", colors="#111111")
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax1.set_title("Observed Recent Water Level and 14-Day Forecast")
    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(versioned_output_file("next_14_day_forecast_plot.png"), dpi=300)
    plt.close()

    plot_stage_forecast_summary(
        historical_df=historical_df,
        test_df=test_df,
        forecast_df=forecast_df,
        output_file=versioned_output_file("observed_estimated_forecast_stages.png"),
        scenario_forecasts=scenario_forecasts
    )

    write_interactive_index(
        historical_df=historical_df,
        test_df=test_df,
        forecast_df=forecast_df,
        forecast_output_df=forecast_output_df,
        scenario_forecasts=scenario_forecasts,
        metrics=metrics,
        output_file=versioned_output_file("index.html")
    )

    # Save full feature list
    pd.DataFrame({"feature": feature_cols}).to_csv(
        versioned_output_file("model_feature_columns.csv"),
        index=False
    )

    print("\n====================================================")
    print("Analysis complete.")
    print(f"Outputs saved here:\n{OUTPUT_DIR}")
    print("====================================================")

    return {
        "slug": WATERSHED_SLUG,
        "name": WATERSHED_NAME,
        "full_name": WATERSHED_FULL_NAME,
        "dashboard_file": versioned_output_file("index.html").name,
        "forecast_start": forecast_df["date"].min(),
        "forecast_end": forecast_df["date"].max(),
        "updated_at": pd.Timestamp.now(tz="America/New_York"),
    }


def write_watershed_selector_index(watershed_results):
    """
    Writes the main GitHub Pages entrypoint with a watershed selector.
    """

    buttons_html = "\n".join(
        f"""
        <button
          class="watershed-button{" active" if index == 0 else ""}"
          type="button"
          data-watershed="{html.escape(result["slug"])}"
          data-dashboard="{html.escape(result["dashboard_file"])}">
          {html.escape(result["name"])}
        </button>
        """
        for index, result in enumerate(watershed_results)
    )
    initial_dashboard = watershed_results[0]["dashboard_file"]
    updated_at = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %I:%M %p %Z")

    cards_html = "\n".join(
        f"""
        <article class="watershed-card">
          <h2>{html.escape(result["full_name"].split(",")[0])}</h2>
          <p>{html.escape(result["full_name"])}</p>
          <p><b>Forecast Window</b><br>{pd.Timestamp(result["forecast_start"]).strftime("%Y-%m-%d")} to {pd.Timestamp(result["forecast_end"]).strftime("%Y-%m-%d")}</p>
        </article>
        """
        for result in watershed_results
    )

    document = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Flood Prediction Watershed Selector</title>
    <style>
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-width: 320px;
        color: #1f2933;
        background: linear-gradient(180deg, rgba(31, 78, 121, 0.10), rgba(45, 122, 120, 0.04) 320px), #f3f7fb;
        font-family: Arial, Helvetica, sans-serif;
      }}
      main {{
        width: min(1900px, calc(100% - 32px));
        margin: 0 auto;
        padding: 28px 0 44px;
      }}
      .selector-bar {{
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 14px;
        margin: 0 0 18px;
        padding: 22px 24px;
        border: 2px solid #1f4e79;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 18px 45px rgba(31, 78, 121, 0.12);
      }}
      .selector-copy h2 {{
        margin: 0;
        color: #1f4e79;
        font-size: 28px;
        line-height: 1.15;
      }}
      .selector-copy p {{
        margin: 8px 0 0;
        color: #334155;
        font-size: 18px;
        line-height: 1.4;
      }}
      .button-group {{
        display: inline-flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 5px;
        border: 1px solid #d7dee8;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 10px 26px rgba(31, 78, 121, 0.10);
      }}
      .watershed-button {{
        border: 1px solid transparent;
        border-radius: 6px;
        background: transparent;
        color: #1f4e79;
        cursor: pointer;
        font-size: 18px;
        font-weight: 700;
        padding: 13px 22px;
      }}
      .watershed-button.active {{
        background: #1f4e79;
        color: #ffffff;
      }}
      .topbar {{
        display: block;
        margin-bottom: 16px;
      }}
      h1 {{ margin: 0; font-size: 34px; line-height: 1.12; }}
      .main-title {{
        display: block;
        margin: 0 0 12px;
        padding: 16px 18px;
        border: 1px solid #9fb6d1;
        border-radius: 6px;
        background: #1f4e79;
        color: #ffffff;
        font-size: 34px;
        font-weight: 700;
        line-height: 1.18;
      }}
      .eyebrow {{
        margin: 0 0 6px;
        color: #2d7a78;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
      }}
      .cards {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
      }}
      .watershed-card {{
        border: 1px solid #d7dee8;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.86);
        padding: 14px;
      }}
      .watershed-card h2 {{
        margin: 0 0 5px;
        color: #1f4e79;
        font-size: 18px;
      }}
      .watershed-card p {{
        margin: 6px 0 0;
        color: #64748b;
        font-size: 14px;
        line-height: 1.35;
      }}
      .watershed-frame {{
        display: block;
        width: 100%;
        height: 3300px;
        min-height: 3300px;
        border: 1px solid #d7dee8;
        border-radius: 8px;
        background: #fff;
        overflow: hidden;
      }}
      @media (max-width: 900px) {{
        .selector-bar {{
          align-items: stretch;
          flex-direction: column;
        }}
        .topbar,
        .cards {{ grid-template-columns: 1fr; }}
        .button-group {{ width: 100%; }}
        .watershed-button {{ flex: 1; }}
        .watershed-frame {{
          height: 3800px;
          min-height: 3800px;
        }}
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="topbar" aria-label="Watershed selection">
        <div>
          <h1 class="main-title">Live Machine Learning-Based 14-Day Flood Forecast and Climate Comparison Dashboard</h1>
          <p>Updated {html.escape(updated_at)}</p>
        </div>
      </section>

      <section class="selector-bar" aria-label="Watershed switcher">
        <div class="selector-copy">
          <p class="eyebrow">Select Watershed</p>
          <h2>View Forecast by Watershed</h2>
          <p>Choose a watershed to update the forecast, map, tables, and model results.</p>
        </div>
        <div class="button-group" role="tablist" aria-label="Watershed dashboards">
          {buttons_html}
        </div>
      </section>

      <section class="cards" aria-label="Watershed forecast summaries">
        {cards_html}
      </section>
      <section aria-label="Selected watershed dashboard">
        <iframe
          id="watershedFrame"
          class="watershed-frame"
          title="Selected watershed dashboard"
          scrolling="no"
          src="{html.escape(initial_dashboard)}">
        </iframe>
      </section>
    </main>
    <script>
      const watershedButtons = Array.from(document.querySelectorAll(".watershed-button"));
      const watershedFrame = document.getElementById("watershedFrame");

      function resizeWatershedFrame() {{
        try {{
          const frameDocument = watershedFrame.contentDocument || watershedFrame.contentWindow.document;
          const body = frameDocument.body;
          const documentElement = frameDocument.documentElement;
          const contentHeight = Math.max(
            body.scrollHeight,
            body.offsetHeight,
            documentElement.clientHeight,
            documentElement.scrollHeight,
            documentElement.offsetHeight
          );
          watershedFrame.style.height = `${{contentHeight + 12}}px`;
        }} catch (error) {{
          watershedFrame.style.height = "4200px";
        }}
      }}

      watershedFrame.addEventListener("load", () => {{
        resizeWatershedFrame();
        window.setTimeout(resizeWatershedFrame, 400);
        window.setTimeout(resizeWatershedFrame, 1400);
      }});
      window.addEventListener("resize", resizeWatershedFrame);

      watershedButtons.forEach((button) => {{
        button.addEventListener("click", () => {{
          const selectedWatershed = button.dataset.watershed;
          watershedButtons.forEach((item) => {{
            item.classList.toggle("active", item.dataset.watershed === selectedWatershed);
          }});
          watershedFrame.src = button.dataset.dashboard;
          window.scrollTo({{top: 0, behavior: "smooth"}});
        }});
      }});
    </script>
  </body>
</html>
"""

    (OUTPUT_DIR / "index.html").write_text(document, encoding="utf-8")


def main():
    watershed_results = []
    for config in WATERSHEDS:
        apply_watershed_config(config)
        watershed_results.append(run_current_watershed())

    write_watershed_selector_index(watershed_results)


if __name__ == "__main__":
    main()
