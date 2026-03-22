import pandas as pd


splits = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}


df_train = pd.read_csv(
    "hf://datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/"
    + splits["train"]
)
df_validation = pd.read_csv(
    "hf://datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/"
    + splits["validation"]
)
df_test = pd.read_csv(
    "hf://datasets/SahandNZ/cryptonews-articles-with-price-momentum-labels/"
    + splits["test"]
)


dfs = [df_train, df_validation, df_test]
for df in dfs:

    df["date"] = pd.to_datetime(df["datetime"]).dt.date  # for merging


import yfinance as yf

# Get the min and max dates from your news data
start_date = min(
    df_train["date"].min(), df_validation["date"].min(), df_test["date"].min()
)
end_date = max(
    df_train["date"].max(), df_validation["date"].max(), df_test["date"].max()
)
yf_end_date = end_date + pd.Timedelta(days=1)

btc = yf.download("BTC-USD", start=start_date, end=yf_end_date, interval="1d")
btc = btc.reset_index()  # 'Date' becomes a column
btc["date"] = btc["Date"].dt.date
btc.rename(
    columns={
        "Open": "btc_open",
        "High": "btc_high",
        "Low": "btc_low",
        "Close": "btc_close",
        "Volume": "btc_volume",
    },
    inplace=True,
)
btc = btc[["date", "btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]]

import requests
import json


def fetch_fng(start_ts, end_ts):
    url = (
        f"https://api.alternative.me/fng/?limit=0&format=json&date_format=us"  # get all
    )
    response = requests.get(url)
    data = response.json()["data"]
    fng_list = []
    for item in data:
        fng_list.append(
            {
                "date": pd.to_datetime(item["timestamp"]).date(),
                "fng_value": int(item["value"]),
                "fng_classification": item["value_classification"],
            }
        )
    df_fng = pd.DataFrame(fng_list)
    # Filter by your date range
    df_fng = df_fng[(df_fng["date"] >= start_date) & (df_fng["date"] <= end_date)]
    return df_fng


df_fng = fetch_fng(start_date, end_date)

df = pd.concat(
    [
        df_train.assign(split="train"),
        df_validation.assign(split="val"),
        df_test.assign(split="test"),
    ],
    ignore_index=True,
)

btc_for_merge = btc.copy()
if isinstance(btc_for_merge.columns, pd.MultiIndex):
    btc_for_merge.columns = [
        (
            "_".join([str(level) for level in col if level not in (None, "")]).strip(
                "_"
            )
            if isinstance(col, tuple)
            else str(col)
        )
        for col in btc_for_merge.columns
    ]


def pick_col(df_columns, must_include):
    for col in df_columns:
        name = str(col).lower()
        if all(token in name for token in must_include):
            return col
    return None


date_col = pick_col(btc_for_merge.columns, ["date"])
open_col = pick_col(btc_for_merge.columns, ["open"])
high_col = pick_col(btc_for_merge.columns, ["high"])
low_col = pick_col(btc_for_merge.columns, ["low"])
close_col = pick_col(btc_for_merge.columns, ["close"])
volume_col = pick_col(btc_for_merge.columns, ["volume"])

btc_for_merge = btc_for_merge.rename(
    columns={
        date_col: "date",
        open_col: "btc_open",
        high_col: "btc_high",
        low_col: "btc_low",
        close_col: "btc_close",
        volume_col: "btc_volume",
    }
)

btc_for_merge["date"] = pd.to_datetime(btc_for_merge["date"]).dt.date
btc_for_merge = btc_for_merge[
    ["date", "btc_open", "btc_high", "btc_low", "btc_close", "btc_volume"]
]

df_merged = df.merge(btc_for_merge, on="date", how="left")
df_merged = df_merged.merge(df_fng, on="date", how="left")


def add_lagged_market_features(news_df):
    # Build one row per date, then shift to D-1 to avoid look-ahead bias.
    daily = (
        news_df[
            [
                "date",
                "btc_open",
                "btc_high",
                "btc_low",
                "btc_close",
                "btc_volume",
                "fng_value",
                "fng_classification",
            ]
        ]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    lag_cols = [
        "btc_open",
        "btc_high",
        "btc_low",
        "btc_close",
        "btc_volume",
        "fng_value",
        "fng_classification",
    ]
    for col in lag_cols:
        daily[f"{col}_lag1"] = daily[col].shift(1)

    daily["btc_return_lag1"] = (
        daily["btc_close_lag1"] / daily["btc_open_lag1"].replace(0, pd.NA)
    ) - 1.0
    daily["btc_volatility_lag1"] = (
        (daily["btc_high_lag1"] - daily["btc_low_lag1"])
        / daily["btc_open_lag1"].replace(0, pd.NA)
    )

    daily["btc_volume_7d_avg_lag1"] = (
        daily["btc_volume"].shift(1).rolling(window=7, min_periods=1).mean()
    )
    daily["btc_volume_change_vs_7d_lag1"] = (
        daily["btc_volume_lag1"] / daily["btc_volume_7d_avg_lag1"].replace(0, pd.NA)
    ) - 1.0

    daily_lagged = daily[
        [
            "date",
            "btc_open_lag1",
            "btc_high_lag1",
            "btc_low_lag1",
            "btc_close_lag1",
            "btc_volume_lag1",
            "fng_value_lag1",
            "fng_classification_lag1",
            "btc_return_lag1",
            "btc_volatility_lag1",
            "btc_volume_change_vs_7d_lag1",
        ]
    ]

    out = news_df.merge(daily_lagged, on="date", how="left")

    # Keep only rows with complete lagged context.
    out = out.dropna(
        subset=[
            "btc_open_lag1",
            "btc_high_lag1",
            "btc_low_lag1",
            "btc_close_lag1",
            "btc_volume_lag1",
            "fng_value_lag1",
            "fng_classification_lag1",
            "btc_return_lag1",
            "btc_volatility_lag1",
            "btc_volume_change_vs_7d_lag1",
        ]
    )

    return out


df_ready = add_lagged_market_features(df_merged)
df_train_after_merge = df_ready[df_ready["split"] == "train"]
df_validation_after_merge = df_ready[df_ready["split"] == "val"]
df_test_after_merge = df_ready[df_ready["split"] == "test"]

df_train_after_merge.to_parquet("train_after_merge.parquet", index=False)
df_validation_after_merge.to_parquet("validation_after_merge.parquet", index=False)
df_test_after_merge.to_parquet("test_after_merge.parquet", index=False)
df_ready.to_parquet("enriched_news.parquet", index=False)
