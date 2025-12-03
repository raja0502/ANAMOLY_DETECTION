import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

DB_PATH = "near_production.db"

def load_transactions(limit=None):
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT
            t.txn_id,
            t.customer_id,
            t.account_id,
            t.txn_date,
            t.amount,
            t.channel,
            t.category,
            t.status,
            c.age,
            c.balance,
            a.account_type,
            a.opened_date
        FROM transactions t
        LEFT JOIN customers c ON t.customer_id = c.customer_id
        LEFT JOIN accounts  a ON t.account_id   = a.account_id
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def build_feature_matrix(df: pd.DataFrame):
    # Keep ID columns separately
    id_cols = ["txn_id", "customer_id", "account_id"]
    meta_cols = ["txn_date", "opened_date"]
    keep_cols = id_cols + meta_cols

    df_work = df.copy()

    # Basic time features from dates
    df_work["txn_date"] = pd.to_datetime(df_work["txn_date"], errors="coerce")
    df_work["opened_date"] = pd.to_datetime(df_work["opened_date"], errors="coerce")

    df_work["txn_year"] = df_work["txn_date"].dt.year
    df_work["txn_month"] = df_work["txn_date"].dt.month
    df_work["txn_day"] = df_work["txn_date"].dt.day
    df_work["txn_dow"] = df_work["txn_date"].dt.dayofweek

    df_work["days_since_open"] = (
        (df_work["txn_date"] - df_work["opened_date"]).dt.days
    )

    # Replace NaNs in numeric cols with 0 before scaling
    numeric_features = ["amount", "age", "balance", "txn_year", "txn_month",
                        "txn_day", "txn_dow", "days_since_open"]
    categorical_features = ["channel", "category", "status", "account_type"]

    X = df_work[numeric_features + categorical_features]

    # Preprocess: scale numeric, one-hot encode categoricals
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    X_processed = preprocessor.fit_transform(X)

    return df_work, X_processed, preprocessor, numeric_features, categorical_features, keep_cols

def run_isolation_forest(X, contamination=0.1, random_state=42):
    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X)
    scores = -iso.score_samples(X)  # higher = more anomalous
    labels = iso.predict(X)         # -1 = anomaly, 1 = normal
    return iso, scores, labels

def save_anomalies(df_work, scores, labels, contamination_tag="0.1"):
    df_out = df_work.copy()
    df_out["anomaly_score"] = scores
    df_out["anomaly_label"] = labels

    # Only anomalies
    anomalies = df_out[df_out["anomaly_label"] == -1].copy()
    anomalies.sort_values("anomaly_score", ascending=False, inplace=True)

    conn = sqlite3.connect(DB_PATH)
    # Save full scores
    df_out.to_sql("transactions_anomaly_iforest", conn,
                  if_exists="replace", index=False)
    # Save just anomalies
    anomalies.to_sql("transactions_anomalies_iforest", conn,
                     if_exists="replace", index=False)
    conn.close()

    print(f"Total rows scored: {len(df_out)}")
    print(f"Anomalies detected: {len(anomalies)}")
    print(f"Top 5 anomaly scores:\n{anomalies[['txn_id','amount','channel','status','anomaly_score']].head()}")

if __name__ == "__main__":
    print("Loading data from SQLite...")
    df = load_transactions()  # or limit=50000 for faster testing
    print("Rows loaded:", len(df))

    print("Building feature matrix...")
    df_work, X, preproc, num_cols, cat_cols, keep_cols = build_feature_matrix(df)

    print("Running Isolation Forest...")
    iso, scores, labels = run_isolation_forest(X, contamination=0.1)

    print("Saving results...")
    save_anomalies(df_work, scores, labels)
    print("Done.")
