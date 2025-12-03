import sqlite3
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

DB_PATH = "near_production.db"

# ---------- 1. Load same features as anomaly_isoforest.py ----------

def load_scored_transactions(limit_anomalies=200):
    conn = sqlite3.connect(DB_PATH)
    # Use the table with all scores/labels from anomaly_isoforest.py
    df = pd.read_sql_query("SELECT * FROM transactions_anomaly_iforest", conn)
    conn.close()

    # Pick only anomalies, highest scores first
    df_anom = df[df["anomaly_label"] == -1].copy()
    df_anom.sort_values("anomaly_score", ascending=False, inplace=True)
    df_anom = df_anom.head(limit_anomalies)

    return df_anom

def build_feature_matrix_for_shap(df_work: pd.DataFrame):
    # Must match anomaly_isoforest.py exactly
    numeric_features = ["amount", "age", "balance", "txn_year", "txn_month",
                        "txn_day", "txn_dow", "days_since_open"]
    categorical_features = ["channel", "category", "status", "account_type"]

    # Recreate time features if needed
    df_work = df_work.copy()
    df_work["txn_date"] = pd.to_datetime(df_work["txn_date"], errors="coerce")
    df_work["opened_date"] = pd.to_datetime(df_work["opened_date"], errors="coerce")

    df_work["txn_year"] = df_work["txn_date"].dt.year
    df_work["txn_month"] = df_work["txn_date"].dt.month
    df_work["txn_day"] = df_work["txn_date"].dt.day
    df_work["txn_dow"] = df_work["txn_date"].dt.dayofweek
    df_work["days_since_open"] = (
        (df_work["txn_date"] - df_work["opened_date"]).dt.days
    )

    X = df_work[numeric_features + categorical_features]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    X_processed = preprocessor.fit_transform(X)

    return X_processed, preprocessor, numeric_features, categorical_features

# ---------- 2. Train small IF on anomalies' neighborhood for SHAP ----------

def train_local_iforest(X):
    # For SHAP we can fit a fresh IF on the same representation;
    # using same parameters as anomaly_isoforest.py
    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    return iso

# ---------- 3. Compute SHAP values ----------

def compute_shap_values(iso_model, X_sample):
    # TreeExplainer works for tree-based models including IsolationForest
    explainer = shap.TreeExplainer(iso_model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values

# ---------- 4. Save per-feature SHAP contributions ----------

def save_shap_to_sqlite(df_anom, shap_values, feature_names):
    # shap_values is (n_samples, n_features)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    out = pd.concat(
        [
            df_anom[["txn_id", "customer_id", "account_id", "amount",
                     "channel", "category", "status", "anomaly_score"]],
            shap_df,
        ],
        axis=1,
    )

    conn = sqlite3.connect(DB_PATH)
    out.to_sql("transactions_anomalies_iforest_shap", conn,
               if_exists="replace", index=False)
    conn.close()

    print("Saved SHAP contributions to table transactions_anomalies_iforest_shap")
    print("Columns:")
    print(out.columns.tolist())

if __name__ == "__main__":
    print("Loading anomalies from SQLite...")
    df_anom = load_scored_transactions(limit_anomalies=200)
    print("Anomalies selected:", len(df_anom))

    print("Building feature matrix for SHAP...")
    X, preproc, num_cols, cat_cols = build_feature_matrix_for_shap(df_anom)

    # Get feature names after one-hot encoding
    ohe = preproc.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(cat_feature_names)

    print("Training local Isolation Forest for SHAP...")
    iso_local = train_local_iforest(X)

    print("Computing SHAP values...")
    shap_vals = compute_shap_values(iso_local, X)

    print("Saving SHAP results to SQLite...")
    save_shap_to_sqlite(df_anom, shap_vals, feature_names)

    print("Done.")
