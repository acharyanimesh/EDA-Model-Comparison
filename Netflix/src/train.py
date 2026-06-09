"""
STEP 2 — Train a Machine Learning Model on the Netflix Dataset
==============================================================
Goal    : Predict a movie's Vote_Average from other features.
Teaches : Feature engineering, one-hot encoding, train/test split,
          training Linear Regression & Random Forest, evaluation metrics.

Run:  python src/train.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 0. Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "mymoviedb (1).csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


# ── 1. Load & Clean ────────────────────────────────────────────────────────────
print("=" * 55)
print(" Loading & cleaning data...")
print("=" * 55)

df = pd.read_csv(CSV_PATH)
df.dropna(subset=["Vote_Average", "Popularity", "Genre", "Release_Date"], inplace=True)

df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"]  = df["Release_Date"].dt.year.fillna(0).astype(int)
df["Month"] = df["Release_Date"].dt.month.fillna(0).astype(int)

df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
df["Vote_Count"]   = pd.to_numeric(df["Vote_Count"],   errors="coerce")
df["Popularity"]   = pd.to_numeric(df["Popularity"],   errors="coerce")

# Cap outliers
df["Popularity"] = np.clip(df["Popularity"], 0, df["Popularity"].quantile(0.995))
df["Vote_Count"] = np.clip(df["Vote_Count"], 0, df["Vote_Count"].quantile(0.995))

df.dropna(subset=["Vote_Average", "Vote_Count", "Popularity", "Year"], inplace=True)
print(f"Dataset size after cleaning: {len(df)} rows")


# ── 2. Feature Engineering ────────────────────────────────────────────────────
print()
print("=" * 55)
print(" Engineering features...")
print("=" * 55)

# 2a. Multi-hot encode Genre
#     "Action, Drama" → [1, 0, 1, 0, ...]
mlb = MultiLabelBinarizer()
genre_lists   = df["Genre"].str.split(", ")
genre_encoded = pd.DataFrame(
    mlb.fit_transform(genre_lists),
    columns=[f"genre_{g}" for g in mlb.classes_],
    index=df.index
)
print(f"  Genres found : {list(mlb.classes_)}")

# 2b. One-hot encode top languages (keep top 8, rest = 'other')
top_langs = df["Original_Language"].value_counts().head(8).index.tolist()
df["Language"] = df["Original_Language"].where(df["Original_Language"].isin(top_langs), other="other")
lang_dummies = pd.get_dummies(df["Language"], prefix="lang")

# 2c. Log-transform skewed numeric features
df["log_Popularity"]  = np.log1p(df["Popularity"])
df["log_Vote_Count"]  = np.log1p(df["Vote_Count"])

# 2d. Assemble final feature matrix
numeric_features = ["log_Popularity", "log_Vote_Count", "Year", "Month"]
X = pd.concat([df[numeric_features], genre_encoded, lang_dummies], axis=1)
y = df["Vote_Average"]

print(f"  Feature matrix shape : {X.shape}")
print(f"  Target (Vote_Average): mean={y.mean():.2f}, std={y.std():.2f}")


# ── 3. Train / Test Split ─────────────────────────────────────────────────────
# 80% for training, 20% for testing — the model never sees test data during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\n  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")


# ── 4. Train Models ───────────────────────────────────────────────────────────
print()
print("=" * 55)
print(" Training models...")
print("=" * 55)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest"    : RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    results[name] = {"preds": preds, "rmse": rmse, "mae": mae, "r2": r2}

    print(f"    RMSE : {rmse:.4f}  (lower = better)")
    print(f"    MAE  : {mae:.4f}  (lower = better)")
    print(f"    R²   : {r2:.4f}  (1.0 = perfect)")


# ── 5. Visualise Results ──────────────────────────────────────────────────────
print()
print("=" * 55)
print(" Saving result charts...")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Actual vs Predicted Vote Average", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res["preds"], alpha=0.3, s=10, color="#4C72B0")
    lims = [y_test.min(), y_test.max()]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Vote Average")
    ax.set_ylabel("Predicted Vote Average")
    ax.set_title(f"{name}\nRMSE={res['rmse']:.3f}  R²={res['r2']:.3f}")
    ax.legend()

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "06_actual_vs_predicted.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

# Feature importance for Random Forest
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top20 = importances.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
top20.plot(kind="barh", ax=ax, color="#DD8452")
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score")
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "07_feature_importance.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

# Residual distribution (how wrong the model is)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Residuals (Predicted − Actual)", fontsize=13, fontweight="bold")

for ax, (name, res) in zip(axes, results.items()):
    residuals = res["preds"] - y_test.values
    ax.hist(residuals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax.set_title(name)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "08_residuals.png")
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"  Saved → {out_path}")

print()
print("All done! Check the 'outputs/' folder for charts.")
