"""
STEP 1 — Explore & Visualize the Netflix Movie Dataset
======================================================
Teaches you:
  - How to load CSV data with pandas
  - How to clean messy data (missing values, wrong types)
  - How to draw histograms and bar graphs with matplotlib + seaborn
  - How to handle multi-value columns like Genre

Run:  python src/visualize.py
Output: PNG charts saved in the outputs/ folder
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # force interactive window backend on Windows
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ── 0. Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "mymoviedb (1).csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120


# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("=" * 55)
print(" Loading dataset...")
print("=" * 55)

df = pd.read_csv(CSV_PATH, encoding="utf-8", on_bad_lines="skip", engine="python")

print(f"Shape        : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Columns      : {list(df.columns)}")
print()
print("First 3 rows:")
print(df.head(3).to_string())
print()


# ── 2. Clean Data ──────────────────────────────────────────────────────────────
print("=" * 55)
print(" Cleaning data...")
print("=" * 55)

# 2a. Drop rows where important columns are missing
before = len(df)
df.dropna(subset=["Vote_Average", "Popularity", "Genre", "Release_Date"], inplace=True)
print(f"Dropped {before - len(df)} rows with missing critical values.")

# 2b. Convert Release_Date to datetime and extract year
df["Release_Date"] = pd.to_datetime(df["Release_Date"], errors="coerce")
df["Year"] = df["Release_Date"].dt.year.astype("Int64")

# 2c. Cast numeric columns (they might have been read as strings if messy)
df["Vote_Average"] = pd.to_numeric(df["Vote_Average"], errors="coerce")
df["Vote_Count"]   = pd.to_numeric(df["Vote_Count"],   errors="coerce")
df["Popularity"]   = pd.to_numeric(df["Popularity"],   errors="coerce")

# 2d. Remove obvious outliers in Popularity (top 0.5%)
pop_cap = df["Popularity"].quantile(0.995)
df = df[df["Popularity"] <= pop_cap]

print(f"Final dataset : {len(df)} rows")
print()

# Show quick stats
print("Numeric summary:")
print(df[["Vote_Average", "Vote_Count", "Popularity"]].describe().round(2))
print()


# ── 3. Genre Helper ────────────────────────────────────────────────────────────
# Genre is a comma-separated string like "Action, Adventure, Science Fiction"
# We explode it so each genre gets its own row for counting.

df_genres = (
    df["Genre"]
    .dropna()
    .str.split(", ")
    .explode()
    .str.strip()
)
genre_counts = df_genres.value_counts()


# ── 4. HISTOGRAMS ─────────────────────────────────────────────────────────────
print("=" * 55)
print(" Drawing histograms...")
print("=" * 55)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Netflix Dataset — Distribution Histograms", fontsize=16, fontweight="bold")

# 4a. Vote Average
ax = axes[0, 0]
ax.hist(df["Vote_Average"].dropna(), bins=30, color="#4C72B0", edgecolor="white", linewidth=0.6)
ax.set_title("Vote Average Distribution")
ax.set_xlabel("Vote Average (0–10)")
ax.set_ylabel("Number of Movies")
mean_va = df["Vote_Average"].mean()
ax.axvline(mean_va, color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {mean_va:.2f}")
ax.legend()

# 4b. Popularity (log scale helps because it's heavily right-skewed)
ax = axes[0, 1]
log_pop = np.log1p(df["Popularity"].dropna())   # log1p = log(x+1) avoids log(0)
ax.hist(log_pop, bins=40, color="#DD8452", edgecolor="white", linewidth=0.6)
ax.set_title("Popularity Distribution (log scale)")
ax.set_xlabel("log(1 + Popularity)")
ax.set_ylabel("Number of Movies")

# 4c. Vote Count (log scale)
ax = axes[1, 0]
log_vc = np.log1p(df["Vote_Count"].dropna())
ax.hist(log_vc, bins=40, color="#55A868", edgecolor="white", linewidth=0.6)
ax.set_title("Vote Count Distribution (log scale)")
ax.set_xlabel("log(1 + Vote Count)")
ax.set_ylabel("Number of Movies")

# 4d. Release Year
ax = axes[1, 1]
years = df["Year"].dropna()
ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2), color="#C44E52", edgecolor="white", linewidth=0.3)
ax.set_title("Movies per Release Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "01_histograms.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"  Saved -> {out_path}")


# ── 5. BAR GRAPHS ─────────────────────────────────────────────────────────────
print(" Drawing bar graphs...")

# 5a. Top 15 Genres by count
fig, ax = plt.subplots(figsize=(12, 6))
top_genres = genre_counts.head(15)
bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=sns.color_palette("muted", 15))
ax.set_title("Top 15 Genres by Number of Movies", fontsize=14, fontweight="bold")
ax.set_xlabel("Number of Movies")
for bar, val in zip(bars, top_genres.values[::-1]):
    ax.text(val + 20, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=9)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "02_top_genres.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"  Saved -> {out_path}")


# 5b. Top 10 Languages by count
fig, ax = plt.subplots(figsize=(10, 5))
lang_counts = df["Original_Language"].value_counts().head(10)
colors = sns.color_palette("pastel", len(lang_counts))
ax.bar(lang_counts.index, lang_counts.values, color=colors, edgecolor="grey", linewidth=0.5)
ax.set_title("Top 10 Original Languages", fontsize=14, fontweight="bold")
ax.set_xlabel("Language Code")
ax.set_ylabel("Number of Movies")
for i, val in enumerate(lang_counts.values):
    ax.text(i, val + 15, str(val), ha="center", fontsize=9)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "03_top_languages.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"  Saved -> {out_path}")


# 5c. Average Vote Score per Top 15 Genre
# Join genres back to vote averages for per-genre stats
df_exploded = df[["Vote_Average", "Genre"]].copy()
df_exploded["Genre"] = df_exploded["Genre"].str.split(", ")
df_exploded = df_exploded.explode("Genre").dropna()
genre_vote_avg = (
    df_exploded.groupby("Genre")["Vote_Average"]
    .agg(["mean", "count"])
    .query("count >= 50")          # only genres with ≥50 movies
    .sort_values("mean", ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(12, 6))
palette = sns.color_palette("coolwarm", len(genre_vote_avg))
bars = ax.barh(genre_vote_avg.index[::-1], genre_vote_avg["mean"].values[::-1], color=palette)
ax.set_title("Average Vote Score by Genre (genres with ≥50 movies)", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Vote Score")
ax.set_xlim(0, 10)
for bar, val in zip(bars, genre_vote_avg["mean"].values[::-1]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "04_avg_vote_by_genre.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"  Saved -> {out_path}")


# 5d. Top 20 most productive years (number of movies released)
fig, ax = plt.subplots(figsize=(13, 5))
year_counts = df["Year"].value_counts().sort_index()
# Only years with real data (1900–2025)
year_counts = year_counts[(year_counts.index >= 1900) & (year_counts.index <= 2025)]
ax.bar(year_counts.index.astype(int), year_counts.values, color="#4C72B0", edgecolor="white", linewidth=0.3)
ax.set_title("Number of Movies Released per Year", fontsize=14, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "05_movies_per_year.png")
plt.savefig(out_path, bbox_inches="tight")
plt.show()
plt.close()
print(f"  Saved -> {out_path}")


print()
print("All charts saved to the 'outputs/' folder.")
print("Open them to see your data visually!")
