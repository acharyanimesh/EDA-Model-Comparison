# Chess Winner Prediction

**Author:** Animesh Acharya  
**Last Updated:** 2026-06-09  
**Notebook:** `Chess/chess_train.ipynb`

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Pipeline](#pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Models & Results](#models--results)
6. [Output Files](#output-files)
7. [Dependencies](#dependencies)

---

## Overview

End-to-end classification pipeline that predicts the outcome of a chess game — **white win**, **black win**, or **draw** — from game-level metadata. The pipeline covers EDA, feature engineering, training three classifiers, cross-validation, and result visualisation.

---

## Dataset

**File:** `Chess/games.csv` (7.5 MB)  
**Shape:** 20,058 games × 16 columns  
**Source archive:** `Chess/archive (1).zip`

| Column | Type | Description |
|---|---|---|
| `id` | str | Unique game identifier |
| `rated` | bool | Whether the game was rated |
| `created_at` | float | Game start timestamp (ms) |
| `last_move_at` | float | Last move timestamp (ms) |
| `turns` | int | Number of half-moves (plies) |
| `victory_status` | str | How the game ended: `mate`, `resign`, `outoftime`, `draw` |
| `winner` | str | **Target** — `white`, `black`, or `draw` |
| `increment_code` | str | Time control (e.g. `5+10`) |
| `white_id` | str | Username of white player |
| `white_rating` | int | ELO rating of white player |
| `black_id` | str | Username of black player |
| `black_rating` | int | ELO rating of black player |
| `moves` | str | Full move list in algebraic notation |
| `opening_eco` | str | ECO code of the opening played |
| `opening_name` | str | Full name of the opening |
| `opening_ply` | int | Number of moves in the opening phase |

**Class distribution (target):**

| Winner | Count | % |
|---|---|---|
| White | ~10,000 | ~49.9% |
| Black | ~9,107 | ~45.4% |
| Draw | ~951 | ~4.7% |

**Train / test split:** 16,046 training samples / 4,012 test samples (80/20, stratified)

---

## Pipeline

```
games.csv
   │
   ├── 1. Load & inspect (shape, dtypes, head)
   ├── 2. EDA (distributions, victory status, ratings, top openings)
   ├── 3. Feature engineering (encoding, derived features)
   ├── 4. Train-test split (stratified, random_state=42)
   ├── 5. Train 3 classifiers
   ├── 6. Evaluate (accuracy, 5-fold CV, classification report, confusion matrix)
   └── 7. Visualise results → eda_plots.png, model_results.png
```

---

## Feature Engineering

From the raw 16 columns, 8 features are selected/derived for modelling:

| Feature | Source | Notes |
|---|---|---|
| `rated` | `rated` column | Bool → int (1/0) |
| `turns` | `turns` | Used as-is |
| `white_rating` | `white_rating` | Used as-is |
| `black_rating` | `black_rating` | Used as-is |
| `rating_diff` | `white_rating − black_rating` | Derived |
| `avg_rating` | `(white_rating + black_rating) / 2` | Derived |
| `opening_eco_enc` | `opening_eco` | Label-encoded ECO code |
| `opening_ply` | `opening_ply` | Used as-is |

Columns dropped from modelling: `id`, `created_at`, `last_move_at`, `increment_code`, `white_id`, `black_id`, `moves`, `opening_name`, `victory_status` (would be data leakage or irrelevant).

---

## Models & Results

Three scikit-learn classifiers are trained and compared:

| Model | Configuration | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|---|
| Logistic Regression | `max_iter=500, random_state=42` | 62.41% | 62.33% |
| Random Forest | `n_estimators=200, random_state=42, n_jobs=-1` | 64.88% | 64.78% |
| **Gradient Boosting** | **`n_estimators=200, random_state=42`** | **80.81%** | **79.54%** |

**Best model:** Gradient Boosting Classifier — 80.81% test accuracy

### Per-class breakdown (Gradient Boosting)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| black | 0.8141 | 0.8172 | 0.8157 | 1,822 |
| draw | 0.5000 | 0.0421 | 0.0777 | 190 |
| white | 0.8053 | 0.8725 | 0.8375 | 2,000 |

Note: `draw` is the hardest class to predict due to severe class imbalance (~4.7% of games).

### Feature importance (Random Forest)

Top features by importance: `white_rating`, `black_rating`, `rating_diff`, `turns`, `avg_rating`. Rating-based features dominate — the skill gap between players is the strongest predictor of outcome.

---

## Output Files

| File | Description |
|---|---|
| `Chess/eda_plots.png` | 6-panel EDA chart: winner distribution, victory status pie chart, rating histograms, game length, wins by rated status, top 10 ECO openings |
| `Chess/model_results.png` | Model accuracy comparison bar chart, confusion matrices for all 3 models, Random Forest feature importance, violin plot of rating diff vs winner |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Run:

```bash
cd "d:\ANIMESH\ML projects\Chess"
jupyter notebook chess_train.ipynb
```
