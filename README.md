# Banking ML Preprocessing Pipeline

> **Role:** Senior Data Scientist — Banking Analytics  
> **Language:** R  
> **Use Cases:** Churn Prediction · Credit Risk Scoring · Customer Segmentation · Recommendation System

---

## Overview

This pipeline takes a merged banking dataset (churn + credit risk + segmentation) and produces a fully clean, machine-learning-ready CSV. Every decision is grounded in banking domain logic — not generic data science defaults.

### Input Files
| File | Rows | Columns | Description |
|---|---|---|---|
| `bank_transactions.csv` | 1,048,567 | 9 | Transaction-level records per customer |
| `Churn_Modelling.csv` | 10,000 | 14 | Customer demographics, financials, churn label |
| `Data_Dictionary.xls` | 13 | 3 | Variable names, descriptions, and types |

### Output File
`banking_cleaned_ml_ready.csv` — fully numeric, zero missing values, ML-ready for Random Forest, XGBoost, K-Means, and collaborative filtering.

---

## Pipeline Script

**File:** `banking_ml_preprocessing.R`

Run in R or RStudio:
```r
source("banking_ml_preprocessing.R")
```

All required packages are auto-installed on first run.

---

## Step-by-Step Breakdown

### STEP 0 — Load & Merge Data

The two CSVs are joined on `CustomerID` / `CustomerId` (case-normalised). Column names are converted to `snake_case` via `janitor::clean_names()` for consistency across the pipeline.

**Key function:** `left_join()` — keeps all transaction rows, enriches with churn/demographic data where available.

---

### STEP 1 — Data Inspection

**Goal:** Understand what you have before touching anything.

| Check | Function | What to look for |
|---|---|---|
| Structure | `str()` | Mistyped columns (e.g. age as character) |
| Summary stats | `summary()` | Impossible min/max values |
| Missing values | Custom `summarise` | Columns > 5% missing need a strategy |
| Type audit | `map_chr(df, class)` | Factors stored as character |
| Rich summary | `skimr::skim()` | Distribution shape, completeness |

**Expected output:** A clear inventory of data quality issues before any cleaning.

---

### STEP 2 — Data Quality Checks

**Goal:** Remove noise and enforce domain validity.

#### 2.1 Duplicate Removal
```r
df <- df_raw %>% distinct()
```
Exact-row duplicates removed. In banking data these typically come from ETL join errors.

#### 2.2 Drop Irrelevant Columns
Columns dropped: `transaction_id`, `row_number`, `surname`

These carry **zero predictive signal** for any of the four ML tasks and risk leaking identity information into the model.

#### 2.3 Domain Validity Rules

| Column | Rule | Reason |
|---|---|---|
| `age` | 18–100 | Minors cannot hold accounts; 100+ are data entry errors |
| `credit_score` | 300–850 | FICO scoring band — values outside are system artefacts |
| `balance` | ≥ 0 | Standard deposit accounts cannot go negative (set to 0) |
| `transaction_amount_inr` | > 0 | Zero and negative transactions are voided/reversed — exclude |

---

### STEP 3 — Missing Value Handling

**Goal:** Eliminate NAs without distorting distributions.

| Missingness Level | Type | Strategy | Why |
|---|---|---|---|
| < 5% | Numeric | **Median imputation** | Median is robust to right-skew (income, balance distributions) |
| 5–30% | Numeric | **MICE (PMM)** | Predictive Mean Matching preserves the empirical distribution |
| > 30% | Numeric | **Drop column** | Too much imputed data makes the feature unreliable |
| Any | Categorical < 5% | **Mode** | Most common value is a safe neutral |
| Any | Categorical ≥ 5% | **"Unknown" sentinel** | Preserves the missing pattern as its own category (often predictive) |

```r
# MICE example
mice_out <- mice(mice_input, m = 1, method = "pmm", seed = 42)
df[mice_cols] <- complete(mice_out)
```

**Expected output:** `sum(is.na(df))` equals 0.

---

### STEP 4 — Data Type Correction

**Goal:** Ensure every column has the type its downstream model expects.

| Action | Columns | Function |
|---|---|---|
| Force numeric | `credit_score`, `age`, `balance`, `tenure`, etc. | `as.numeric()` |
| Force integer (0/1) | `has_cr_card`, `is_active_member`, `exited` | `as.integer()` |
| Convert to factor | `geography`, `gender`, `cust_location`, `cust_gender` | `as.factor()` |
| Parse dates | `customer_dob`, `transaction_date` | `lubridate::parse_date_time()` |

Date parsing also derives new features:

```r
txn_year   = year(txn_date_parsed)
txn_month  = month(txn_date_parsed)
txn_dow    = wday(txn_date_parsed)   # day of week (1 = Sunday)
age_from_dob = difftime(Sys.Date(), dob, units="days") / 365.25
```

---

### STEP 5 — Outlier Detection & Treatment

**Goal:** Control extreme values without losing legitimate high-value customers.

#### Why 3× IQR instead of the standard 1.5×?

In retail banking, **high-net-worth (HNW) customers** have legitimately extreme balances and incomes. Using 1.5× IQR would silently delete your most valuable segment. The 3× fence catches only true data errors.

#### Method: Winsorization (Capping)
Values are **capped to the fence**, not removed. This:
- Preserves the customer record
- Limits the distortion without deleting signal
- Is standard practice in credit risk model development (Basel III guidelines)

```r
df[[col]] <- pmax(pmin(df[[col]], upper_fence), lower_fence)
```

#### Fraud Signal Preservation
```r
high_value_txn_flag = as.integer(transaction_amount_inr > quantile(..., 0.99))
```
The top 1% of transactions are flagged **before** capping so the extreme-value signal is not lost.

**Columns treated:** `balance`, `cust_account_balance`, `estimated_salary`, `transaction_amount_inr`, `credit_score`

---

### STEP 6 — Feature Engineering

**Goal:** Create banking-domain features that raw columns cannot express alone.

#### `income_to_balance_ratio`
```r
estimated_salary / (balance + 1)
```
A customer with high income but low balance is likely banking elsewhere — a strong churn predictor. A low-income, high-balance customer is a loyal saver with low risk.

#### `credit_utilization_rate`
```r
pmin(balance / (estimated_salary + 1), 1.0)
```
Mirrors the **FICO credit utilization component**. Utilization above 70% correlates strongly with default risk. Capped at 100% to prevent ratio explosion.

#### `activity_score` (0–100)
```r
rescale(num_of_products, to=c(0,40)) +
rescale(has_cr_card,      to=c(0,30)) +
rescale(is_active_member, to=c(0,30))
```
Multi-product customers are **4× less likely to churn** (Accenture Banking Study). Combines product count, card ownership, and active-member status into one composite score.

#### `risk_index` (0–100)
```r
rescale(1000 - credit_score,      to=c(0,50)) +   # inverted: lower score = higher risk
rescale(pmax(10 - tenure, 0),     to=c(0,30)) +   # short tenure = higher risk
rescale(credit_utilization_rate,  to=c(0,20))
```
A composite risk signal. Useful as both a standalone feature and as a target proxy for unsupervised segmentation.

#### `engagement_score` (0–100)
```r
rescale(tenure,          to=c(0,50)) +
rescale(num_of_products, to=c(0,30)) +
rescale(activity_score,  to=c(0,20))
```
Drives recommendation system personalisation. High-engagement customers respond better to cross-sell; low-engagement customers need retention offers.

#### Transaction Aggregates (per customer)
| Feature | Formula | Signal |
|---|---|---|
| `avg_txn_amount` | `mean(transaction_amount_inr)` | Spending level |
| `max_txn_amount` | `max(transaction_amount_inr)` | Peak transaction size |
| `txn_count` | `n()` per customer | Transaction frequency |
| `txn_amount_stddev` | `sd(transaction_amount_inr)` | Spending volatility |

#### `age_band`
Cuts age into 6 bands: `18-25`, `26-35`, `36-45`, `46-55`, `56-65`, `65+`. Age has a non-linear relationship with churn and risk — banding captures this better than a raw numeric for tree-based models.

---

### STEP 7 — Encoding for ML

**Goal:** Convert all categoricals to numbers without introducing spurious ordinal relationships.

| Column Type | Strategy | Why |
|---|---|---|
| Low-cardinality nominal (≤ 10 levels) | **One-hot encoding** (`fastDummies`) | No ordinal assumption; `remove_first_dummy=TRUE` prevents multicollinearity |
| High-cardinality nominal (`cust_location`) | **Frequency encoding** | Avoids dimensionality explosion; preserves location popularity signal |
| Ordinal (`age_band`) | **Label encoding** (integer 1–6) | Order is meaningful; OHE would destroy that |
| Binary (`has_cr_card`, `exited`, etc.) | Already 0/1 integer | No further encoding needed |

```r
# One-hot
df_encoded <- fastDummies::dummy_cols(
  df,
  select_columns          = ohe_candidates,
  remove_first_dummy      = TRUE,
  remove_selected_columns = TRUE
)

# Frequency encode
freq_map <- df %>%
  count(cust_location) %>%
  mutate(cust_location_freq = n / nrow(df))
```

**Expected output:** Zero non-numeric columns remaining.

---

### STEP 8 — Final Clean Dataset Output

**Goal:** Validate and save the production-ready dataset.

#### Final Validation Checklist

| Check | Function | Pass Condition |
|---|---|---|
| Missing values | `sum(is.na(df_final))` | = 0 |
| Duplicate rows | `sum(duplicated(df_final))` | = 0 |
| Constant columns | `janitor::remove_constant()` | Removed automatically |
| Near-zero variance | `caret::nearZeroVar()` | Flagged and removed |
| High correlation | Custom pairs check | Pairs with \|r\| > 0.85 reported |

#### Column Manifest
The script prints a full manifest at the end, categorising every column:

| Tag | Meaning |
|---|---|
| `TARGET: churn` | `exited` — binary churn label |
| `TARGET/FEATURE: credit risk` | Risk-related columns |
| `TARGET: segmentation` | Cluster/segment labels |
| `ENGINEERED feature` | Created in Step 6 |
| `DERIVED feature` | Flags, encodings, frequencies |
| `ENCODED demographic` | One-hot columns |
| `RAW feature` | Original dataset columns |

#### Save
```r
write_csv(df_final, "banking_cleaned_ml_ready.csv")
```

---

## ML Compatibility

| Model Type | Compatible | Notes |
|---|---|---|
| Random Forest (`randomForest`, `ranger`) | ✅ | All numeric; handles scale differences natively |
| XGBoost (`xgboost`) | ✅ | Requires matrix input — use `as.matrix(df_final)` |
| K-Means Clustering | ✅ | Recommend scaling first: `scale(df_final)` |
| Logistic Regression | ✅ | Check for high-correlation pairs from Step 8 report |
| Collaborative Filtering | ✅ | Use `engagement_score` + transaction aggregates as user vectors |

---

## Dependencies

```r
install.packages(c(
  "tidyverse", "janitor", "skimr", "DataExplorer",
  "mice", "VIM", "caret", "DescTools", "lubridate",
  "fastDummies", "corrplot", "ggplot2", "scales"
))
```

R version ≥ 4.1.0 recommended.

---

## File Structure

```
project/
├── bank_transactions.csv          # Raw input
├── Churn_Modelling.csv            # Raw input
├── Data_Dictionary.xls            # Reference
├── banking_ml_preprocessing.R     # This pipeline
├── banking_cleaned_ml_ready.csv   # Output (generated)
└── README.md                      # This file
```

---

## Quick Reference Card

```
STEP 1  Inspect          str · summary · skim · missing %
STEP 2  Quality          dedup · drop IDs · domain bounds
STEP 3  Impute           median · MICE · mode · "Unknown"
STEP 4  Fix types        numeric · integer · factor · dates
STEP 5  Outliers         3× IQR winsorize · fraud flag
STEP 6  Engineer         5 composite scores · txn aggs · age bands
STEP 7  Encode           one-hot · freq · label
STEP 8  Validate & save  NZV · correlation · write_csv
```
