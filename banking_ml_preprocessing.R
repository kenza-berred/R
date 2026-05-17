# ============================================================
#  BANKING ML PREPROCESSING PIPELINE
#  Senior Data Scientist — Production Grade
#  Covers: Churn | Credit Risk | Segmentation | Recommendation
# ============================================================

# ── Dependencies ────────────────────────────────────────────
required_packages <- c(
  "tidyverse", "janitor", "skimr", "DataExplorer",
  "mice", "VIM", "caret", "DescTools", "lubridate",
  "fastDummies", "corrplot", "ggplot2", "scales"
)

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
invisible(lapply(required_packages, install_if_missing))

library(tidyverse)
library(janitor)
library(skimr)
library(mice)
library(caret)
library(DescTools)
library(lubridate)
library(fastDummies)
library(corrplot)
library(ggplot2)
library(scales)


# ============================================================
# STEP 0 — LOAD DATA
# ============================================================
# Adjust paths to match your merged file
transactions <- read_csv("bank_transactions.csv")
churn        <- read_csv("Churn_Modelling.csv")

# --- Merge on CustomerID (case-insensitive alignment) -------
churn <- churn %>% rename(CustomerID = CustomerId)

df_raw <- transactions %>%
  left_join(churn, by = "CustomerID") %>%
  clean_names()   # snake_case all column names (janitor)

cat("\n✅ Raw merged dataset dimensions:", dim(df_raw), "\n")


# ============================================================
# STEP 1 — DATA INSPECTION
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 1 — DATA INSPECTION\n")
cat(strrep("=", 60), "\n")

# 1.1 Structure
cat("\n--- 1.1 Structure ---\n")
str(df_raw)

# 1.2 Summary statistics
cat("\n--- 1.2 Summary Statistics ---\n")
print(summary(df_raw))

# 1.3 Missing values per column (sorted descending)
cat("\n--- 1.3 Missing Values ---\n")
missing_summary <- df_raw %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "missing_count") %>%
  mutate(
    pct_missing = round(missing_count / nrow(df_raw) * 100, 2)
  ) %>%
  filter(missing_count > 0) %>%
  arrange(desc(pct_missing))

print(missing_summary)

# 1.4 Data type audit
cat("\n--- 1.4 Data Type Audit ---\n")
type_audit <- df_raw %>%
  summarise(across(everything(), class)) %>%
  pivot_longer(everything(), names_to = "column", values_to = "detected_type")
print(type_audit, n = Inf)

# 1.5 Quick skim (rich summary)
cat("\n--- 1.5 Skim Summary ---\n")
print(skim(df_raw))

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • Columns with > 5 % missing flagged for imputation
# • Mistyped columns (e.g. age stored as character) identified
# • Numeric cols with suspicious min/max values visible


# ============================================================
# STEP 2 — DATA QUALITY CHECKS
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 2 — DATA QUALITY CHECKS\n")
cat(strrep("=", 60), "\n")

# 2.1 Duplicate rows
cat("\n--- 2.1 Duplicate Detection ---\n")
n_dupes <- sum(duplicated(df_raw))
cat("Total duplicate rows:", n_dupes, "\n")
df <- df_raw %>% distinct()
cat("Rows after deduplication:", nrow(df), "\n")

# 2.2 Drop irrelevant / leakage columns
#     IDs carry no signal for ML; row numbers are artifacts
cat("\n--- 2.2 Dropping ID / Index Columns ---\n")
id_cols <- c("transaction_id", "row_number", "surname")
id_cols_present <- intersect(id_cols, names(df))
cat("Dropping:", paste(id_cols_present, collapse = ", "), "\n")
df <- df %>% select(-any_of(id_cols_present))

# 2.3 Inconsistent / impossible values
cat("\n--- 2.3 Inconsistency Checks ---\n")

# Age: banking customers must be 18–100
if ("age" %in% names(df)) {
  bad_age <- df %>% filter(!is.na(age) & (age < 18 | age > 100))
  cat("Age out of [18, 100]:", nrow(bad_age), "rows\n")
  df <- df %>% filter(is.na(age) | (age >= 18 & age <= 100))
}

# Credit score: FICO range 300–850
if ("credit_score" %in% names(df)) {
  bad_cs <- df %>% filter(!is.na(credit_score) & (credit_score < 300 | credit_score > 850))
  cat("Credit score out of [300, 850]:", nrow(bad_cs), "rows\n")
  df <- df %>% filter(is.na(credit_score) | between(credit_score, 300, 850))
}

# Balance: cannot be negative in a standard deposit account
if ("balance" %in% names(df)) {
  neg_bal <- df %>% filter(!is.na(balance) & balance < 0)
  cat("Negative balance rows:", nrow(neg_bal), "— setting to 0\n")
  df <- df %>% mutate(balance = ifelse(balance < 0, 0, balance))
}

# Transaction amount: no zero or negative
if ("transaction_amount_inr" %in% names(df)) {
  bad_txn <- df %>% filter(!is.na(transaction_amount_inr) & transaction_amount_inr <= 0)
  cat("Non-positive transaction amounts:", nrow(bad_txn), "rows — removed\n")
  df <- df %>% filter(is.na(transaction_amount_inr) | transaction_amount_inr > 0)
}

cat("Rows after quality checks:", nrow(df), "\n")

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • Zero duplicate rows after step 2.1
# • All ages in valid range, credit scores in FICO band
# • No negative balances; no zero-value transactions


# ============================================================
# STEP 3 — MISSING VALUE HANDLING
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 3 — MISSING VALUE HANDLING\n")
cat(strrep("=", 60), "\n")

# Strategy:
#   Numeric  < 5 % missing  → median imputation (robust to skew)
#   Numeric  5–30 % missing → predictive MICE imputation
#   Numeric  > 30 % missing → drop column (too much information loss)
#   Categorical              → mode or "Unknown" sentinel

# 3.1 Identify numeric vs categorical columns
num_cols  <- df %>% select(where(is.numeric))  %>% names()
cat_cols  <- df %>% select(where(is.character)) %>% names()

# 3.2 Numeric: compute missing %
num_missing_pct <- df %>%
  select(all_of(num_cols)) %>%
  summarise(across(everything(), ~ mean(is.na(.)) * 100)) %>%
  pivot_longer(everything(), names_to = "col", values_to = "pct") %>%
  filter(pct > 0) %>%
  arrange(desc(pct))

cat("\nNumeric columns with missing values:\n")
print(num_missing_pct)

# Drop numeric cols > 30 % missing
drop_high_missing <- num_missing_pct %>% filter(pct > 30) %>% pull(col)
if (length(drop_high_missing) > 0) {
  cat("Dropping (>30% missing):", paste(drop_high_missing, collapse = ", "), "\n")
  df <- df %>% select(-all_of(drop_high_missing))
  num_cols <- setdiff(num_cols, drop_high_missing)
}

# Median imputation for cols with < 5 % missing
low_missing_num <- num_missing_pct %>% filter(pct < 5) %>% pull(col)
low_missing_num <- intersect(low_missing_num, names(df))

df <- df %>%
  mutate(across(
    all_of(low_missing_num),
    ~ ifelse(is.na(.), median(., na.rm = TRUE), .)
  ))
cat("Median-imputed:", paste(low_missing_num, collapse = ", "), "\n")

# MICE for cols with 5–30 % missing
mice_cols <- num_missing_pct %>% filter(pct >= 5 & pct <= 30) %>% pull(col)
mice_cols <- intersect(mice_cols, names(df))

if (length(mice_cols) > 0) {
  cat("\nRunning MICE imputation for:", paste(mice_cols, collapse = ", "), "\n")
  mice_input <- df %>% select(all_of(mice_cols))
  mice_out   <- mice(mice_input, m = 1, method = "pmm",
                     seed = 42, printFlag = FALSE)
  df[mice_cols] <- complete(mice_out)
  cat("MICE imputation complete.\n")
}

# 3.3 Categorical: mode or "Unknown"
mode_fn <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

for (col in cat_cols) {
  if (col %in% names(df) && sum(is.na(df[[col]])) > 0) {
    missing_pct <- mean(is.na(df[[col]])) * 100
    if (missing_pct < 5) {
      df[[col]][is.na(df[[col]])] <- mode_fn(df[[col]])
      cat("Mode-imputed:", col, "\n")
    } else {
      df[[col]][is.na(df[[col]])] <- "Unknown"
      cat("'Unknown' sentinel applied:", col, "\n")
    }
  }
}

cat("\nMissing values remaining:", sum(is.na(df)), "\n")

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • 0 missing values in final dataset
# • High-missing cols dropped; MICE used for moderate missingness
# • Categoricals have no NAs — mode or "Unknown" filled


# ============================================================
# STEP 4 — DATA TYPE CORRECTION
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 4 — DATA TYPE CORRECTION\n")
cat(strrep("=", 60), "\n")

# 4.1 Force-numeric columns that may have been read as character
force_numeric <- c(
  "credit_score", "age", "tenure", "balance",
  "num_of_products", "estimated_salary",
  "cust_account_balance", "transaction_amount_inr"
)
for (col in intersect(force_numeric, names(df))) {
  df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
}
cat("Forced to numeric:", paste(intersect(force_numeric, names(df)), collapse = ", "), "\n")

# 4.2 Convert binary / ordinal flags to integer (0/1)
binary_cols <- c("has_cr_card", "is_active_member", "exited")
for (col in intersect(binary_cols, names(df))) {
  df[[col]] <- as.integer(df[[col]])
}
cat("Binary columns set to integer:", paste(intersect(binary_cols, names(df)), collapse = ", "), "\n")

# 4.3 Convert nominal categoricals to factor
nominal_cols <- c("geography", "gender", "cust_location", "cust_gender")
for (col in intersect(nominal_cols, names(df))) {
  df[[col]] <- as.factor(df[[col]])
}
cat("Converted to factor:", paste(intersect(nominal_cols, names(df)), collapse = ", "), "\n")

# 4.4 Parse date/time fields
if ("customer_dob" %in% names(df)) {
  df <- df %>%
    mutate(
      cust_dob_parsed = parse_date_time(customer_dob,
                                        orders = c("dmy", "mdy", "ymd"),
                                        quiet  = TRUE),
      age_from_dob    = as.integer(
        difftime(Sys.Date(), cust_dob_parsed, units = "days") / 365.25
      )
    )
  cat("customer_dob parsed; age_from_dob derived\n")
}

if ("transaction_date" %in% names(df)) {
  df <- df %>%
    mutate(
      txn_date_parsed = parse_date_time(transaction_date,
                                        orders = c("dmy", "mdy", "ymd"),
                                        quiet  = TRUE),
      txn_year        = year(txn_date_parsed),
      txn_month       = month(txn_date_parsed),
      txn_dow         = wday(txn_date_parsed, label = FALSE)   # 1=Sun
    )
  cat("transaction_date parsed; year/month/dow extracted\n")
}

cat("\nColumn types after correction:\n")
df %>%
  summarise(across(everything(), class)) %>%
  pivot_longer(everything(), names_to = "col", values_to = "type") %>%
  print(n = Inf)

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • All numeric cols are <dbl> or <int>
# • Geography, Gender are <fct>
# • Date fields parsed; temporal features derived


# ============================================================
# STEP 5 — OUTLIER DETECTION & TREATMENT
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 5 — OUTLIER DETECTION (IQR)\n")
cat(strrep("=", 60), "\n")

# Banking rationale:
#   - Balance / income can be legitimately very large (HNW customers)
#     → CAPPING preferred over removal (preserve the customer)
#   - Transaction amount extremes may be fraud signals
#     → Flag + cap, don't silently delete
#   - Age, tenure, credit score have hard domain bounds already
#     applied in Step 2

iqr_cap <- function(x, multiplier = 3.0) {
  # Banking uses 3× IQR (more lenient than 1.5×) to avoid
  # removing genuine HNW / large-corporate customers
  Q1  <- quantile(x, 0.25, na.rm = TRUE)
  Q3  <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower <- Q1 - multiplier * IQR
  upper <- Q3 + multiplier * IQR
  list(lower = lower, upper = upper,
       n_low  = sum(x < lower, na.rm = TRUE),
       n_high = sum(x > upper, na.rm = TRUE))
}

# Columns to cap (continuous financial features)
cap_cols <- intersect(
  c("balance", "cust_account_balance", "estimated_salary",
    "transaction_amount_inr", "credit_score"),
  names(df)
)

outlier_report <- tibble(
  column      = character(),
  lower_fence = numeric(),
  upper_fence = numeric(),
  n_capped_low  = integer(),
  n_capped_high = integer()
)

for (col in cap_cols) {
  stats <- iqr_cap(df[[col]])
  df[[col]] <- pmax(pmin(df[[col]], stats$upper), stats$lower)

  outlier_report <- outlier_report %>%
    add_row(
      column        = col,
      lower_fence   = stats$lower,
      upper_fence   = stats$upper,
      n_capped_low  = as.integer(stats$n_low),
      n_capped_high = as.integer(stats$n_high)
    )
}

cat("\nOutlier capping report (3× IQR):\n")
print(outlier_report)

# Flag extreme transaction amounts as potential fraud signal
if ("transaction_amount_inr" %in% names(df)) {
  q99 <- quantile(df$transaction_amount_inr, 0.99, na.rm = TRUE)
  df  <- df %>%
    mutate(high_value_txn_flag = as.integer(transaction_amount_inr > q99))
  cat("\nhigh_value_txn_flag created (top 1% transactions)\n")
}

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • Outlier report shows how many values were winsorized per col
# • No values deleted — HNW customers preserved via capping
# • high_value_txn_flag usable as a fraud/risk feature


# ============================================================
# STEP 6 — FEATURE ENGINEERING
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 6 — FEATURE ENGINEERING\n")
cat(strrep("=", 60), "\n")

# ── 6.1 income_to_balance_ratio ──────────────────────────────
# Why: High income + low balance → possible churn (money going elsewhere)
#      Low income + high balance → loyal saver, low risk
if (all(c("estimated_salary", "balance") %in% names(df))) {
  df <- df %>%
    mutate(
      income_to_balance_ratio = ifelse(
        balance > 0,
        estimated_salary / (balance + 1),   # +1 avoids /0 for zero-balance
        estimated_salary
      )
    )
  cat("✅ income_to_balance_ratio created\n")
}

# ── 6.2 credit_utilization_rate ─────────────────────────────
# Why: Core credit risk metric. > 70 % → elevated default probability
#      Mirrors real-world FICO scoring methodology
if (all(c("balance", "estimated_salary") %in% names(df))) {
  df <- df %>%
    mutate(
      credit_utilization_rate = ifelse(
        estimated_salary > 0,
        pmin(balance / (estimated_salary + 1), 1),  # cap at 100%
        0
      )
    )
  cat("✅ credit_utilization_rate created\n")
}

# ── 6.3 activity_score ──────────────────────────────────────
# Why: Combines product usage + card ownership + active status
#      Multi-product customers are 4× less likely to churn (Accenture)
if (all(c("num_of_products", "has_cr_card", "is_active_member") %in% names(df))) {
  df <- df %>%
    mutate(
      activity_score = (
        rescale(num_of_products,   to = c(0, 40)) +
        rescale(has_cr_card,       to = c(0, 30)) +
        rescale(is_active_member,  to = c(0, 30))
      )
    )
  cat("✅ activity_score created (0–100 scale)\n")
}

# ── 6.4 risk_index ──────────────────────────────────────────
# Why: Composite risk signal combining credit score, tenure, utilization
#      Low credit score + short tenure + high utilization = high risk
if (all(c("credit_score", "tenure") %in% names(df))) {
  df <- df %>%
    mutate(
      risk_index = (
        rescale(1000 - credit_score,       to = c(0, 50)) +  # invert: higher score = lower risk
        rescale(pmax(10 - tenure, 0),      to = c(0, 30)) +  # short tenure = higher risk
        if_else(
          "credit_utilization_rate" %in% names(df),
          rescale(credit_utilization_rate, to = c(0, 20)),
          0
        )
      )
    )
  cat("✅ risk_index created (0–100 scale)\n")
}

# ── 6.5 engagement_score ────────────────────────────────────
# Why: Differentiates transactional customers from deeply engaged ones
#      High engagement → lower churn propensity; key for recommendation
if (all(c("tenure", "num_of_products") %in% names(df))) {
  df <- df %>%
    mutate(
      engagement_score = (
        rescale(tenure,           to = c(0, 50)) +
        rescale(num_of_products,  to = c(0, 30)) +
        if_else(
          "activity_score" %in% names(df),
          rescale(activity_score, to = c(0, 20)),
          0
        )
      )
    )
  cat("✅ engagement_score created (0–100 scale)\n")
}

# ── 6.6 Transaction-level aggregates (from transactions file) ──
# Only relevant if multiple transaction rows per customer remain
if ("transaction_amount_inr" %in% names(df)) {
  df <- df %>%
    group_by(customer_id) %>%
    mutate(
      avg_txn_amount    = mean(transaction_amount_inr, na.rm = TRUE),
      max_txn_amount    = max(transaction_amount_inr,  na.rm = TRUE),
      txn_count         = n(),
      txn_amount_stddev = sd(transaction_amount_inr,   na.rm = TRUE)
    ) %>%
    ungroup()
  cat("✅ Transaction aggregates created (avg, max, count, stddev)\n")
}

# ── 6.7 Age segmentation (ordinal band) ─────────────────────
if ("age" %in% names(df)) {
  df <- df %>%
    mutate(
      age_band = cut(
        age,
        breaks = c(17, 25, 35, 45, 55, 65, Inf),
        labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "65+"),
        right  = TRUE
      )
    )
  cat("✅ age_band created\n")
}

cat("\nNew features added. Dataset now has", ncol(df), "columns.\n")

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • 6+ new banking-domain features appended
# • All on comparable numeric scales (0–100 or ratio)
# • age_band as ordered factor for tree-based models


# ============================================================
# STEP 7 — ENCODING FOR ML
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 7 — ENCODING FOR ML\n")
cat(strrep("=", 60), "\n")

# Strategy:
#   Low-cardinality nominal (<=10 levels) → one-hot (dummy)
#   High-cardinality nominal              → target / frequency encode
#   Ordinal (age_band)                   → label encode (integer)
#   Binary (0/1)                         → already integer

# 7.1 Drop raw date columns now that features are extracted
date_raw_cols <- c("customer_dob", "transaction_date",
                   "cust_dob_parsed", "txn_date_parsed")
df <- df %>% select(-any_of(date_raw_cols))
cat("Dropped raw date columns:", paste(intersect(date_raw_cols, names(df)), collapse = ", "), "\n")

# 7.2 Label-encode ordinal: age_band
if ("age_band" %in% names(df)) {
  df <- df %>%
    mutate(age_band_encoded = as.integer(age_band))
  cat("age_band label-encoded → age_band_encoded\n")
}

# 7.3 One-hot encode low-cardinality categoricals
ohe_candidates <- df %>%
  select(where(is.factor)) %>%
  select(where(~ nlevels(.) <= 10)) %>%
  names()

cat("One-hot encoding:", paste(ohe_candidates, collapse = ", "), "\n")

df_encoded <- fastDummies::dummy_cols(
  df,
  select_columns  = ohe_candidates,
  remove_first_dummy      = TRUE,   # avoid perfect multicollinearity
  remove_selected_columns = TRUE    # drop original factor columns
)

# 7.4 Frequency-encode high-cardinality: cust_location
if ("cust_location" %in% names(df_encoded)) {
  freq_map <- df_encoded %>%
    count(cust_location) %>%
    mutate(cust_location_freq = n / nrow(df_encoded)) %>%
    select(-n)

  df_encoded <- df_encoded %>%
    left_join(freq_map, by = "cust_location") %>%
    select(-cust_location)
  cat("cust_location frequency-encoded\n")
}

# 7.5 Drop remaining character columns (IDs, unmapped text)
char_remaining <- df_encoded %>% select(where(is.character)) %>% names()
if (length(char_remaining) > 0) {
  cat("Dropping remaining character cols:", paste(char_remaining, collapse = ", "), "\n")
  df_encoded <- df_encoded %>% select(-all_of(char_remaining))
}

# 7.6 Final check — all columns numeric
non_numeric <- df_encoded %>%
  select(!where(is.numeric)) %>%
  names()

if (length(non_numeric) == 0) {
  cat("\n✅ All columns are numeric — dataset is ML-ready.\n")
} else {
  cat("\n⚠️  Non-numeric columns remaining:", paste(non_numeric, collapse = ", "), "\n")
}

cat("Final dimensions:", dim(df_encoded), "\n")

# ── EXPECTED OUTPUT ─────────────────────────────────────────
# • Gender, Geography → binary dummy columns
# • age_band → integer 1–6
# • cust_location → float frequency column
# • Zero non-numeric columns


# ============================================================
# STEP 8 — FINAL CLEAN DATASET OUTPUT
# ============================================================
cat("\n", strrep("=", 60), "\n")
cat("STEP 8 — SAVE CLEANED DATASET\n")
cat(strrep("=", 60), "\n")

# 8.1 Deduplicate column names (safety net)
df_final <- df_encoded %>%
  select(!where(anyNA)) %>%          # belt-and-suspenders NA check
  janitor::remove_constant() %>%     # drop zero-variance columns
  janitor::clean_names()

# 8.2 Final validation report
cat("\n--- Final Dataset Report ---\n")
cat("Rows           :", nrow(df_final), "\n")
cat("Columns        :", ncol(df_final), "\n")
cat("Missing values :", sum(is.na(df_final)), "\n")
cat("Duplicate rows :", sum(duplicated(df_final)), "\n")

# Variance check — low-variance features can hurt ML
near_zero <- nearZeroVar(df_final, saveMetrics = TRUE) %>%
  rownames_to_column("feature") %>%
  filter(nzv == TRUE)

if (nrow(near_zero) > 0) {
  cat("\n⚠️  Near-zero variance features (consider dropping):\n")
  print(near_zero$feature)
  df_final <- df_final %>%
    select(-all_of(near_zero$feature))
  cat("Removed", nrow(near_zero), "near-zero variance features.\n")
}

# 8.3 Correlation matrix (numeric check)
cat("\n--- Top 10 Highly Correlated Pairs (|r| > 0.85) ---\n")
cor_matrix  <- cor(df_final, use = "pairwise.complete.obs")
cor_pairs   <- as.data.frame(as.table(cor_matrix)) %>%
  filter(Var1 != Var2, abs(Freq) > 0.85) %>%
  distinct(Freq, .keep_all = TRUE) %>%
  arrange(desc(abs(Freq))) %>%
  head(10)
print(cor_pairs)

# 8.4 Save
output_path <- "banking_cleaned_ml_ready.csv"
write_csv(df_final, output_path)
cat("\n✅ Saved:", output_path, "\n")
cat("Ready for: Random Forest | XGBoost | K-Means | Collaborative Filtering\n")

# 8.5 Print column manifest
cat("\n--- Final Column Manifest ---\n")
tibble(
  column = names(df_final),
  type   = map_chr(df_final, class)
) %>%
  mutate(purpose = case_when(
    column %in% c("exited")                        ~ "TARGET: churn",
    str_detect(column, "risk|default|late")        ~ "TARGET/FEATURE: credit risk",
    str_detect(column, "segment|cluster")          ~ "TARGET: segmentation",
    str_detect(column, "ratio|rate|score|index")   ~ "ENGINEERED feature",
    str_detect(column, "_flag|_freq|_encoded")     ~ "DERIVED feature",
    str_detect(column, "geography_|gender_|age_")  ~ "ENCODED demographic",
    TRUE                                            ~ "RAW feature"
  )) %>%
  print(n = Inf)
