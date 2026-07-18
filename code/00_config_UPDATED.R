# ==============================================================================
# 00_config.R - Configuration and Library Loading
# ==============================================================================
# Project: Traffic Incident Prediction (Cape Town Road Network - Zindi Challenge)
# Description: Predict traffic incidents on road segments using historical data
# Target: Binary classification (incident = 1, no incident = 0)
# Data Period: Train (Sept-Dec 2018), Test (Jan-Mar 2019)
# ==============================================================================

cat("\n")
cat("==============================================================\n")
cat("    TRAFFIC INCIDENT PREDICTION - CONFIGURATION\n")
cat("==============================================================\n\n")

# ---- Global Options ----
options(scipen = 999) # Disable scientific notation
options(sqldf.driver = "RSQLite") # SQLite driver for sqldf
options(stringsAsFactors = FALSE) # Don't auto-convert to factors

# ---- Random Seed ----
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)

cat("Setting random seed to", GLOBAL_SEED, "\n")

# ---- Required Libraries ----
cat("\nLoading required libraries...\n")

# Data manipulation
suppressPackageStartupMessages({
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(sqldf)
    library(Matrix)
})

# Machine Learning
suppressPackageStartupMessages({
    library(xgboost)
    library(caret)
    library(caretEnsemble)
    library(randomForest)
    library(pROC)
})

# Imbalanced Data Handling
suppressPackageStartupMessages({
    library(splitstackshape)
    library(ROSE)
    library(DMwR)
})

# Missing Value Imputation
suppressPackageStartupMessages({
    library(mice)
})

# Geospatial
suppressPackageStartupMessages({
    library(sf)
    library(rgdal)
    library(foreign)
    library(geosphere)
    library(pracma)
})

# Visualization
suppressPackageStartupMessages({
    library(ggplot2)
    library(corrplot)
    library(ggcorrplot)
})

# Utilities
suppressPackageStartupMessages({
    library(chron)
    library(Information)
    library(lubridate)
    library(parallel)
})

# Optional: H2O for AutoML
# library(h2o)

# Optional: Google Maps API for visualization
# library(googleway)

cat("All libraries loaded successfully!\n")

# ---- Directory Configuration ----
DATA_DIR <- ""
OUTPUT_DIR <- "output/"
MODEL_DIR <- "models/"
SUBMISSION_DIR <- "submissions/"

# Create directories if they don't exist
for (dir_path in c(OUTPUT_DIR, MODEL_DIR, SUBMISSION_DIR)) {
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
        cat("Created directory:", dir_path, "\n")
    }
}

# ---- Date Configuration ----
# Training period: September 2018 - December 2018 (4 months)
TRAIN_START <- as.POSIXct("2018-08-31 22:00:00", tz = "UTC")
TRAIN_END <- as.POSIXct("2018-12-31 22:00:00", tz = "UTC")

# Test period: January 2019 - March 2019 (3 months)
TEST_START <- as.POSIXct("2019-01-01 00:00:00", tz = "UTC")
TEST_END <- as.POSIXct("2019-03-31 21:00:00", tz = "UTC")

# Special datetime for DST (Daylight Saving Time) adjustment
# March 31, 2019 - hour 2 is missing due to DST
DST_SPECIAL_DATE <- as.POSIXct("2019-03-31 01:59:00", tz = "UTC")

cat("\nDate configuration:\n")
cat("  Training period:", format(TRAIN_START), "to", format(TRAIN_END), "\n")
cat("  Test period:", format(TEST_START), "to", format(TEST_END), "\n")

# ---- Special Days Configuration ----
# Heritage Day (Sep 24), Christmas Eve/Day, Boxing Day, New Year's Eve
# These are day_year values (day of year, 1-365)
SPECIAL_DAYS_2018 <- c(267, 350, 351, 358, 359, 360, 365) # 2018 days
SPECIAL_DAYS_2019 <- c(1, 80) # 2019: New Year, Mar 21 (Human Rights Day)

# School holidays in 2018: September 29 - October 8
HOLIDAY_START_2018 <- 272
HOLIDAY_END_2018 <- 281

# Holidays in 2019
HOLIDAY_START_2019_1 <- 2 # Jan 2-8 (New Year holiday)
HOLIDAY_END_2019_1 <- 8
HOLIDAY_START_2019_2 <- 75 # Easter holidays (around mid-March)
HOLIDAY_END_2019_2 <- 90

cat("  Special days configured for 2018 and 2019\n")

# ---- Model Parameters ----
TRAIN_RATIO <- 0.75 # Train/validation split ratio
CV_FOLDS <- 10 # Cross-validation folds
N_THREADS <- 10 # Parallel threads for XGBoost

# XGBoost default parameters for binary classification
# Based on experiments in W_2.R (best performing configuration)
XGBOOST_PARAMS <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.025, # Learning rate (from experiments)
    max_depth = 6, # Tree depth (from experiments)
    subsample = 0.85, # Row sampling (from experiments)
    colsample_bytree = 0.4, # Column sampling (from experiments)
    min_child_weight = 4, # Min weight in leaf (from experiments)
    gamma = 5, # Regularization (from experiments)
    scale_pos_weight = 1, # Class weight balance
    nthread = N_THREADS
)

# Alternative parameters for different sampling strategies
XGBOOST_PARAMS_UNDERSAMPLING <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.025,
    max_depth = 4,
    subsample = 0.85,
    colsample_bytree = 0.4,
    min_child_weight = 4,
    gamma = 10,
    scale_pos_weight = 1,
    nthread = N_THREADS
)

XGBOOST_PARAMS_SMOTE <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.01,
    max_depth = 8,
    subsample = 0.75,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    gamma = 12,
    scale_pos_weight = 3,
    nthread = N_THREADS
)

cat("\nModel parameters configured\n")

# ---- Weather Station Locations ----
# Geographic coordinates for weather stations in Cape Town region
WEATHER_STATIONS <- data.frame(
    Loc = c("Loc_25", "Loc_26", "Loc_27", "Loc_28", "Loc_29"),
    name = c("City Centre", "Paarl", "Strand", "Airbase", "Airport"),
    long = c(18.412146, 18.969934, 18.821307, 18.48849, 18.592119),
    lat = c(-33.938629, -33.716755, -34.103854, -33.91129, -33.967621),
    stringsAsFactors = FALSE
)

cat("  Weather stations:", nrow(WEATHER_STATIONS), "\n")

# ---- VDS (Vehicle Detection System) Configuration ----
# Default values for missing VDS data
VDS_DEFAULT_SPEED <- 300 # km/h (high value indicating no data)
VDS_DEFAULT_VEHICLES <- 0 # No vehicles counted

# ---- Feature Engineering Configuration ----
# Default values for segments without incident history
FREQ_DAY_MEAN_DEFAULT <- 500
FREQ_DAY_MAX_DEFAULT <- 1000
FREQ_DAY_MIN_DEFAULT <- 500
FREQ_DAY_MED_DEFAULT <- 500
FREQ_DAY_SD_DEFAULT <- 300

FREQ_HOUR_MEAN_DEFAULT <- 10000
FREQ_HOUR_MAX_DEFAULT <- 20000
FREQ_HOUR_MIN_DEFAULT <- 10000
FREQ_HOUR_MED_DEFAULT <- 10000
FREQ_HOUR_SD_DEFAULT <- 3000

LAST_INCIDENT_DEFAULT <- 500 # Days since last incident
DAY_PREF_DEFAULT <- 8 # Invalid day (no preference)
HOUR_PREF_DEFAULT <- 25 # Invalid hour (no preference)
PRCT_PREF_DEFAULT <- 1 # 100% (no data)

# ---- Threshold Configuration ----
# Optimal thresholds found through experimentation (from submission files)
# These are from various experiments in Under.R, Smote.R, auc-pr.R
THRESHOLD_CONSERVATIVE <- 0.775 # Higher precision, lower recall
THRESHOLD_BALANCED <- 0.77 # Balanced F1 score
THRESHOLD_AGGRESSIVE <- 0.5725 # Higher recall, lower precision
THRESHOLD_DEFAULT <- 0.775 # Best overall performance

cat("  Default prediction threshold:", THRESHOLD_DEFAULT, "\n")

# ---- Columns to Exclude from Modeling ----
# Non-predictive or identifier columns
EXCLUDE_COLS_MODELING <- c(
    "Date", "segment_id", "Name", "Loc", "EventID",
    "Date_Time", "Occurrence.Local.Date.Time", "incident",
    "REGION", "LANES", "WIDTH", "date_col", "date_new",
    "vehic_class", "Color", "VehicleType", "Cause", "Subcause"
)

# ---- Helper Functions ----

#' Print formatted section header
print_header <- function(text, char = "=", width = 60) {
    cat("\n", paste(rep(char, width), collapse = ""), "\n")
    cat(" ", text, "\n")
    cat(paste(rep(char, width), collapse = ""), "\n\n")
}

#' Print sub-section header
print_subheader <- function(text, char = "-", width = 60) {
    cat("\n", paste(rep(char, width), collapse = ""), "\n")
    cat(" ", text, "\n")
    cat(paste(rep(char, width), collapse = ""), "\n\n")
}

#' Calculate F1 Score
calc_f1 <- function(precision, recall) {
    if (precision + recall == 0) {
        return(0)
    }
    (2 * precision * recall) / (precision + recall)
}

#' Calculate precision
calc_precision <- function(tp, fp) {
    if (tp + fp == 0) {
        return(0)
    }
    tp / (tp + fp)
}

#' Calculate recall
calc_recall <- function(tp, fn) {
    if (tp + fn == 0) {
        return(0)
    }
    tp / (tp + fn)
}

#' Create date sequence for time-based features
create_date_sequence <- function(start, end, by = "hour") {
    dates <- seq(from = start, to = end, by = by)
    aux_date <- data.table(Date = dates)
    return(aux_date)
}

#' Safe division (returns 0 for division by zero)
safe_divide <- function(numerator, denominator, default = 0) {
    ifelse(denominator == 0, default, numerator / denominator)
}

#' Check if date is weekend
is_weekend_day <- function(date) {
    day_of_week <- wday(date)
    return(day_of_week %in% c(1, 7)) # Sunday = 1, Saturday = 7
}

#' Format timestamp for filenames
timestamp_for_filename <- function() {
    format(Sys.time(), "%Y%m%d_%H%M%S")
}

#' Memory usage report
print_memory_usage <- function() {
    cat("\nMemory Usage:\n")
    print(gc())
}

#' Timer helper
start_timer <- function() {
    Sys.time()
}

#' Timer helper - report elapsed time
report_timer <- function(start_time, message = "Elapsed time") {
    elapsed <- difftime(Sys.time(), start_time, units = "secs")
    cat(sprintf(
        "  %s: %.2f seconds (%.2f minutes)\n",
        message, as.numeric(elapsed), as.numeric(elapsed) / 60
    ))
}

# ---- Validation Helpers ----

#' Check if required columns exist in dataframe
check_required_columns <- function(df, required_cols, df_name = "dataframe") {
    missing_cols <- setdiff(required_cols, names(df))
    if (length(missing_cols) > 0) {
        warning(sprintf(
            "Missing columns in %s: %s",
            df_name, paste(missing_cols, collapse = ", ")
        ))
        return(FALSE)
    }
    return(TRUE)
}

#' Validate data ranges
validate_data_ranges <- function(df) {
    cat("\nValidating data ranges...\n")

    # Check for NAs
    na_counts <- sapply(df, function(x) sum(is.na(x)))
    if (any(na_counts > 0)) {
        cat("  Columns with NAs:\n")
        print(na_counts[na_counts > 0])
    }

    # Check numeric ranges
    numeric_cols <- sapply(df, is.numeric)
    if (any(numeric_cols)) {
        cat("  Numeric column summaries:\n")
        print(summary(df[, numeric_cols]))
    }
}

# ---- Configuration Summary ----
cat("\n")
cat("==============================================================\n")
cat("             CONFIGURATION SUMMARY\n")
cat("==============================================================\n")
cat("Training Period:", format(TRAIN_START), "to", format(TRAIN_END), "\n")
cat("Test Period:", format(TEST_START), "to", format(TEST_END), "\n")
cat("Random Seed:", GLOBAL_SEED, "\n")
cat("CV Folds:", CV_FOLDS, "\n")
cat("Prediction Threshold:", THRESHOLD_DEFAULT, "\n")
cat("XGBoost eta:", XGBOOST_PARAMS$eta, "\n")
cat("XGBoost max_depth:", XGBOOST_PARAMS$max_depth, "\n")
cat("Output Directory:", OUTPUT_DIR, "\n")
cat("==============================================================\n\n")

cat("Configuration loaded successfully!\n")
