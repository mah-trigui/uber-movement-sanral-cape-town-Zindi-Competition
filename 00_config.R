# ==============================================================================
# 00_config.R - Configuration and Library Loading
# ==============================================================================
# Project: Traffic Incident Prediction (Cape Town Road Network)
# Description: Predict traffic incidents on road segments
# Target: Binary classification (incident = 1, no incident = 0)
# ==============================================================================

# ---- Global Options ----
options(scipen = 999)
options(sqldf.driver = "RSQLite")

# ---- Random Seed ----
GLOBAL_SEED <- 123
set.seed(GLOBAL_SEED)

# ---- Required Libraries ----

# Data manipulation
library(data.table)
library(dplyr)
library(tidyr)
library(sqldf)
library(Matrix)

# Machine Learning
library(xgboost)
library(caret)
library(caretEnsemble)
library(randomForest)

# Imbalanced Data Handling
library(splitstackshape)
library(ROSE)
library(DMwR)

# Missing Value Imputation
library(mice)

# Geospatial
library(sf)
library(rgdal)
library(foreign)
library(geosphere)
library(pracma)

# Visualization
library(ggplot2)
library(corrplot)
library(ggcorrplot)

# Utilities
library(chron)
library(Information)
library(lubridate)
library(parallel)

# H2O (optional - for AutoML)
# library(h2o)

# ---- Directory Configuration ----
DATA_DIR <- ""
OUTPUT_DIR <- "output/"

if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE)
}

# ---- Date Configuration ----
# Training period: September 2018 - December 2018
TRAIN_START <- as.POSIXct("2018-08-31 22:00:00", tz = "UTC")
TRAIN_END <- as.POSIXct("2018-12-31 22:00:00", tz = "UTC")

# Test period: January 2019 - March 2019
TEST_START <- as.POSIXct("2019-01-01 00:00:00", tz = "UTC")
TEST_END <- as.POSIXct("2019-03-31 21:00:00", tz = "UTC")

# ---- Special Days Configuration ----
# Heritage Day, Christmas Eve/Day, Dec 26, Boxing Day, New Year's Eve
SPECIAL_DAYS <- c(267, 350, 351, 358, 359, 360, 365)
# School holidays: Sep 29 - Oct 8
HOLIDAY_START <- 272
HOLIDAY_END <- 281

# ---- Model Parameters ----
TRAIN_RATIO <- 0.75
CV_FOLDS <- 10

# XGBoost default parameters for binary classification
XGBOOST_PARAMS <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.025,
    max_depth = 6,
    subsample = 0.85,
    colsample_bytree = 0.7,
    min_child_weight = 4,
    gamma = 10,
    scale_pos_weight = 2,
    nthread = 12
)

# ---- Weather Station Locations ----
WEATHER_STATIONS <- data.frame(
    Loc = c("Loc_25", "Loc_26", "Loc_27", "Loc_28", "Loc_29"),
    name = c("Town City", "Paarl", "Strand", "Airbase", "Airport"),
    long = c(18.412146, 18.969934, 18.821307, 18.48849, 18.592119),
    lat = c(-33.938629, -33.716755, -34.103854, -33.91129, -33.967621)
)

# ---- Helper Functions ----

#' Print section header
print_header <- function(text) {
    cat("\n", paste(rep("=", 60), collapse = ""), "\n")
    cat(" ", text, "\n")
    cat(paste(rep("=", 60), collapse = ""), "\n\n")
}

#' Calculate F1 Score
calc_f1 <- function(precision, recall) {
    (2 * precision * recall) / (precision + recall)
}

#' Create date sequence for cross join
create_date_sequence <- function(start, end) {
    dates <- seq(from = start, to = end, by = "hour")
    aux_date <- data.table(Date = dates)
    return(aux_date)
}

cat("Configuration loaded successfully!\n")
cat("Training period:", format(TRAIN_START), "to", format(TRAIN_END), "\n")
cat("Test period:", format(TEST_START), "to", format(TEST_END), "\n")
