# Traffic Incident Prediction — Cape Town Road Network

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [Uber Movement SANRAL Cape Town Challenge 🌾 - Win $5 500 USD](https://zindi.africa/competitions/uber-movement-sanral-cape-town-challenge)

Ranked in the place 13 over 133
---

Binary classification: predict whether a traffic incident will occur on a given road segment during a given hour.
Zindi ML Competition — Training: Sep–Dec 2018 / Test: Jan–Mar 2019.

---

## Problem Framing

The dataset only contains rows where incidents occurred. The first challenge was converting this into a proper supervised learning problem by generating the full space of segment × hour combinations and labeling unobserved combinations as 0.

---

## Key Engineering Decisions

### 1. Incident frequency profiling per segment

Each road segment was profiled by the statistical distribution of time gaps between its historical incidents — both at daily and hourly granularity:

```r
# Inter-incident interval statistics per segment
freq_daily <- daily_filtered %>%
  group_by(segment_id) %>%
  summarise(
    freq_daily_mean = mean(Diff, na.rm = TRUE),
    freq_daily_median  = median(Diff, na.rm = TRUE),
    freq_daily_sd   = sd(Diff, na.rm = TRUE)
  )
```

A consecutive integer timeline (`hour_cons`) was used to correctly compute inter-incident intervals across the hourly axis.

### 2. Temporal preference profile per segment

Each segment's top-3 historically preferred hours and days were extracted, then at prediction time the pipeline computed how far the current moment deviated from those preferences:

```r
data$prct_hour_1 <- ifelse(
  abs(data$hour - data$hour_pref_1) == 0, data$prct_hour_pref_1,
  ifelse(abs(data$hour - data$hour_pref_2) == 0, data$prct_hour_pref_2,
    (1 - data$prct_hour_pref) / 21
  )
)

data$diff_hour <- ((data$prct_hour_pref_1 - data$prct_hour_1) +
                   (data$prct_hour_pref_2 - data$prct_hour_2)) / 2
```

### 3. Domain-specific temporal flags

- `decemb_variat`: 1 if a segment shows December-only or December-increasing incidents
- `weekend_pref`: 1 if weekend incidents are disproportionately high for that segment
- `special`: N/S/H flag encoding normal days, special events, and holiday periods

### 4. Multi-source data fusion

Road network attributes, VDS sensor readings (speed, vehicle counts by class), and weather station data were all joined to the base segment × hour grid.

### 5. Imbalanced classification handling

Three sampling strategies implemented and compared: ROSE undersampling, ROSE combined, and SMOTE oversampling.

---

## Project Structure

```
├── 00_config.R              # Libraries, paths, dates, model parameters
├── 01_data_loading.R        # Load incidents, road shapefile, VDS, weather
├── 02_data_cleaning.R       # Parse datetimes, consolidate causes, clean road data
├── 03_feature_engineering.R # Frequency profiles, preference features, time flags
├── 04_build_train_test.R    # Cross-join segment × hour grid, join all features
├── 05_sampling.R            # ROSE, SMOTE, stratified sampling
├── 06_model_xgboost.R       # XGBoost training, CV, hyperparameter tuning
├── 07_evaluation.R          # AUC, F1, confusion matrix
├── 08_submission.R          # Submission file generation
└── MAIN.R                   # Full pipeline orchestration
```

---

## Technical Stack

- **Language**: R
- **Core packages**: xgboost, caret, data.table, dplyr, lubridate
- **Geospatial**: sf, rgdal, geosphere
- **Imbalanced data**: ROSE, DMwR (SMOTE)
- **Feature selection**: Information (IV/WOE)

---

## How to Run

```r
source("MAIN.R")
```

Requires `train.csv`, road shapefile, `VDS_hourly/`, and `weather/` in the working directory.

---

## Scope

Competition data is not included. The repository shares the pipeline structure and feature engineering approach for a multi-source geospatial classification problem.
source("organized/06_model_xgboost.R")
prepared <- prepare_for_xgboost(train_balanced)
split <- create_train_val_split(prepared$data, prepared$label)
model <- train_xgboost(split$dtrain, split$dval)

# 8. Evaluate
source("organized/07_evaluation.R")
report <- generate_evaluation_report(val_labels, val_probs)

# 9. Generate submission
source("organized/08_submission.R")
submission <- generate_submission(test, test_probs, threshold = 0.5)
```

## Configuration

Key parameters are defined in `00_config.R`:

```r
# Dates
TRAIN_START <- as.POSIXct("2018-09-01 00:00:00")
TRAIN_END <- as.POSIXct("2018-12-31 23:00:00")
TEST_START <- as.POSIXct("2019-01-01 00:00:00")
TEST_END <- as.POSIXct("2019-03-31 23:00:00")

# Model parameters
XGBOOST_PARAMS <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.7,
  colsample_bytree = 0.7
)

# Random seed for reproducibility
GLOBAL_SEED <- 123
```

## Imbalanced Data Handling

Three strategies are implemented:

1. **ROSE Undersampling** - Undersample majority class using ROSE
2. **ROSE Both** - Combination of oversampling and undersampling
3. **SMOTE** - Synthetic Minority Over-sampling Technique

## Model

- **Algorithm**: XGBoost (gradient boosting)
- **Objective**: Binary logistic
- **Metric**: AUC (Area Under ROC Curve)
- **Early Stopping**: 50 rounds without improvement

## Evaluation Metrics

- AUC (primary metric)
- F1 Score
- Precision / Recall
- Confusion Matrix

## Output

The pipeline generates:

1. **Submission file**: `output/submission_final.csv`
   - Format: `datetime_x_segment_id,target`
   - Target: 0 or 1

2. **Model file**: `output/best_model.rds`
   - Saved XGBoost model with feature names

## Requirements

### R Packages

```r
# Data manipulation
library(data.table)
library(dplyr)
library(tidyr)
library(sqldf)

# Spatial data
library(sf)
library(foreign)
library(geojsonsf)

# Modeling
library(xgboost)
library(caret)
library(ROSE)
library(DMwR)

# Evaluation
library(pROC)

# Imputation
library(mice)

# Utilities
library(lubridate)
library(ggplot2)
```

## Notes

1. **VDS Data**: Vehicle Detection System data is aggregated by hour. Missing values are filled with defaults (speed=300, vehicle counts=0).

2. **Weather Data**: Weather data comes from 5 stations around Cape Town. Missing values are imputed using MICE or median.

3. **Special Days**: Includes holidays, month-end dates, and other special events that may affect traffic patterns.

4. **Threshold Tuning**: The optimal classification threshold is determined by maximizing F1 score on validation data.

## Author

Traffic Incident Prediction Project
Zindi Competition Entry
