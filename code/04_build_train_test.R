# ==============================================================================
# 04_build_train_test.R - Build Training and Test Datasets
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Create full train/test datasets with all features joined
# ==============================================================================

source("00_config.R")
print_header("Step 4: Build Train/Test Datasets")

# ==============================================================================
# 1. CREATE BASE TRAINING DATASET
# ==============================================================================
cat("Creating base training dataset...\n")

#' Create base training dataset (all segment x hour combinations)
create_train_base <- function(rd, train_start, train_end) {
  # Generate all hours in training period
  dates <- seq(from = train_start, to = train_end, by = "hour")
  aux_date <- data.table(Date = dates)

  # Get distinct segments
  segments <- rd[rd$segment_id != "0", c("segment_id")]

  # Cross join: all dates x all segments
  train <- as.data.table(sqldf("
    SELECT a.Date, b.segment_id
    FROM aux_date a
    CROSS JOIN segments b
  "))

  # Extract time components
  train$month <- month(train$Date)
  train$day <- mday(train$Date)
  train$day_year <- yday(train$Date)
  train$hour <- hour(train$Date)

  cat("  Base train rows:", nrow(train), "\n")
  return(train)
}

# ==============================================================================
# 2. CREATE BASE TEST DATASET
# ==============================================================================
cat("\nCreating base test dataset...\n")

#' Create base test dataset
create_test_base <- function(rd, test_start, test_end) {
  # Generate all hours in test period
  dates <- seq(from = test_start, to = test_end, by = "hour")
  aux_date <- data.table(Date = dates)

  # Get distinct segments (from submission file or road data)
  segments <- rd[rd$segment_id != "0", c("segment_id")]

  # Cross join
  test <- as.data.table(sqldf("
    SELECT a.Date, b.segment_id
    FROM aux_date a
    CROSS JOIN segments b
  "))

  # Extract time components
  test$month <- month(test$Date)
  test$day <- mday(test$Date)
  test$day_year <- yday(test$Date)
  test$hour <- hour(test$Date)

  # Handle special case: March 31, 2019 02:00 (DST issue)
  # Some rows may have missing hour 2, add them manually if needed

  cat("  Base test rows:", nrow(test), "\n")
  return(test)
}

# ==============================================================================
# 3. JOIN ROAD FEATURES
# ==============================================================================
cat("\nJoining road features...\n")

#' Join road segment features to train/test
join_road_features <- function(data, rd) {
  # Select relevant road features
  road_cols <- c(
    "segment_id", "ROADNO", "PAVETYPE", "CONDITION", "length_1",
    "road_width", "road_len", "nb_incd_cat", "last_inc",
    "decemb_variat", "weekend_pref",
    "freq_d_mean", "freq_d_max", "freq_d_min", "freq_d_med", "freq_d_sd",
    "freq_h_mean", "freq_h_max", "freq_h_min", "freq_h_med", "freq_h_sd",
    "day_pref_1", "day_pref_2", "prct_day_pref_1", "prct_day_pref_2",
    "hour_pref_1", "hour_pref_2", "hour_pref_3",
    "prct_hour_pref_1", "prct_hour_pref_2", "prct_hour_pref_3"
  )

  # Keep only columns that exist
  road_cols <- road_cols[road_cols %in% names(rd)]

  # Add Name (VDS station) if exists
  if ("Name" %in% names(rd)) {
    road_cols <- c(road_cols, "Name")
  }
  if ("Loc" %in% names(rd)) {
    road_cols <- c(road_cols, "Loc")
  }

  data <- merge(data, rd[, ..road_cols], by = "segment_id", all.x = TRUE)

  cat("  Joined road features:", length(road_cols) - 1, "columns\n")
  return(data)
}

# ==============================================================================
# 4. JOIN VDS FEATURES
# ==============================================================================
cat("\nJoining VDS features...\n")

#' Join VDS (traffic) features to train/test
join_vds_features <- function(data, vds, rd) {
  # Create reference table: segment_id -> VDS Name
  ref <- unique(rd[, c("segment_id", "Name")])

  # Add Name to data
  if (!"Name" %in% names(data)) {
    data <- merge(data, ref, by = "segment_id", all.x = TRUE)
  }

  # Join VDS features
  vds_cols <- c("Name", "hour", "day_year", "speed", "nb_veh_1", "nb_veh_2", "nb_veh_3")
  vds_cols <- vds_cols[vds_cols %in% names(vds)]

  data <- merge(data, vds[, ..vds_cols],
    by = c("Name", "hour", "day_year"), all.x = TRUE
  )

  # Fill NAs
  data$speed[is.na(data$speed)] <- 300
  data$nb_veh_1[is.na(data$nb_veh_1)] <- 0
  data$nb_veh_2[is.na(data$nb_veh_2)] <- 0
  data$nb_veh_3[is.na(data$nb_veh_3)] <- 0

  cat("  Joined VDS features\n")
  return(data)
}

# ==============================================================================
# 5. JOIN WEATHER FEATURES
# ==============================================================================
cat("\nJoining weather features...\n")

#' Join weather features to train/test
join_weather_features <- function(data, weather, rd) {
  # Create reference table: segment_id -> Weather Location
  ref <- unique(rd[, c("segment_id", "Loc")])

  # Add Loc to data
  if (!"Loc" %in% names(data)) {
    data <- merge(data, ref, by = "segment_id", all.x = TRUE)
  }

  # Join weather features
  weather_cols <- c(
    "Loc", "hour", "day_year", "temp", "press", "press_3h",
    "humid", "wind", "gust"
  )
  weather_cols <- weather_cols[weather_cols %in% names(weather)]

  data <- merge(data, weather[, ..weather_cols],
    by = c("Loc", "hour", "day_year"), all.x = TRUE
  )

  # Fill NAs with forward fill or median
  weather_num_cols <- c("temp", "press", "press_3h", "humid", "wind", "gust")
  for (col in weather_num_cols) {
    if (col %in% names(data)) {
      median_val <- median(data[[col]], na.rm = TRUE)
      data[[col]][is.na(data[[col]])] <- median_val
    }
  }

  cat("  Joined weather features\n")
  return(data)
}

# ==============================================================================
# 6. JOIN INCIDENT LABELS (TRAIN ONLY)
# ==============================================================================
cat("\nJoining incident labels...\n")

#' Join incident labels to training data
join_incident_labels <- function(train, df) {
  # Get unique incidents per segment, hour, day_year
  incidents <- df[, c("segment_id", "hour", "day_year", "incident", "target")]
  incidents <- unique(incidents)

  train <- merge(train, incidents,
    by = c("segment_id", "hour", "day_year"), all.x = TRUE
  )

  # Fill NAs (no incident)
  train$incident[is.na(train$incident)] <- 0
  train$target[is.na(train$target)] <- 0

  # Convert target to factor
  train$target <- as.factor(train$target)

  cat("  Incidents (class 1):", sum(train$target == 1), "\n")
  cat("  No incidents (class 0):", sum(train$target == 0), "\n")

  return(train)
}

# ==============================================================================
# 7. FINALIZE DATASET
# ==============================================================================
cat("\nFinalizing dataset...\n")

#' Clean and finalize dataset
finalize_dataset <- function(data) {
  # Remove duplicates
  data <- unique(data)

  # Convert factor columns
  factor_cols <- c(
    "PAVETYPE", "CONDITION", "road_width", "road_len",
    "nb_incd_cat", "special", "hour_weo"
  )
  for (col in factor_cols) {
    if (col %in% names(data)) {
      data[[col]] <- as.factor(data[[col]])
    }
  }

  cat("  Final dataset rows:", nrow(data), "\n")
  cat("  Final dataset cols:", ncol(data), "\n")

  return(data)
}

# ==============================================================================
# 8. MASTER BUILD FUNCTION
# ==============================================================================

#' Build complete training dataset
build_train_dataset <- function(rd, df, vds, weather, train_start, train_end) {
  cat("\n--- Building Training Dataset ---\n")

  # Create base
  train <- create_train_base(rd, train_start, train_end)

  # Join all features
  train <- join_road_features(train, rd)
  train <- join_vds_features(train, vds, rd)
  train <- join_weather_features(train, weather, rd)
  train <- join_incident_labels(train, df)

  # Add time features (from 03_feature_engineering.R)
  train <- add_time_features(train)
  train <- add_preference_diff_features(train)

  # Finalize
  train <- finalize_dataset(train)

  return(train)
}

#' Build complete test dataset
build_test_dataset <- function(rd, vds, weather, test_start, test_end) {
  cat("\n--- Building Test Dataset ---\n")

  # Create base
  test <- create_test_base(rd, test_start, test_end)

  # Join all features
  test <- join_road_features(test, rd)
  test <- join_vds_features(test, vds, rd)
  test <- join_weather_features(test, weather, rd)

  # Add time features
  test <- add_time_features(test)
  test <- add_preference_diff_features(test)

  # Finalize
  test <- finalize_dataset(test)

  return(test)
}

cat("\nTrain/Test build functions loaded!\n")
