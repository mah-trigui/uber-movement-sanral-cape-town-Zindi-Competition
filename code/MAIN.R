# ==============================================================================
# MAIN.R - Traffic Incident Prediction Pipeline
# ==============================================================================
# Project: Traffic Incident Prediction (Cape Town Road Network)
# Competition: Zindi Challenge
# Description: Main orchestration script to run the complete pipeline
# ==============================================================================

cat("\n")
cat("==============================================================\n")
cat("    TRAFFIC INCIDENT PREDICTION PIPELINE\n")
cat("    Cape Town Road Network\n")
cat("==============================================================\n\n")

# ==============================================================================
# 1. LOAD ALL MODULES
# ==============================================================================
cat("Loading modules...\n")

source("00_config.R")
source("01_data_loading.R")
source("02_data_cleaning.R")
source("03_feature_engineering.R")
source("04_build_train_test.R")
source("05_sampling.R")
source("06_model_xgboost.R")
source("07_evaluation.R")
source("08_submission.R")

cat("All modules loaded successfully!\n\n")

# ==============================================================================
# 2. LOAD RAW DATA
# ==============================================================================
print_header("Loading Raw Data")

df <- load_incidents(INCIDENTS_FILE)
rd <- load_road_segments(ROAD_SEGMENTS_FILE)
vds <- load_vds_data(VDS_DATA_DIR)
weather <- load_weather_data(WEATHER_DATA_DIR)

cat("\nRaw data loaded!\n")

# ==============================================================================
# 3. CLEAN DATA
# ==============================================================================
print_header("Cleaning Data")

df <- clean_incidents(df)
rd <- clean_roads(rd)
vds <- clean_vds(vds)
weather <- clean_weather(weather)

cat("\nData cleaned!\n")

# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================
print_header("Feature Engineering")

# Add features to road segments
rd <- add_incident_count_features(rd, df)
rd <- add_daily_frequency_features(rd, df)
rd <- add_hourly_frequency_features(rd, df)
rd <- add_day_preference_features(rd, df)
rd <- add_hour_preference_features(rd, df)

cat("\nFeatures engineered!\n")

# ==============================================================================
# 5. BUILD TRAINING DATA
# ==============================================================================
print_header("Building Training Data")

train <- build_train_dataset(rd, df, vds, weather, TRAIN_START, TRAIN_END)

cat("\nTraining data built!\n")
cat("  Rows:", nrow(train), "\n")
cat("  Columns:", ncol(train), "\n")
cat("  Incident ratio:", round(mean(as.numeric(train$target) - 1) * 100, 2), "%\n")

# ==============================================================================
# 6. BUILD TEST DATA
# ==============================================================================
print_header("Building Test Data")

test <- build_test_dataset(rd, vds, weather, TEST_START, TEST_END)

cat("\nTest data built!\n")
cat("  Rows:", nrow(test), "\n")
cat("  Columns:", ncol(test), "\n")

# ==============================================================================
# 7. SAMPLING (HANDLE IMBALANCED DATA)
# ==============================================================================
print_header("Sampling Training Data")

# Option 1: ROSE undersampling
train_rose_under <- apply_rose_under(train, target_col = "target", p = 0.3)

# Option 2: ROSE both (over + under)
train_rose_both <- apply_rose_both(train, target_col = "target", p = 0.5)

# Option 3: SMOTE
train_smote <- apply_smote(train, target_col = "target", perc_over = 200, perc_under = 100)

cat("\nSampling complete!\n")

# ==============================================================================
# 8. TRAIN MODELS
# ==============================================================================
print_header("Training XGBoost Models")

# Prepare sampled data for modeling
sampled_list <- list(
    rose_under = train_rose_under,
    rose_both = train_rose_both,
    smote = train_smote
)

# Train multiple models
models <- train_multiple_models(sampled_list, params = XGBOOST_PARAMS)

cat("\nModels trained!\n")

# ==============================================================================
# 9. EVALUATE MODELS
# ==============================================================================
print_header("Evaluating Models")

# Create validation set from original data
val_ratio <- 0.2
set.seed(GLOBAL_SEED)
val_idx <- sample(1:nrow(train), size = floor(nrow(train) * val_ratio))
train_subset <- train[-val_idx, ]
val_subset <- train[val_idx, ]

# Prepare validation data
val_labels <- as.integer(val_subset$target) - 1

# Compare models
comparison <- compare_models(models, val_subset, val_labels)

cat("\nEvaluation complete!\n")

# ==============================================================================
# 10. SELECT BEST MODEL
# ==============================================================================
print_header("Selecting Best Model")

# Select model with highest AUC
best_model_name <- comparison$model[1]
best_model <- models[[best_model_name]]

cat("Best model:", best_model_name, "\n")
cat("AUC:", comparison$auc[1], "\n")

# ==============================================================================
# 11. GENERATE PREDICTIONS
# ==============================================================================
print_header("Generating Predictions")

# Prepare test data
test_prepared <- prepare_test_for_xgboost(test, best_model$feature_names)

# Predict probabilities
test_probs <- predict(best_model$model, xgb.DMatrix(test_prepared))

cat("Predictions generated!\n")
cat("  Mean probability:", round(mean(test_probs), 4), "\n")
cat("  Min probability:", round(min(test_probs), 4), "\n")
cat("  Max probability:", round(max(test_probs), 4), "\n")

# ==============================================================================
# 12. GENERATE SUBMISSION
# ==============================================================================
print_header("Generating Submission")

# Use optimal threshold from evaluation
optimal_threshold <- comparison$best_threshold[1]
cat("Using threshold:", optimal_threshold, "\n")

# Generate submission
submission <- generate_submission(
    test = test,
    predictions = test_probs,
    threshold = optimal_threshold,
    output_dir = "output",
    filename = "submission_final.csv"
)

# ==============================================================================
# 13. SAVE MODEL
# ==============================================================================
print_header("Saving Model")

if (!dir.exists("output")) {
    dir.create("output", recursive = TRUE)
}

save_model_complete(
    model = best_model$model,
    feature_names = best_model$feature_names,
    filepath = "output/best_model.rds"
)

# ==============================================================================
# 14. SUMMARY
# ==============================================================================
cat("\n")
cat("==============================================================\n")
cat("                    PIPELINE COMPLETE!\n")
cat("==============================================================\n")
cat("\n")
cat("Summary:\n")
cat("  - Training period:", as.character(TRAIN_START), "to", as.character(TRAIN_END), "\n")
cat("  - Test period:", as.character(TEST_START), "to", as.character(TEST_END), "\n")
cat("  - Best model:", best_model_name, "\n")
cat("  - Best AUC:", round(comparison$auc[1], 4), "\n")
cat("  - Best F1:", round(comparison$best_f1[1], 4), "\n")
cat("  - Threshold:", optimal_threshold, "\n")
cat("  - Predicted incidents:", sum(submission$target == 1), "\n")
cat("\n")
cat("Output files:\n")
cat("  - Submission: output/submission_final.csv\n")
cat("  - Model: output/best_model.rds\n")
cat("\n")
cat("==============================================================\n")
