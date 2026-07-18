# ==============================================================================
# 06_model_xgboost.R - XGBoost Model Training
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: XGBoost model training, cross-validation, and prediction
# ==============================================================================

source("00_config.R")
print_header("Step 6: XGBoost Model")

# ==============================================================================
# 1. CREATE DMATRIX
# ==============================================================================
cat("Loading DMatrix creation functions...\n")

#' Create XGBoost DMatrix from prepared data
create_dmatrix <- function(data, label = NULL) {
    if (!is.null(label)) {
        dmatrix <- xgb.DMatrix(data = data, label = label)
    } else {
        dmatrix <- xgb.DMatrix(data = data)
    }

    return(dmatrix)
}

#' Create train/validation split
create_train_val_split <- function(data, label, val_ratio = 0.2) {
    set.seed(GLOBAL_SEED)

    n <- nrow(data)
    val_idx <- sample(1:n, size = floor(n * val_ratio))
    train_idx <- setdiff(1:n, val_idx)

    dtrain <- xgb.DMatrix(data = data[train_idx, ], label = label[train_idx])
    dval <- xgb.DMatrix(data = data[val_idx, ], label = label[val_idx])

    cat("  Train samples:", length(train_idx), "\n")
    cat("  Validation samples:", length(val_idx), "\n")

    return(list(
        dtrain = dtrain,
        dval = dval,
        train_idx = train_idx,
        val_idx = val_idx
    ))
}

# ==============================================================================
# 2. MODEL TRAINING
# ==============================================================================
cat("\nLoading model training functions...\n")

#' Train XGBoost model with early stopping
train_xgboost <- function(dtrain, dval = NULL, params = NULL,
                          nrounds = 1000, early_stopping = 50) {
    # Default parameters
    if (is.null(params)) {
        params <- XGBOOST_PARAMS
    }

    # Watchlist
    if (!is.null(dval)) {
        watchlist <- list(train = dtrain, val = dval)
    } else {
        watchlist <- list(train = dtrain)
    }

    # Train model
    set.seed(GLOBAL_SEED)
    model <- xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        watchlist = watchlist,
        early_stopping_rounds = early_stopping,
        print_every_n = 100,
        verbose = 1
    )

    cat("\n  Best iteration:", model$best_iteration, "\n")
    cat("  Best AUC:", model$best_score, "\n")

    return(model)
}

#' Train multiple models with different sampling strategies
train_multiple_models <- function(sampled_data_list, params = NULL) {
    models <- list()

    for (name in names(sampled_data_list)) {
        cat("\n=== Training model:", name, "===\n")

        data <- sampled_data_list[[name]]

        # Prepare data
        prepared <- prepare_for_xgboost(data, target_col = "target")

        # Create train/val split
        split <- create_train_val_split(prepared$data, prepared$label, val_ratio = 0.2)

        # Train model
        model <- train_xgboost(split$dtrain, split$dval, params)

        models[[name]] <- list(
            model = model,
            feature_names = prepared$feature_names
        )
    }

    return(models)
}

# ==============================================================================
# 3. CROSS-VALIDATION
# ==============================================================================
cat("\nLoading cross-validation functions...\n")

#' Perform k-fold cross-validation
xgboost_cv <- function(data, label, params = NULL, nrounds = 1000,
                       nfold = 5, early_stopping = 50) {
    # Default parameters
    if (is.null(params)) {
        params <- XGBOOST_PARAMS
    }

    # Create DMatrix
    dtrain <- xgb.DMatrix(data = data, label = label)

    # Cross-validation
    set.seed(GLOBAL_SEED)
    cv_result <- xgb.cv(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        nfold = nfold,
        stratified = TRUE,
        early_stopping_rounds = early_stopping,
        print_every_n = 100,
        verbose = 1
    )

    # Best result
    best_iter <- which.max(cv_result$evaluation_log$test_auc_mean)
    best_auc <- cv_result$evaluation_log$test_auc_mean[best_iter]
    best_auc_std <- cv_result$evaluation_log$test_auc_std[best_iter]

    cat("\n  Best CV iteration:", best_iter, "\n")
    cat("  Best CV AUC:", round(best_auc, 4), "+/-", round(best_auc_std, 4), "\n")

    return(cv_result)
}

# ==============================================================================
# 4. HYPERPARAMETER TUNING
# ==============================================================================
cat("\nLoading hyperparameter tuning functions...\n")

#' Grid search for XGBoost hyperparameters using caret
tune_xgboost_caret <- function(train, target_col = "target") {
    # Prepare data
    train_caret <- train[, !names(train) %in% c("Date", "segment_id", "Name", "Loc")]
    train_caret[[target_col]] <- as.factor(ifelse(train_caret[[target_col]] == 1, "Yes", "No"))

    # Define grid
    tune_grid <- expand.grid(
        nrounds = c(100, 200, 500),
        max_depth = c(4, 6, 8),
        eta = c(0.01, 0.05, 0.1),
        gamma = 0,
        colsample_bytree = c(0.6, 0.8),
        min_child_weight = c(1, 3),
        subsample = c(0.7, 0.8)
    )

    # Train control
    train_control <- trainControl(
        method = "cv",
        number = 5,
        verboseIter = TRUE,
        classProbs = TRUE,
        summaryFunction = twoClassSummary,
        allowParallel = TRUE
    )

    # Train
    set.seed(GLOBAL_SEED)
    xgb_tune <- train(
        as.formula(paste(target_col, "~ .")),
        data = train_caret,
        method = "xgbTree",
        trControl = train_control,
        tuneGrid = tune_grid,
        metric = "ROC",
        verbose = FALSE
    )

    cat("\n  Best tuned parameters:\n")
    print(xgb_tune$bestTune)

    return(xgb_tune)
}

# ==============================================================================
# 5. PREDICTION
# ==============================================================================
cat("\nLoading prediction functions...\n")

#' Predict probabilities on test data
predict_proba <- function(model, test_matrix) {
    dtest <- xgb.DMatrix(data = test_matrix)
    probs <- predict(model, dtest)

    return(probs)
}

#' Predict with threshold
predict_class <- function(model, test_matrix, threshold = 0.5) {
    probs <- predict_proba(model, test_matrix)
    classes <- ifelse(probs >= threshold, 1, 0)

    return(list(probs = probs, classes = classes))
}

#' Ensemble predictions from multiple models
ensemble_predict <- function(models, test_matrix, weights = NULL) {
    n_models <- length(models)

    if (is.null(weights)) {
        weights <- rep(1 / n_models, n_models)
    }

    # Collect predictions
    all_probs <- matrix(0, nrow = nrow(test_matrix), ncol = n_models)

    for (i in seq_along(models)) {
        model <- models[[i]]$model
        all_probs[, i] <- predict_proba(model, test_matrix)
    }

    # Weighted average
    ensemble_probs <- rowSums(all_probs * matrix(weights,
        nrow = nrow(all_probs),
        ncol = n_models, byrow = TRUE
    ))

    return(ensemble_probs)
}

# ==============================================================================
# 6. FEATURE IMPORTANCE
# ==============================================================================
cat("\nLoading feature importance functions...\n")

#' Get feature importance from model
get_feature_importance <- function(model, top_n = 20) {
    importance <- xgb.importance(model = model)

    cat("\n  Top", top_n, "important features:\n")
    print(head(importance, top_n))

    return(importance)
}

#' Plot feature importance
plot_feature_importance <- function(model, top_n = 20) {
    importance <- xgb.importance(model = model)

    xgb.plot.importance(
        importance_matrix = importance, top_n = top_n,
        main = "XGBoost Feature Importance"
    )
}

# ==============================================================================
# 7. SAVE/LOAD MODEL
# ==============================================================================
cat("\nLoading model save/load functions...\n")

#' Save model to file
save_xgboost_model <- function(model, filepath) {
    xgb.save(model, filepath)
    cat("  Model saved to:", filepath, "\n")
}

#' Load model from file
load_xgboost_model <- function(filepath) {
    model <- xgb.load(filepath)
    cat("  Model loaded from:", filepath, "\n")
    return(model)
}

#' Save complete model with metadata
save_model_complete <- function(model, feature_names, filepath) {
    model_data <- list(
        model = model,
        feature_names = feature_names,
        params = XGBOOST_PARAMS,
        timestamp = Sys.time()
    )

    saveRDS(model_data, filepath)
    cat("  Complete model saved to:", filepath, "\n")
}

#' Load complete model with metadata
load_model_complete <- function(filepath) {
    model_data <- readRDS(filepath)
    cat("  Complete model loaded from:", filepath, "\n")
    cat("  Trained on:", as.character(model_data$timestamp), "\n")

    return(model_data)
}

cat("\nXGBoost model functions loaded!\n")
