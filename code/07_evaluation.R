# ==============================================================================
# 07_evaluation.R - Model Evaluation & Metrics
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Evaluation metrics, threshold optimization, and analysis
# ==============================================================================

source("00_config.R")
print_header("Step 7: Model Evaluation")

# ==============================================================================
# 1. BASIC METRICS
# ==============================================================================
cat("Loading basic metrics functions...\n")

#' Calculate AUC
calculate_auc <- function(actual, predicted) {
    roc_obj <- roc(actual, predicted, quiet = TRUE)
    auc_val <- auc(roc_obj)

    cat("  AUC:", round(auc_val, 4), "\n")
    return(as.numeric(auc_val))
}

#' Calculate confusion matrix metrics
calculate_confusion_metrics <- function(actual, predicted_class) {
    # Confusion matrix
    cm <- confusionMatrix(as.factor(predicted_class), as.factor(actual),
        positive = "1"
    )

    metrics <- list(
        accuracy = cm$overall["Accuracy"],
        precision = cm$byClass["Precision"],
        recall = cm$byClass["Recall"],
        f1 = cm$byClass["F1"],
        specificity = cm$byClass["Specificity"],
        sensitivity = cm$byClass["Sensitivity"],
        balanced_accuracy = cm$byClass["Balanced Accuracy"]
    )

    cat("  Accuracy:", round(metrics$accuracy, 4), "\n")
    cat("  Precision:", round(metrics$precision, 4), "\n")
    cat("  Recall:", round(metrics$recall, 4), "\n")
    cat("  F1 Score:", round(metrics$f1, 4), "\n")

    return(list(metrics = metrics, confusion_matrix = cm))
}

#' Print confusion matrix
print_confusion_matrix <- function(actual, predicted_class) {
    cat("\n  Confusion Matrix:\n")
    print(table(Predicted = predicted_class, Actual = actual))

    # Percentages
    cm_table <- table(Predicted = predicted_class, Actual = actual)
    cat("\n  Percentages:\n")
    print(round(prop.table(cm_table) * 100, 2))
}

# ==============================================================================
# 2. THRESHOLD OPTIMIZATION
# ==============================================================================
cat("\nLoading threshold optimization functions...\n")

#' Find optimal threshold for F1 score
find_optimal_threshold_f1 <- function(actual, predicted_probs, thresholds = seq(0.1, 0.9, 0.01)) {
    results <- data.frame(
        threshold = thresholds,
        f1 = NA,
        precision = NA,
        recall = NA
    )

    for (i in seq_along(thresholds)) {
        thresh <- thresholds[i]
        pred_class <- ifelse(predicted_probs >= thresh, 1, 0)

        # Calculate metrics
        tp <- sum(pred_class == 1 & actual == 1)
        fp <- sum(pred_class == 1 & actual == 0)
        fn <- sum(pred_class == 0 & actual == 1)

        precision <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        recall <- ifelse(tp + fn > 0, tp / (tp + fn), 0)
        f1 <- ifelse(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)

        results$f1[i] <- f1
        results$precision[i] <- precision
        results$recall[i] <- recall
    }

    # Find best threshold
    best_idx <- which.max(results$f1)
    best_threshold <- results$threshold[best_idx]

    cat("  Best threshold:", best_threshold, "\n")
    cat("  Best F1:", round(results$f1[best_idx], 4), "\n")
    cat("  Precision:", round(results$precision[best_idx], 4), "\n")
    cat("  Recall:", round(results$recall[best_idx], 4), "\n")

    return(list(
        best_threshold = best_threshold,
        results = results
    ))
}

#' Find optimal threshold using Youden's J statistic
find_optimal_threshold_youden <- function(actual, predicted_probs) {
    roc_obj <- roc(actual, predicted_probs, quiet = TRUE)

    # Youden's J = Sensitivity + Specificity - 1
    coords <- coords(roc_obj, "best", best.method = "youden")

    cat("  Optimal threshold (Youden):", round(coords$threshold, 4), "\n")
    cat("  Sensitivity:", round(coords$sensitivity, 4), "\n")
    cat("  Specificity:", round(coords$specificity, 4), "\n")

    return(coords$threshold)
}

# ==============================================================================
# 3. ROC CURVE
# ==============================================================================
cat("\nLoading ROC curve functions...\n")

#' Plot ROC curve
plot_roc_curve <- function(actual, predicted_probs, title = "ROC Curve") {
    roc_obj <- roc(actual, predicted_probs, quiet = TRUE)
    auc_val <- auc(roc_obj)

    plot(roc_obj,
        main = paste(title, "- AUC:", round(auc_val, 4)),
        col = "blue", lwd = 2
    )
    abline(a = 0, b = 1, lty = 2, col = "gray")

    return(roc_obj)
}

#' Plot multiple ROC curves (for comparing models)
plot_multiple_roc <- function(actual, predictions_list, model_names) {
    colors <- c("blue", "red", "green", "orange", "purple")

    # First plot
    roc_obj <- roc(actual, predictions_list[[1]], quiet = TRUE)
    auc_val <- auc(roc_obj)

    plot(roc_obj,
        main = "ROC Curve Comparison",
        col = colors[1], lwd = 2
    )
    legend_text <- paste(model_names[1], "- AUC:", round(auc_val, 4))

    # Additional plots
    for (i in 2:length(predictions_list)) {
        roc_obj <- roc(actual, predictions_list[[i]], quiet = TRUE)
        auc_val <- auc(roc_obj)

        lines(roc_obj, col = colors[i], lwd = 2)
        legend_text <- c(legend_text, paste(model_names[i], "- AUC:", round(auc_val, 4)))
    }

    abline(a = 0, b = 1, lty = 2, col = "gray")
    legend("bottomright", legend = legend_text, col = colors[1:length(predictions_list)], lwd = 2)
}

# ==============================================================================
# 4. MODEL COMPARISON
# ==============================================================================
cat("\nLoading model comparison functions...\n")

#' Compare multiple models on validation set
compare_models <- function(models, val_data, val_labels) {
    results <- data.frame(
        model = character(),
        auc = numeric(),
        best_f1 = numeric(),
        best_threshold = numeric(),
        stringsAsFactors = FALSE
    )

    for (name in names(models)) {
        cat("\n=== Evaluating:", name, "===\n")

        model <- models[[name]]$model
        feature_names <- models[[name]]$feature_names

        # Prepare test data
        test_matrix <- prepare_test_for_xgboost(val_data, feature_names)

        # Predict
        probs <- predict(model, xgb.DMatrix(test_matrix))

        # Calculate AUC
        auc_val <- calculate_auc(val_labels, probs)

        # Find optimal threshold
        thresh_result <- find_optimal_threshold_f1(val_labels, probs)

        results <- rbind(results, data.frame(
            model = name,
            auc = auc_val,
            best_f1 = max(thresh_result$results$f1),
            best_threshold = thresh_result$best_threshold
        ))
    }

    # Sort by AUC
    results <- results[order(-results$auc), ]

    cat("\n=== Model Comparison ===\n")
    print(results)

    return(results)
}

# ==============================================================================
# 5. VALIDATION ANALYSIS
# ==============================================================================
cat("\nLoading validation analysis functions...\n")

#' Analyze predictions by segment
analyze_predictions_by_segment <- function(data, actual, predicted_probs, threshold = 0.5) {
    pred_class <- ifelse(predicted_probs >= threshold, 1, 0)

    analysis <- data.frame(
        segment_id = data$segment_id,
        actual = actual,
        predicted = pred_class,
        prob = predicted_probs
    )

    # Group by segment
    segment_analysis <- analysis %>%
        group_by(segment_id) %>%
        summarise(
            total = n(),
            actual_incidents = sum(actual == 1),
            predicted_incidents = sum(predicted == 1),
            true_positives = sum(actual == 1 & predicted == 1),
            false_positives = sum(actual == 0 & predicted == 1),
            false_negatives = sum(actual == 1 & predicted == 0),
            avg_prob = mean(prob)
        ) %>%
        ungroup()

    return(segment_analysis)
}

#' Analyze predictions by time
analyze_predictions_by_time <- function(data, actual, predicted_probs, threshold = 0.5) {
    pred_class <- ifelse(predicted_probs >= threshold, 1, 0)

    analysis <- data.frame(
        hour = data$hour,
        day = if ("day" %in% names(data)) data$day else NA,
        actual = actual,
        predicted = pred_class,
        prob = predicted_probs
    )

    # Group by hour
    hour_analysis <- analysis %>%
        group_by(hour) %>%
        summarise(
            total = n(),
            actual_incidents = sum(actual == 1),
            predicted_incidents = sum(predicted == 1),
            precision = sum(actual == 1 & predicted == 1) / sum(predicted == 1),
            recall = sum(actual == 1 & predicted == 1) / sum(actual == 1)
        ) %>%
        ungroup()

    return(hour_analysis)
}

# ==============================================================================
# 6. FULL EVALUATION REPORT
# ==============================================================================
cat("\nLoading full evaluation report function...\n")

#' Generate complete evaluation report
generate_evaluation_report <- function(actual, predicted_probs, model_name = "XGBoost") {
    cat("\n")
    cat("==============================================================\n")
    cat("           EVALUATION REPORT -", model_name, "\n")
    cat("==============================================================\n")

    # AUC
    cat("\n1. AUC Score:\n")
    auc_val <- calculate_auc(actual, predicted_probs)

    # Optimal threshold
    cat("\n2. Threshold Optimization:\n")
    thresh_result <- find_optimal_threshold_f1(actual, predicted_probs)
    optimal_threshold <- thresh_result$best_threshold

    # Confusion matrix at optimal threshold
    cat("\n3. Confusion Matrix (threshold =", optimal_threshold, "):\n")
    pred_class <- ifelse(predicted_probs >= optimal_threshold, 1, 0)
    cm_result <- calculate_confusion_metrics(actual, pred_class)
    print_confusion_matrix(actual, pred_class)

    # Class distribution
    cat("\n4. Class Distribution:\n")
    cat("  Actual 1s:", sum(actual == 1), "\n")
    cat("  Actual 0s:", sum(actual == 0), "\n")
    cat("  Predicted 1s:", sum(pred_class == 1), "\n")
    cat("  Predicted 0s:", sum(pred_class == 0), "\n")

    cat("\n==============================================================\n")

    return(list(
        auc = auc_val,
        optimal_threshold = optimal_threshold,
        metrics = cm_result$metrics,
        threshold_results = thresh_result$results
    ))
}

cat("\nEvaluation functions loaded!\n")
