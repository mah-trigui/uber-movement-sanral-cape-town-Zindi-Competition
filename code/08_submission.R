# ==============================================================================
# 08_submission.R - Generate Competition Submission
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Generate submission file in required format
# ==============================================================================

source("00_config.R")
print_header("Step 8: Generate Submission")

# ==============================================================================
# 1. CREATE SUBMISSION ID
# ==============================================================================
cat("Loading submission ID functions...\n")

#' Create submission ID column
#' Format: YYYY-MM-DD HH:MM:SS_segment_id
create_submission_id <- function(test) {
    # Format datetime
    datetime_str <- format(test$Date, "%Y-%m-%d %H:%M:%S")

    # Create ID
    test$datetime_x_segment_id <- paste(datetime_str, test$segment_id, sep = "_")

    return(test)
}

#' Validate submission ID format
validate_submission_id <- function(submission) {
    # Check format
    sample_id <- submission$datetime_x_segment_id[1]

    # Expected pattern: YYYY-MM-DD HH:MM:SS_segmentid
    pattern <- "^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}_"

    if (!grepl(pattern, sample_id)) {
        warning("Submission ID format may be incorrect!")
        cat("  Sample ID:", sample_id, "\n")
    } else {
        cat("  Submission ID format validated\n")
    }

    return(invisible(NULL))
}

# ==============================================================================
# 2. GENERATE SUBMISSION
# ==============================================================================
cat("\nLoading submission generation functions...\n")

#' Generate submission file from predictions
generate_submission <- function(test, predictions, threshold = 0.5,
                                output_dir = "output", filename = NULL) {
    # Create submission ID if not exists
    if (!"datetime_x_segment_id" %in% names(test)) {
        test <- create_submission_id(test)
    }

    # Apply threshold
    target <- ifelse(predictions >= threshold, 1, 0)

    # Create submission dataframe
    submission <- data.frame(
        datetime_x_segment_id = test$datetime_x_segment_id,
        target = target
    )

    # Validate
    validate_submission_id(submission)

    # Summary
    cat("\n  Submission Summary:\n")
    cat("  Total rows:", nrow(submission), "\n")
    cat("  Predicted incidents (1):", sum(submission$target == 1), "\n")
    cat("  Predicted no incidents (0):", sum(submission$target == 0), "\n")
    cat("  Incident ratio:", round(mean(submission$target) * 100, 2), "%\n")

    # Create output directory if not exists
    if (!dir.exists(output_dir)) {
        dir.create(output_dir, recursive = TRUE)
    }

    # Generate filename
    if (is.null(filename)) {
        timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
        filename <- paste0("submission_", timestamp, ".csv")
    }

    filepath <- file.path(output_dir, filename)

    # Write CSV
    write.csv(submission, filepath, row.names = FALSE)
    cat("\n  Submission saved to:", filepath, "\n")

    return(submission)
}

#' Generate multiple submissions with different thresholds
generate_threshold_submissions <- function(test, predictions,
                                           thresholds = c(0.3, 0.4, 0.5, 0.6),
                                           output_dir = "output") {
    submissions <- list()

    for (thresh in thresholds) {
        cat("\n=== Generating submission with threshold:", thresh, "===\n")

        filename <- paste0("submission_thresh_", thresh, ".csv")
        sub <- generate_submission(test, predictions,
            threshold = thresh,
            output_dir = output_dir, filename = filename
        )

        submissions[[as.character(thresh)]] <- sub
    }

    return(submissions)
}

# ==============================================================================
# 3. ENSEMBLE SUBMISSION
# ==============================================================================
cat("\nLoading ensemble submission functions...\n")

#' Generate submission from ensemble of models
generate_ensemble_submission <- function(test, models, weights = NULL,
                                         threshold = 0.5, output_dir = "output") {
    # Get predictions from all models
    all_preds <- list()

    for (name in names(models)) {
        cat("  Getting predictions from:", name, "\n")

        model <- models[[name]]$model
        feature_names <- models[[name]]$feature_names

        # Prepare test data
        test_matrix <- prepare_test_for_xgboost(test, feature_names)

        # Predict
        probs <- predict(model, xgb.DMatrix(test_matrix))
        all_preds[[name]] <- probs
    }

    # Ensemble (weighted average)
    n_models <- length(all_preds)
    if (is.null(weights)) {
        weights <- rep(1 / n_models, n_models)
    }

    ensemble_probs <- rep(0, nrow(test))
    for (i in seq_along(all_preds)) {
        ensemble_probs <- ensemble_probs + weights[i] * all_preds[[i]]
    }

    cat("\n  Ensemble weights:", paste(names(all_preds), "=", round(weights, 2), collapse = ", "), "\n")

    # Generate submission
    filename <- paste0("submission_ensemble_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv")
    submission <- generate_submission(test, ensemble_probs,
        threshold = threshold,
        output_dir = output_dir, filename = filename
    )

    return(list(
        submission = submission,
        ensemble_probs = ensemble_probs,
        individual_probs = all_preds
    ))
}

# ==============================================================================
# 4. SUBMISSION ANALYSIS
# ==============================================================================
cat("\nLoading submission analysis functions...\n")

#' Analyze submission distribution
analyze_submission <- function(submission) {
    cat("\n=== Submission Analysis ===\n")

    # Parse datetime and segment
    parts <- strsplit(submission$datetime_x_segment_id, "_")

    submission$datetime <- sapply(parts, function(x) paste(x[1:2], collapse = " "))
    submission$segment_id <- sapply(parts, function(x) x[3])

    submission$datetime <- as.POSIXct(submission$datetime, format = "%Y-%m-%d %H:%M:%S")

    # By month
    submission$month <- format(submission$datetime, "%Y-%m")
    month_summary <- aggregate(target ~ month,
        data = submission,
        FUN = function(x) c(total = length(x), incidents = sum(x))
    )

    cat("\nIncidents by Month:\n")
    print(month_summary)

    # By hour
    submission$hour <- as.numeric(format(submission$datetime, "%H"))
    hour_summary <- aggregate(target ~ hour, data = submission, sum)

    cat("\nIncidents by Hour:\n")
    print(hour_summary)

    return(invisible(submission))
}

#' Compare two submissions
compare_submissions <- function(sub1, sub2, name1 = "Submission 1", name2 = "Submission 2") {
    cat("\n=== Comparing Submissions ===\n")

    # Merge on ID
    comparison <- merge(sub1, sub2,
        by = "datetime_x_segment_id",
        suffixes = c("_1", "_2")
    )

    # Agreement
    agreement <- mean(comparison$target_1 == comparison$target_2)
    cat("  Agreement:", round(agreement * 100, 2), "%\n")

    # Confusion matrix
    cat("\n  Cross-tabulation:\n")
    print(table(comparison$target_1, comparison$target_2,
        dnn = c(name1, name2)
    ))

    # Differences
    differences <- comparison[comparison$target_1 != comparison$target_2, ]
    cat("\n  Number of disagreements:", nrow(differences), "\n")

    return(invisible(comparison))
}

# ==============================================================================
# 5. LOAD SAMPLE SUBMISSION
# ==============================================================================
cat("\nLoading sample submission functions...\n")

#' Load and validate sample submission
load_sample_submission <- function(filepath) {
    sample_sub <- read.csv(filepath, stringsAsFactors = FALSE)

    cat("  Sample submission loaded\n")
    cat("  Rows:", nrow(sample_sub), "\n")
    cat("  Columns:", paste(names(sample_sub), collapse = ", "), "\n")

    return(sample_sub)
}

#' Match submission to sample format
match_to_sample <- function(submission, sample_sub) {
    # Ensure all IDs in sample are in submission
    missing <- setdiff(sample_sub$datetime_x_segment_id, submission$datetime_x_segment_id)
    extra <- setdiff(submission$datetime_x_segment_id, sample_sub$datetime_x_segment_id)

    if (length(missing) > 0) {
        cat("  Warning: Missing", length(missing), "IDs from sample\n")
    }

    if (length(extra) > 0) {
        cat("  Warning:", length(extra), "extra IDs not in sample\n")
    }

    # Reorder to match sample
    submission <- submission[match(
        sample_sub$datetime_x_segment_id,
        submission$datetime_x_segment_id
    ), ]

    # Fill missing with 0
    submission$target[is.na(submission$target)] <- 0

    cat("  Submission matched to sample format\n")

    return(submission)
}

cat("\nSubmission functions loaded!\n")
