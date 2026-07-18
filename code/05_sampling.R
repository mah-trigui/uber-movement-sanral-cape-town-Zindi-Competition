# ==============================================================================
# 05_sampling.R - Imbalanced Data Handling
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Stratified sampling, ROSE, SMOTE for imbalanced classification
# ==============================================================================

source("00_config.R")
print_header("Step 5: Sampling & Balancing")

# ==============================================================================
# 1. STRATIFIED SAMPLING
# ==============================================================================
cat("Loading stratified sampling functions...\n")

#' Perform stratified sampling by segment and hour
stratified_sample <- function(train, n_samples = 50000) {
    # Create stratification key
    train$strat_key <- paste(train$segment_id, train$hour, sep = "_")

    # Get distribution of incidents per stratum
    strat_summary <- train %>%
        group_by(strat_key) %>%
        summarise(
            total = n(),
            incidents = sum(target == 1),
            no_incidents = sum(target == 0),
            prop_incident = mean(target == 1)
        ) %>%
        ungroup()

    cat("  Number of strata:", nrow(strat_summary), "\n")

    # Calculate samples per stratum (proportional)
    total_rows <- nrow(train)
    strat_summary$n_sample <- round(strat_summary$total / total_rows * n_samples)

    # Ensure minimum of 1 sample per stratum
    strat_summary$n_sample[strat_summary$n_sample == 0] <- 1

    # Sample from each stratum
    set.seed(GLOBAL_SEED)

    sampled <- train %>%
        group_by(strat_key) %>%
        slice_sample(
            n = min(n(), strat_summary$n_sample[match(first(strat_key), strat_summary$strat_key)]),
            replace = FALSE
        ) %>%
        ungroup()

    # Remove stratification key
    sampled$strat_key <- NULL

    cat("  Sampled rows:", nrow(sampled), "\n")
    cat("  Incidents:", sum(sampled$target == 1), "\n")
    cat("  No incidents:", sum(sampled$target == 0), "\n")

    return(sampled)
}

#' Alternative: Sample keeping all incidents
stratified_sample_keep_incidents <- function(train, n_no_incident = 20000) {
    set.seed(GLOBAL_SEED)

    # Keep all incidents
    incidents <- train[train$target == 1, ]

    # Sample from no-incident class
    no_incidents <- train[train$target == 0, ]
    no_incidents <- no_incidents[sample(
        1:nrow(no_incidents),
        min(n_no_incident, nrow(no_incidents))
    ), ]

    sampled <- rbind(incidents, no_incidents)

    cat("  Incidents (all kept):", nrow(incidents), "\n")
    cat("  No incidents (sampled):", nrow(no_incidents), "\n")

    return(sampled)
}

# ==============================================================================
# 2. ROSE UNDERSAMPLING
# ==============================================================================
cat("\nLoading ROSE functions...\n")

#' Apply ROSE undersampling
apply_rose_under <- function(train, target_col = "target", p = 0.3) {
    formula <- as.formula(paste(target_col, "~ ."))

    # Remove non-predictive columns
    train_rose <- train[, !names(train) %in% c("Date", "segment_id", "Name", "Loc")]

    # Apply ROSE
    set.seed(GLOBAL_SEED)
    rose_result <- ROSE(formula,
        data = train_rose,
        p = p, seed = GLOBAL_SEED,
        hmult.majo = 0.25, hmult.mino = 0.5
    )

    rose_data <- rose_result$data

    cat("  ROSE Under - Class distribution:\n")
    print(table(rose_data[[target_col]]))

    return(rose_data)
}

#' Apply ROSE with both over and under sampling
apply_rose_both <- function(train, target_col = "target", n_target = NULL, p = 0.5) {
    formula <- as.formula(paste(target_col, "~ ."))

    # Remove non-predictive columns
    train_rose <- train[, !names(train) %in% c("Date", "segment_id", "Name", "Loc")]

    # Determine target N (default: 2x incident count)
    if (is.null(n_target)) {
        n_target <- sum(train_rose[[target_col]] == 1) * 2
    }

    # Apply ROSE with both
    set.seed(GLOBAL_SEED)
    rose_result <- ovun.sample(formula,
        data = train_rose,
        method = "both", N = n_target, p = p,
        seed = GLOBAL_SEED
    )

    rose_data <- rose_result$data

    cat("  ROSE Both - Class distribution:\n")
    print(table(rose_data[[target_col]]))

    return(rose_data)
}

# ==============================================================================
# 3. SMOTE OVERSAMPLING
# ==============================================================================
cat("\nLoading SMOTE functions...\n")

#' Apply SMOTE oversampling
apply_smote <- function(train, target_col = "target", perc_over = 200, perc_under = 100) {
    # Remove non-predictive columns and prepare data
    train_smote <- train[, !names(train) %in% c("Date", "segment_id", "Name", "Loc")]

    # Ensure target is factor
    train_smote[[target_col]] <- as.factor(train_smote[[target_col]])

    # Remove columns with all NA
    na_cols <- sapply(train_smote, function(x) all(is.na(x)))
    train_smote <- train_smote[, !na_cols]

    # Fill remaining NAs
    for (col in names(train_smote)) {
        if (is.numeric(train_smote[[col]])) {
            train_smote[[col]][is.na(train_smote[[col]])] <- median(train_smote[[col]], na.rm = TRUE)
        }
    }

    # Apply SMOTE
    set.seed(GLOBAL_SEED)
    smote_data <- SMOTE(as.formula(paste(target_col, "~ .")),
        data = train_smote,
        perc.over = perc_over,
        perc.under = perc_under
    )

    cat("  SMOTE - Class distribution:\n")
    print(table(smote_data[[target_col]]))

    return(smote_data)
}

# ==============================================================================
# 4. COMBINED SAMPLING STRATEGIES
# ==============================================================================
cat("\nLoading combined sampling functions...\n")

#' Apply multiple sampling strategies and return list
apply_all_sampling <- function(train, target_col = "target") {
    results <- list()

    # 1. ROSE Under
    cat("\n--- Applying ROSE Undersampling ---\n")
    results$rose_under <- apply_rose_under(train, target_col, p = 0.3)

    # 2. ROSE Both
    cat("\n--- Applying ROSE Both ---\n")
    results$rose_both <- apply_rose_both(train, target_col, p = 0.5)

    # 3. SMOTE
    cat("\n--- Applying SMOTE ---\n")
    results$smote <- apply_smote(train, target_col, perc_over = 200, perc_under = 100)

    return(results)
}

# ==============================================================================
# 5. FEATURE SELECTION HELPERS
# ==============================================================================
cat("\nLoading feature selection helpers...\n")

#' Remove highly correlated features
remove_correlated_features <- function(train, threshold = 0.95) {
    # Select numeric columns only
    num_cols <- sapply(train, is.numeric)
    num_data <- train[, num_cols, drop = FALSE]

    # Remove columns with zero variance
    zero_var <- sapply(num_data, function(x) var(x, na.rm = TRUE) == 0)
    num_data <- num_data[, !zero_var, drop = FALSE]

    # Compute correlation matrix
    cor_matrix <- cor(num_data, use = "pairwise.complete.obs")

    # Find highly correlated pairs
    high_cor <- findCorrelation(cor_matrix, cutoff = threshold)

    if (length(high_cor) > 0) {
        removed_cols <- names(num_data)[high_cor]
        cat("  Removing", length(removed_cols), "highly correlated features:\n")
        cat("   ", paste(removed_cols, collapse = ", "), "\n")

        train <- train[, !names(train) %in% removed_cols]
    } else {
        cat("  No highly correlated features found\n")
    }

    return(train)
}

#' Remove near-zero variance features
remove_nzv_features <- function(train) {
    nzv <- nearZeroVar(train, saveMetrics = TRUE)
    nzv_cols <- rownames(nzv)[nzv$nzv]

    # Don't remove target
    nzv_cols <- nzv_cols[nzv_cols != "target"]

    if (length(nzv_cols) > 0) {
        cat("  Removing", length(nzv_cols), "near-zero variance features:\n")
        cat(
            "   ", paste(head(nzv_cols, 10), collapse = ", "),
            ifelse(length(nzv_cols) > 10, "...", ""), "\n"
        )

        train <- train[, !names(train) %in% nzv_cols]
    } else {
        cat("  No near-zero variance features found\n")
    }

    return(train)
}

# ==============================================================================
# 6. DATA PREPARATION FOR MODELING
# ==============================================================================
cat("\nLoading data preparation functions...\n")

#' Prepare data for XGBoost (one-hot encoding, etc.)
prepare_for_xgboost <- function(train, target_col = "target") {
    # Separate target
    target <- as.integer(train[[target_col]]) - 1 # 0/1 encoding

    # Remove non-predictive columns
    exclude_cols <- c(target_col, "Date", "segment_id", "Name", "Loc", "strat_key")
    train <- train[, !names(train) %in% exclude_cols]

    # Convert factors to numeric (one-hot or label encoding)
    factor_cols <- sapply(train, is.factor)

    for (col in names(train)[factor_cols]) {
        train[[col]] <- as.integer(train[[col]])
    }

    # Convert to matrix
    train_matrix <- as.matrix(train)

    return(list(
        data = train_matrix,
        label = target,
        feature_names = names(train)
    ))
}

#' Prepare test data (same transformations as train)
prepare_test_for_xgboost <- function(test, train_features) {
    # Remove non-predictive columns
    exclude_cols <- c("Date", "segment_id", "Name", "Loc", "strat_key")
    test <- test[, !names(test) %in% exclude_cols]

    # Keep only features that exist in training
    common_features <- intersect(names(test), train_features)
    missing_features <- setdiff(train_features, names(test))

    if (length(missing_features) > 0) {
        cat("  Warning: Missing features in test:", length(missing_features), "\n")
        # Add missing features as 0
        for (feat in missing_features) {
            test[[feat]] <- 0
        }
    }

    # Reorder to match training
    test <- test[, train_features]

    # Convert factors
    factor_cols <- sapply(test, is.factor)
    for (col in names(test)[factor_cols]) {
        test[[col]] <- as.integer(test[[col]])
    }

    # Convert to matrix
    test_matrix <- as.matrix(test)

    return(test_matrix)
}

cat("\nSampling functions loaded!\n")
