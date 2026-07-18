# ==============================================================================
# 03_feature_engineering.R - Feature Engineering
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Create features from road, VDS, weather, and time data
# ==============================================================================

source("00_config.R")
print_header("Step 3: Feature Engineering")

# ==============================================================================
# 1. ROAD SEGMENT FEATURES
# ==============================================================================
cat("Creating road segment features...\n")

#' Create incident count features for road segments
create_incident_features <- function(rd, df) {
    # Count incidents per segment
    incident_counts <- as.data.table(sqldf("
    SELECT segment_id, COUNT(*) as nb_incd
    FROM df
    GROUP BY segment_id
  "))

    rd <- merge(rd, incident_counts, by = "segment_id", all.x = TRUE)
    rd$nb_incd[is.na(rd$nb_incd)] <- 0

    # Create incident count categories
    rd$nb_incd_cat <- cut(rd$nb_incd,
        breaks = c(-1, 3, 10, 30, 70, 120, Inf),
        labels = c("I_3", "I_10", "I_30", "I_70", "I_120", "I_250")
    )

    cat("  Added incident count features\n")
    return(rd)
}

#' Calculate days since last incident
create_last_incident_feature <- function(rd, df) {
    aux_last <- as.data.table(sqldf("
    SELECT DISTINCT segment_id, 365 - MAX(day_year) as last_inc
    FROM df
    GROUP BY segment_id
  "))

    rd <- merge(rd, aux_last, by = "segment_id", all.x = TRUE)
    rd$last_inc[is.na(rd$last_inc)] <- 500 # High value for segments without incidents

    cat("  Added last incident feature\n")
    return(rd)
}

#' Calculate December variation indicator
create_december_variation <- function(rd, df) {
    # Count incidents per month per segment
    monthly_counts <- as.data.table(sqldf("
    SELECT DISTINCT a.segment_id,
           COALESCE(b.nb_09, 0) as nb_09,
           COALESCE(c.nb_10, 0) as nb_10,
           COALESCE(d.nb_11, 0) as nb_11,
           COALESCE(e.nb_12, 0) as nb_12
    FROM (SELECT DISTINCT segment_id FROM df) a
    LEFT JOIN (SELECT segment_id, COUNT(*) as nb_09 FROM df WHERE month = 9 GROUP BY segment_id) b ON a.segment_id = b.segment_id
    LEFT JOIN (SELECT segment_id, COUNT(*) as nb_10 FROM df WHERE month = 10 GROUP BY segment_id) c ON a.segment_id = c.segment_id
    LEFT JOIN (SELECT segment_id, COUNT(*) as nb_11 FROM df WHERE month = 11 GROUP BY segment_id) d ON a.segment_id = d.segment_id
    LEFT JOIN (SELECT segment_id, COUNT(*) as nb_12 FROM df WHERE month = 12 GROUP BY segment_id) e ON a.segment_id = e.segment_id
  "))

    # December variation: 1 if only December OR increasing in December
    monthly_counts$decemb_variat <- ifelse(
        (monthly_counts$nb_09 == 0 & monthly_counts$nb_10 == 0 & monthly_counts$nb_11 == 0) |
            (monthly_counts$nb_12 >= monthly_counts$nb_11 & monthly_counts$nb_12 > 1),
        1, 0
    )

    rd <- merge(rd, monthly_counts[, c("segment_id", "decemb_variat")],
        by = "segment_id", all.x = TRUE
    )
    rd$decemb_variat[is.na(rd$decemb_variat)] <- 0

    cat("  Added December variation feature\n")
    return(rd)
}

#' Calculate weekend preference
create_weekend_preference <- function(rd, df) {
    weekend_counts <- as.data.table(sqldf("
    SELECT DISTINCT a.segment_id,
           COALESCE(b.week_1, 0) as week_1,
           COALESCE(c.week_0, 0) as week_0
    FROM (SELECT DISTINCT segment_id FROM df) a
    LEFT JOIN (SELECT segment_id, COUNT(*) as week_1 FROM df WHERE weekend = 1 GROUP BY segment_id) b ON a.segment_id = b.segment_id
    LEFT JOIN (SELECT segment_id, COUNT(*) as week_0 FROM df WHERE weekend = 0 GROUP BY segment_id) c ON a.segment_id = c.segment_id
  "))

    # Weekend preference: 1 if weekend incidents >= 2/5 of weekday
    weekend_counts$weekend_pref <- ifelse(weekend_counts$week_0 >= 2 * weekend_counts$week_1 / 5, 1, 0)

    rd <- merge(rd, weekend_counts[, c("segment_id", "weekend_pref")],
        by = "segment_id", all.x = TRUE
    )
    rd$weekend_pref[is.na(rd$weekend_pref)] <- 0

    cat("  Added weekend preference feature\n")
    return(rd)
}

# ==============================================================================
# 2. FREQUENCY FEATURES
# ==============================================================================
cat("\nCreating frequency features...\n")

#' Calculate daily incident frequency statistics
create_daily_frequency <- function(rd, df) {
    # Get incidents per segment per day
    daily <- as.data.table(sqldf("
    SELECT DISTINCT segment_id, COUNT(*) as op, day_year
    FROM df
    GROUP BY segment_id, day_year
  "))

    # Calculate difference between consecutive days
    daily <- daily %>%
        group_by(segment_id) %>%
        arrange(day_year) %>%
        mutate(Diff = day_year - lag(day_year))

    daily$Diff[is.na(daily$Diff)] <- 0

    # Filter to only positive differences
    daily_filtered <- daily[daily$Diff > 0, ]

    # Aggregate frequency statistics
    freq_daily <- daily_filtered %>%
        group_by(segment_id) %>%
        summarise(
            freq_d_mean = mean(Diff, na.rm = TRUE),
            freq_d_max = max(Diff, na.rm = TRUE),
            freq_d_min = min(Diff, na.rm = TRUE),
            freq_d_med = median(Diff, na.rm = TRUE),
            freq_d_sd = sd(Diff, na.rm = TRUE)
        )

    rd <- merge(rd, freq_daily, by = "segment_id", all.x = TRUE)

    # Fill NAs with high values (segments without frequency data)
    rd$freq_d_mean[is.na(rd$freq_d_mean)] <- 500
    rd$freq_d_max[is.na(rd$freq_d_max)] <- 1000
    rd$freq_d_min[is.na(rd$freq_d_min)] <- 500
    rd$freq_d_med[is.na(rd$freq_d_med)] <- 500
    rd$freq_d_sd[is.na(rd$freq_d_sd)] <- 300

    cat("  Added daily frequency features\n")
    return(rd)
}

#' Calculate hourly incident frequency statistics
create_hourly_frequency <- function(rd, df, train) {
    # Create consecutive hour mapping
    hour_map <- as.data.table(sqldf("SELECT DISTINCT day_year, hour FROM train"))
    hour_map$hour_cons <- seq.int(nrow(hour_map))

    # Add hour_cons to incidents
    d <- merge(df[, c("segment_id", "hour", "day_year")], hour_map,
        by = c("day_year", "hour"), all.x = TRUE
    )

    # Get incidents per segment per consecutive hour
    hourly <- as.data.table(sqldf("
    SELECT DISTINCT segment_id, COUNT(*) as op, hour_cons
    FROM d
    GROUP BY segment_id, hour_cons
  "))

    # Calculate difference between consecutive hours
    hourly <- hourly %>%
        group_by(segment_id) %>%
        arrange(hour_cons) %>%
        mutate(Diff = hour_cons - lag(hour_cons, default = first(hour_cons)))

    hourly$Diff[hourly$Diff == 0] <- NA

    # Aggregate frequency statistics
    freq_hourly <- hourly %>%
        filter(!is.na(Diff)) %>%
        group_by(segment_id) %>%
        summarise(
            freq_h_mean = mean(Diff, na.rm = TRUE),
            freq_h_max = max(Diff, na.rm = TRUE),
            freq_h_min = min(Diff, na.rm = TRUE),
            freq_h_med = median(Diff, na.rm = TRUE),
            freq_h_sd = sd(Diff, na.rm = TRUE)
        )

    rd <- merge(rd, freq_hourly, by = "segment_id", all.x = TRUE)

    # Fill NAs
    rd$freq_h_mean[is.na(rd$freq_h_mean)] <- 10000
    rd$freq_h_max[is.na(rd$freq_h_max)] <- 20000
    rd$freq_h_min[is.na(rd$freq_h_min)] <- 10000
    rd$freq_h_med[is.na(rd$freq_h_med)] <- 10000
    rd$freq_h_sd[is.na(rd$freq_h_sd)] <- 3000

    cat("  Added hourly frequency features\n")
    return(rd)
}

#' Calculate preferred day and hour features
create_preference_features <- function(rd, df) {
    # Preferred day of week
    day_pref <- as.data.table(sqldf("
    SELECT segment_id, day_week, COUNT(*) as op,
           ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY COUNT(*) DESC) as rn
    FROM df
    GROUP BY segment_id, day_week
  "))

    day_pref_1 <- day_pref[day_pref$rn == 1, c("segment_id", "day_week", "op")]
    day_pref_2 <- day_pref[day_pref$rn == 2, c("segment_id", "day_week", "op")]

    setnames(day_pref_1, c("day_week", "op"), c("day_pref_1", "day_op_1"))
    setnames(day_pref_2, c("day_week", "op"), c("day_pref_2", "day_op_2"))

    rd <- merge(rd, day_pref_1, by = "segment_id", all.x = TRUE)
    rd <- merge(rd, day_pref_2, by = "segment_id", all.x = TRUE)

    # Calculate percentage
    rd$prct_day_pref_1 <- rd$day_op_1 / rd$nb_incd
    rd$prct_day_pref_2 <- rd$day_op_2 / rd$nb_incd

    # Fill NAs
    rd$day_pref_1[is.na(rd$day_pref_1)] <- 8
    rd$day_pref_2[is.na(rd$day_pref_2)] <- 8
    rd$prct_day_pref_1[is.na(rd$prct_day_pref_1)] <- 1
    rd$prct_day_pref_2[is.na(rd$prct_day_pref_2)] <- 1

    # Preferred hour
    hour_pref <- as.data.table(sqldf("
    SELECT segment_id, hour, COUNT(*) as op,
           ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY COUNT(*) DESC) as rn
    FROM df
    GROUP BY segment_id, hour
  "))

    hour_pref_1 <- hour_pref[hour_pref$rn == 1, c("segment_id", "hour", "op")]
    hour_pref_2 <- hour_pref[hour_pref$rn == 2, c("segment_id", "hour", "op")]
    hour_pref_3 <- hour_pref[hour_pref$rn == 3, c("segment_id", "hour", "op")]

    setnames(hour_pref_1, c("hour", "op"), c("hour_pref_1", "hour_op_1"))
    setnames(hour_pref_2, c("hour", "op"), c("hour_pref_2", "hour_op_2"))
    setnames(hour_pref_3, c("hour", "op"), c("hour_pref_3", "hour_op_3"))

    rd <- merge(rd, hour_pref_1, by = "segment_id", all.x = TRUE)
    rd <- merge(rd, hour_pref_2, by = "segment_id", all.x = TRUE)
    rd <- merge(rd, hour_pref_3, by = "segment_id", all.x = TRUE)

    # Calculate percentage
    rd$prct_hour_pref_1 <- rd$hour_op_1 / rd$nb_incd
    rd$prct_hour_pref_2 <- rd$hour_op_2 / rd$nb_incd
    rd$prct_hour_pref_3 <- rd$hour_op_3 / rd$nb_incd

    # Fill NAs
    rd$hour_pref_1[is.na(rd$hour_pref_1)] <- 25
    rd$hour_pref_2[is.na(rd$hour_pref_2)] <- 25
    rd$hour_pref_3[is.na(rd$hour_pref_3)] <- 25
    rd$prct_hour_pref_1[is.na(rd$prct_hour_pref_1)] <- 1
    rd$prct_hour_pref_2[is.na(rd$prct_hour_pref_2)] <- 1
    rd$prct_hour_pref_3[is.na(rd$prct_hour_pref_3)] <- 1

    # Cleanup temp columns
    rd$day_op_1 <- NULL
    rd$day_op_2 <- NULL
    rd$hour_op_1 <- NULL
    rd$hour_op_2 <- NULL
    rd$hour_op_3 <- NULL

    cat("  Added preference features\n")
    return(rd)
}

# ==============================================================================
# 3. TIME-BASED FEATURES
# ==============================================================================
cat("\nCreating time-based features...\n")

#' Add time-based features to training/test data
add_time_features <- function(data) {
    # Special day indicator
    data$special <- "N" # Normal
    data$special[data$day_year %in% SPECIAL_DAYS] <- "S" # Special
    data$special[data$day_year >= HOLIDAY_START & data$day_year <= HOLIDAY_END] <- "H" # Holiday
    data$special <- as.factor(data$special)

    # Weekend indicator
    data$weekend <- ifelse(is.weekend(data$Date), 0, 1)

    # Hour category (categorical)
    data$hour_cat <- as.factor(data$hour)

    # Hour WEO (time of day categories for WEather Operations)
    data$hour_weo <- "V20_23" # Default evening
    data$hour_weo[data$hour <= 2] <- "Z0_2"
    data$hour_weo[data$hour >= 3 & data$hour <= 6] <- "T3_6"
    data$hour_weo[data$hour >= 7 & data$hour <= 9] <- "S7_9"
    data$hour_weo[data$hour >= 10 & data$hour <= 12] <- "D10_12"
    data$hour_weo[data$hour >= 13 & data$hour <= 16] <- "T13_16"
    data$hour_weo[data$hour >= 17 & data$hour <= 19] <- "D17_19"
    data$hour_weo <- as.factor(data$hour_weo)

    # Day of week
    data$day_week <- wday(as.Date(data$Date))

    cat("  Added time-based features\n")
    return(data)
}

#' Calculate preference difference features
add_preference_diff_features <- function(data) {
    # Day preference percentage
    data$prct_day_pref <- data$prct_day_pref_1 + data$prct_day_pref_2

    data$prct_day <- ifelse(
        abs(data$day_week - data$day_pref_1) == 0, data$prct_day_pref_1,
        ifelse(abs(data$day_week - data$day_pref_2) == 0, data$prct_day_pref_2,
            (1 - data$prct_day_pref) / 5
        )
    )

    # Hour preference percentage
    data$prct_hour_pref <- data$prct_hour_pref_1 + data$prct_hour_pref_2 + data$prct_hour_pref_3

    data$prct_hour_1 <- ifelse(
        abs(data$hour - data$hour_pref_1) == 0, data$prct_hour_pref_1,
        ifelse(abs(data$hour - data$hour_pref_2) == 0, data$prct_hour_pref_2,
            ifelse(abs(data$hour - data$hour_pref_3) == 0, data$prct_hour_pref_3,
                (1 - data$prct_hour_pref) / 21
            )
        )
    )

    data$prct_hour_2 <- ifelse(
        abs(data$hour - data$hour_pref_1) == 1, data$prct_hour_pref_1,
        ifelse(abs(data$hour - data$hour_pref_2) == 1, data$prct_hour_pref_2,
            ifelse(abs(data$hour - data$hour_pref_3) == 1, data$prct_hour_pref_3,
                (1 - data$prct_hour_pref) / 18
            )
        )
    )

    # Difference from preferred
    data$diff_day <- data$prct_day_pref_1 - data$prct_day
    data$diff_hour <- ((data$prct_hour_pref_1 - data$prct_hour_1) +
        (data$prct_hour_pref_2 - data$prct_hour_2)) / 2

    cat("  Added preference difference features\n")
    return(data)
}

cat("\nFeature engineering functions loaded!\n")
