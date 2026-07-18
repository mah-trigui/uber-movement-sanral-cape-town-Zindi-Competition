# ==============================================================================
# 02_data_cleaning.R - Data Cleaning and Preprocessing
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Clean incidents data and road network data
# ==============================================================================

source("00_config.R")
print_header("Step 2: Data Cleaning")

# ==============================================================================
# 1. CLEAN INCIDENT DATA
# ==============================================================================
cat("Cleaning incident data...\n")

#' Clean and prepare incident data
clean_incident_data <- function(df) {
    # Filter to 2018 data (Sep-Dec)
    df <- df[df$year == 18 & df$month >= 9, ]

    # Standardize segment_id column name
    if ("road_segment_id" %in% names(df)) {
        setnames(df, "road_segment_id", "segment_id")
    }
    names(df)[which(names(df) == names(df)[6])] <- "segment_id"
    df$segment_id <- as.character(df$segment_id)

    # Remove rows with invalid longitude
    df <- df[!df$longitude == "Closed", ]
    df$longitude <- as.numeric(as.matrix(df$longitude))

    # Remove non-predictive columns
    df$Status <- NULL
    df$Reporting.Agency <- NULL

    # Parse datetime
    df$Date_Time <- as.POSIXct(df$Occurrence.Local.Date.Time, format = "%d/%m/%Y %H:%M")
    df$Occurrence.Local.Date.Time <- NULL

    # Remove 2016 data (too old)
    df$year <- year(df$Date_Time)
    df <- df[!df$year == 16, ]

    # Extract time features
    df$month <- month(df$Date_Time)
    df$day <- mday(as.Date(df$Date_Time))
    df$day_year <- yday(as.Date(df$Date_Time))
    df$day_week <- wday(as.Date(df$Date_Time))
    df$hour <- hour(df$Date_Time)
    df$weekend <- ifelse(is.weekend(df$Date_Time), 0, 1)

    # Clean VehicleType
    df$VehicleType <- as.character(df$VehicleType)
    df$VehicleType[is.na(df$VehicleType)] <- "unknown"
    df$VehicleType[df$VehicleType == "Unable to ID" | df$VehicleType == "Other"] <- "unknown"

    # Clean Color
    df$Color <- as.character(df$Color)
    df$Color[is.na(df$Color)] <- "unknown"

    # Consolidate Cause categories
    df$Cause <- as.character(df$Cause)
    df$Cause[df$Cause == "Stationary Vehicle"] <- "Vehicle"
    df$Cause[df$Cause %in% c(
        "Routine Road Maintenance", "Field Device Maintenance",
        "Road Construction", "Roadworks",
        "Weather & Road Conditions"
    )] <- "Road"
    df$Cause[df$Cause %in% c("Police and Military", "Arrestor")] <- "Forces"
    df$Cause[df$Cause %in% c("Fire", "Fire & Smoke")] <- "Fire"
    df$Cause[df$Cause %in% c("Sporting Events", "Concerts/Other", "Poor Visibility")] <- "Others"

    # Create incident class
    df$incident <- 0
    df$incident[df$Cause == "Vehicle"] <- 1
    df$incident[df$Cause == "Crash"] <- 2
    df$incident[df$Cause == "Congestion"] <- 3
    df$incident[df$Cause == "Forces"] <- 4
    df$incident[df$Cause == "Road"] <- 5
    df$incident[df$Cause == "Fire"] <- 6
    df$incident[df$Cause == "Obstruction"] <- 7
    df$incident[df$Cause == "Lost Load"] <- 8
    df$incident[df$Cause == "Pedestrians"] <- 9

    # Set target = 1 for all incidents
    df$target <- 1

    cat("  Cleaned incidents:", nrow(df), "rows\n")
    return(df)
}

# ==============================================================================
# 2. CLEAN ROAD NETWORK DATA
# ==============================================================================
cat("\nCleaning road network data...\n")

#' Clean and prepare road network data
clean_road_data <- function(rd) {
    rd$segment_id <- as.character(rd$segment_id)

    # Remove invalid segments
    rd <- rd[!(rd$segment_id == "0"), ]

    # Fill missing ROADNO
    rd$ROADNO[is.na(rd$ROADNO)] <- "ARP"

    # Manual fixes for specific segments
    fix_segments <- c("2B7HTHS", "FKBUC13", "S2QPOTD")
    rd$ROADNO[rd$segment_id %in% fix_segments] <- "ARP"
    rd$ROADNO[rd$segment_id == "YLQRLAD"] <- "M5"
    n1_segments <- c(
        "87Z5O7Q", "CCMG98M", "OCR113O", "UABQ9EK", "4SJWPE0",
        "MTPU2BE", "0GZ5KS3", "AQW5HO1", "IEBUIXM"
    )
    rd$ROADNO[rd$segment_id %in% n1_segments] <- "N1"
    rd$ROADNO[rd$segment_id %in% c("LO3764F", "J0SM52K")] <- "N2"

    # Remove unnecessary columns
    rd$REGION <- NULL
    rd$LANES <- NULL # All = 2

    # Create road width category
    rd$road_width <- "O" # Other/Unknown
    rd$road_width[rd$WIDTH == 20.2] <- "L" # Large
    rd$road_width[rd$WIDTH == 12.8] <- "M" # Medium
    rd$road_width[rd$WIDTH == 7.4] <- "S" # Small
    rd$road_width <- as.factor(rd$road_width)
    rd$WIDTH <- NULL

    # Create road length category
    rd$road_len <- cut(rd$length_1,
        breaks = c(0, 500, 1000, 2000, Inf),
        labels = c("S", "M", "L", "XL")
    )

    cat("  Cleaned road segments:", nrow(rd), "\n")
    return(rd)
}

# ==============================================================================
# 3. CLEAN VDS DATA
# ==============================================================================
cat("\nSetting up VDS cleaning functions...\n")

#' Standardize VDS column names
standardize_vds <- function(vds) {
    # Standardize column names
    if ("VDS Station" %in% names(vds)) {
        setnames(vds, "VDS Station", "Name")
    }
    if ("Date" %in% names(vds)) {
        setnames(vds, "Date", "date_col")
    }
    if ("Hour" %in% names(vds)) {
        setnames(vds, "Hour", "hour")
    }
    if ("Vehicle Type" %in% names(vds)) {
        setnames(vds, "Vehicle Type", "vehic_type")
    }
    if ("Number of Vehicles" %in% names(vds)) {
        setnames(vds, "Number of Vehicles", "nb_vehic")
    }
    if ("Average Speed (km/h)" %in% names(vds)) {
        setnames(vds, "Average Speed (km/h)", "speed_avg")
    }

    vds$speed_avg <- as.numeric(as.character(vds$speed_avg))

    # Create vehicle class
    vds$vehic_class <- "Other"
    vds$vehic_class[vds$vehic_type == 1] <- "Light"
    vds$vehic_class[vds$vehic_type == 2] <- "Small_Goods"
    vds$vehic_class[vds$vehic_type == 3] <- "Large"
    vds$vehic_class <- as.factor(vds$vehic_class)

    # Extract date components
    vds$date_new <- as.Date(vds$date_col)
    vds$year <- year(vds$date_new) - 2000
    vds$month <- month(vds$date_new)
    vds$day <- mday(vds$date_new)
    vds$day_year <- yday(vds$date_new)

    return(vds)
}

#' Aggregate VDS data by Name, hour, day
aggregate_vds <- function(vds) {
    result <- as.data.table(sqldf("
    SELECT DISTINCT a.Name, a.hour, date_new, year, month, a.day, day_year,
           b.speed, b.nb_veh_1, c.nb_veh_2, d.nb_veh_3
    FROM vds a
    LEFT JOIN (
      SELECT Name, day, hour, MIN(speed_avg) as speed, MAX(nb_vehic) as nb_veh_1
      FROM vds WHERE vehic_type = 1
      GROUP BY Name, day, hour
    ) b ON a.Name = b.Name AND a.day = b.day AND a.hour = b.hour
    LEFT JOIN (
      SELECT Name, day, hour, MAX(nb_vehic) as nb_veh_2
      FROM vds WHERE vehic_type = 2
      GROUP BY Name, day, hour
    ) c ON a.Name = c.Name AND a.day = c.day AND a.hour = c.hour
    LEFT JOIN (
      SELECT Name, day, hour, MAX(nb_vehic) as nb_veh_3
      FROM vds WHERE vehic_type = 3
      GROUP BY Name, day, hour
    ) d ON a.Name = d.Name AND a.day = d.day AND a.hour = d.hour
  "))

    return(result)
}

# ==============================================================================
# 4. CLEAN WEATHER DATA
# ==============================================================================
cat("\nSetting up weather cleaning functions...\n")

#' Standardize weather data columns
standardize_weather <- function(weath, loc_id) {
    # Rename columns to standard names
    std_names <- c("temp", "press", "press_3h", "humid", "wind", "gust")
    old_names <- c("T", "Po", "Pa", "U", "Ff", "ff10")

    for (i in seq_along(old_names)) {
        if (old_names[i] %in% names(weath)) {
            setnames(weath, old_names[i], std_names[i])
        }
    }

    # Parse datetime
    if ("Local time" %in% names(weath)) {
        weath$Date_Time <- as.POSIXct(weath$`Local time`, format = "%d.%m.%Y %H:%M")
        weath$`Local time` <- NULL
    }

    # Extract time components
    weath$month <- month(weath$Date_Time)
    weath$day <- mday(as.Date(weath$Date_Time))
    weath$day_year <- yday(as.Date(weath$Date_Time))
    weath$hour <- hour(weath$Date_Time)

    # Add location info
    weath$Loc <- paste0("Loc_", loc_id)

    # Add coordinates from WEATHER_STATIONS
    station <- WEATHER_STATIONS[WEATHER_STATIONS$Loc == weath$Loc[1], ]
    weath$long <- station$long
    weath$lat <- station$lat

    return(weath)
}

#' Impute missing weather values
impute_weather <- function(weath) {
    # Use MICE for imputation
    imputed <- complete(mice(data = weath, m = 1, method = "cart", printFlag = FALSE))
    return(as.data.table(imputed))
}

cat("\nData cleaning functions loaded!\n")
