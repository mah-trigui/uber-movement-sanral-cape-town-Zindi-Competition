# ==============================================================================
# 01_data_loading.R - Load All Data Sources
# ==============================================================================
# Project: Traffic Incident Prediction
# Description: Load train, road network, VDS, weather, and geographic data
# ==============================================================================

source("organized/00_config.R")
print_header("Step 1: Loading Data")

# ==============================================================================
# 1. INCIDENT DATA (Training Labels)
# ==============================================================================
cat("Loading incident data...\n")

df <- as.data.table(read.csv("train.csv"))
cat("  Raw incidents:", nrow(df), "rows\n")

# ==============================================================================
# 2. ROAD NETWORK DATA
# ==============================================================================
cat("\nLoading road network data...\n")

# DBF file (attributes)
road_dbf <- read.dbf("road_segments.dbf")

# Shapefile (with geometry)
road_sf <- sf::read_sf("road_segments.shp")

# Convert to data.table
rd <- as.data.table(road_dbf)
cat("  Road segments:", nrow(rd), "\n")

# ==============================================================================
# 3. VEHICLE DETECTION SYSTEM (VDS) DATA
# ==============================================================================
cat("\nLoading VDS hourly data...\n")

# All cameras reference
vds_cameras <- as.data.table(read.csv("All_Cam.csv"))

# Training period VDS (Sep-Dec 2018)
vds_09_18 <- fread("WC September 2018 Hourly.csv")
vds_10_18 <- fread("WC October 2018 Hourly.csv")
vds_11_18 <- fread("WC November 2018 Hourly.csv")
vds_12_18 <- fread("WC December 2018 Hourly.csv")

cat("  VDS September 2018:", nrow(vds_09_18), "rows\n")
cat("  VDS October 2018:", nrow(vds_10_18), "rows\n")
cat("  VDS November 2018:", nrow(vds_11_18), "rows\n")
cat("  VDS December 2018:", nrow(vds_12_18), "rows\n")

# Test period VDS (Jan-Mar 2019)
vds_01_19 <- fread("WC January 2019 Hourly.csv")
vds_02_19 <- fread("WC February 2019 Hourly.csv")
vds_03_19 <- fread("WC March 2019 Hourly.csv")

cat("  VDS January 2019:", nrow(vds_01_19), "rows\n")
cat("  VDS February 2019:", nrow(vds_02_19), "rows\n")
cat("  VDS March 2019:", nrow(vds_03_19), "rows\n")

# ==============================================================================
# 4. WEATHER DATA
# ==============================================================================
cat("\nLoading weather data...\n")

# Training period weather
weath_25 <- fread("Weath twon city 25.csv", sep = ";")
weath_26 <- fread("Weath paarl 26.csv", sep = ";")
weath_27 <- fread("Weath strand 27.csv", sep = ";")
weath_28 <- fread("Weath airbase 28.csv", sep = ";")
weath_29 <- fread("Weath airport 29.csv", sep = ";")

cat("  Weather stations loaded: 5\n")

# Test period weather
weath_t_25 <- fread("Wea_Test_City_25.csv", sep = ";")
weath_t_26 <- fread("Wea_Test_Paarl_26.csv", sep = ";")
weath_t_27 <- fread("Wea_Test_Strand_27.csv", sep = ";")
weath_t_28 <- fread("Wea_Test_Airbase_28.csv", sep = ";")
weath_t_29 <- fread("Wea_Test_Airport_29.csv", sep = ";")

cat("  Test weather stations loaded: 5\n")

# ==============================================================================
# 5. VEHICLE DATA (Optional)
# ==============================================================================
cat("\nLoading vehicle registration data...\n")

vehic <- as.data.table(read.csv("Vehicles2016_2019.csv"))
cat("  Vehicle records:", nrow(vehic), "\n")

# ==============================================================================
# 6. UBER TRAVEL TIME DATA (Optional)
# ==============================================================================
cat("\nLoading Uber travel time data...\n")

travel_zone <- geojsonsf::geojson_sf("cape_town_travel_zones.json")
travel_zone_dt <- as.data.table(travel_zone)

travel_time <- fread("cape_town-travel_zones-2018-4-All-MonthlyAggregate.csv")
cat("  Travel zones:", nrow(travel_zone_dt), "\n")
cat("  Travel time records:", nrow(travel_time), "\n")

# ==============================================================================
# 7. SAMPLE SUBMISSION
# ==============================================================================
cat("\nLoading sample submission...\n")

# sample_submission <- as.data.table(read.csv("SampleSubmission.csv"))
# cat("  Submission rows:", nrow(sample_submission), "\n")

# ==============================================================================
# SAVE RAW DATA
# ==============================================================================
cat("\nSaving raw data objects...\n")

saveRDS(df, paste0(OUTPUT_DIR, "df_raw.rds"))
saveRDS(rd, paste0(OUTPUT_DIR, "rd_raw.rds"))
saveRDS(list(
    vds_09_18 = vds_09_18,
    vds_10_18 = vds_10_18,
    vds_11_18 = vds_11_18,
    vds_12_18 = vds_12_18
), paste0(OUTPUT_DIR, "vds_train.rds"))

cat("\nData loading complete!\n")
