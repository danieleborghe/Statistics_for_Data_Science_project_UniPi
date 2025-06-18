# R/sec2_2_run_experiment.R
#
# Purpose:
# This script executes the Conformalized Quantile Regression experiment (Paper Section 2.2).
# It uses the Iris dataset for a regression task (predicting Petal.Length). The script
# includes multiple runs for coverage distribution and a single, detailed evaluation.
#
# Structure:
#   0. Setup: Create Results Directories, Load Libraries
#   1. Experiment Settings: Define parameters like ALPHA, N_RUNS, seeds.
#   2. Multiple Runs for Marginal Coverage & Width Distribution:
#      2.1 Load Full Dataset (once)
#      2.2 Define Loop-Specific Data Split Parameters
#      2.3 Start N_RUNS Loop
#          - Iteration X: Data Splitting
#          - Iteration X: Model Training
#          - Iteration X: Calibration
#          - Iteration X: Prediction
#          - Iteration X: Store Metrics (Coverage and Width)
#   3. Analyze and Save Marginal Coverage & Width Distribution from N_RUNS:
#      3.1 Calculate and Print Summary Statistics
#      3.2 Save Raw Distribution Values
#      3.3 Plot and Save Histogram of Coverages
#   4. Detailed Single-Run Evaluation (using BASE_SEED):
#      4.1 Setup for Single Detailed Run
#      4.2 Data Splitting for Single Run
#      4.3 Model Training for Single Run
#      4.4 Calibration for Single Run
#      4.5 Prediction for Single Run
#      4.6 Save Detailed CSV of Test Intervals
#      4.7 Evaluate and Save All Metrics for Single Run:
#          4.7.1 Single-Run Marginal Coverage & Width Summary
#          4.7.2 Single-Run Interval Widths (Summary, Raw, Histogram)
#          4.7.3 Single-Run FSC (Table, Plot)
#          4.7.4 Single-Run SSC (Table, Plot)

# --- 0. Setup: Sourcing and Libraries ---
cat("INFO: Sourcing shared R scripts for Experiment (Section 2.2 - Quantile Regression)...\n")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")

cat("INFO: --- Starting Experiment: Conformalized Quantile Regression (Section 2.2) ---\n")

# --- 0. Setup: Load Libraries ---
all_required_packages <- c("e1071", "dplyr", "ggplot2", "quantreg")
check_and_load_packages(all_required_packages)

# --- 0. Setup: Create Results Directories ---
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section2_2_quantile_reg"
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME_SUFFIX)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/ (subfolders: ", METHOD_NAME_SUFFIX, ")'\n"))

# --- 1. Experiment Settings ---
cat("INFO: Defining experiment settings...\n")
BASE_SEED <- 2024
ALPHA_CONF <- 0.1
PROP_TRAIN <- 0.7
PROP_CALIB <- 0.15
N_RUNS <- 100
TARGET_VARIABLE <- "Petal.Length"
REG_FORMULA <- as.formula(paste(TARGET_VARIABLE, "~ ."))
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", N_RUNS=", N_RUNS, "\n"))
cat(paste0("INFO: BASE_SEED for detailed single run analysis: ", BASE_SEED, "\n"))

# --- 2. Multiple Runs for Marginal Coverage & Width Distribution ---
all_empirical_coverages_qr <- numeric(N_RUNS)
all_avg_widths_qr <- numeric(N_RUNS)
cat(paste0("INFO: Starting ", N_RUNS, " runs to collect coverage and width data...\n"))

# --- 2.1 Load Full Dataset (once) ---
iris_data_full <- load_iris_for_regression()
n_total <- nrow(iris_data_full)

# --- 2.2 Define Loop-Specific Data Split Parameters ---
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop

# --- 2.3 Start N_RUNS Loop ---
for (run_iter in 1:N_RUNS) {
  current_run_seed <- BASE_SEED + run_iter
  set.seed(current_run_seed)
  
  if (run_iter == 1 || run_iter %% (N_RUNS / 10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: --- Run ", run_iter, "/", N_RUNS, " (Seed: ", current_run_seed, ") ---\n"))
  }
  
  # ------ Iteration X: Data Splitting ------
  shuffled_indices <- sample(n_total)
  train_df_iter <- iris_data_full[shuffled_indices[1:n_train_loop], ]
  calib_df_iter <- iris_data_full[shuffled_indices[(n_train_loop + 1):(n_train_loop + n_calib_loop)], ]
  test_df_iter <- iris_data_full[shuffled_indices[(n_train_loop + n_calib_loop + 1):n_total], ]
  
  # ------ Iteration X: Model Training ------
  quantile_models_iter <- train_quantile_models(REG_FORMULA, train_df_iter, alpha = ALPHA_CONF)
  
  # ------ Iteration X: Calibration ------
  non_conf_scores_iter <- get_non_conformity_scores_quantile(quantile_models_iter, calib_df_iter, TARGET_VARIABLE)
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iteration X: Prediction ------
  prediction_intervals_iter <- create_prediction_intervals_quantile(quantile_models_iter, test_df_iter, q_hat_iter)
  
  # ------ Iteration X: Store Metrics ------
  test_true_values_iter <- test_df_iter[[TARGET_VARIABLE]]
  all_empirical_coverages_qr[run_iter] <- mean((test_true_values_iter >= prediction_intervals_iter$lower_bound) & (test_true_values_iter <= prediction_intervals_iter$upper_bound))
  all_avg_widths_qr[run_iter] <- mean(prediction_intervals_iter$upper_bound - prediction_intervals_iter$lower_bound, na.rm = TRUE)
}
cat("INFO: All ", N_RUNS, " runs completed.\n")

# --- 3. Analyze and Save Marginal Coverage & Width Distribution from N_RUNS ---
cat("\nINFO: --- Analyzing Marginal Coverage & Width Distribution ---\n")
# --- 3.1 Calculate and Print Summary Statistics ---
mean_coverage <- mean(all_empirical_coverages_qr, na.rm = TRUE)
sd_coverage <- sd(all_empirical_coverages_qr, na.rm = TRUE)
mean_width <- mean(all_avg_widths_qr, na.rm = TRUE)
cat(paste0("RESULT: Mean Empirical Coverage over ", N_RUNS, " runs: ", round(mean_coverage, 3), " (SD: ", round(sd_coverage, 3), ")\n"))
cat(paste0("RESULT: Mean Average Interval Width over ", N_RUNS, " runs: ", round(mean_width, 3), "\n"))

# --- 3.2 Save Raw Distribution Values ---
write.csv(
  data.frame(run = 1:N_RUNS, coverage = all_empirical_coverages_qr, avg_width = all_avg_widths_qr),
  file.path(TABLES_DIR, "coverage_width_distribution.csv"), row.names = FALSE
)

# --- 3.3 Plot and Save Histogram of Coverages ---
png(file.path(PLOTS_DIR, "histogram_marginal_coverage.png"), width=800, height=600)
plot_coverage_histogram(all_empirical_coverages_qr, ALPHA_CONF, N_RUNS, "Conformalized Quantile Regression")
dev.off()
cat("INFO: Distribution data and histogram saved.\n")

# --- 4. Detailed Single-Run Evaluation (using BASE_SEED) ---
cat("\nINFO: --- Detailed Evaluation for a Single Reproducible Run ---\n")
# --- 4.1 Setup for Single Detailed Run ---
set.seed(BASE_SEED)
cat(paste0("INFO: Performing single data split with BASE_SEED = ", BASE_SEED, " for detailed analysis...\n"))

# --- 4.2 Data Splitting for Single Run ---
shuffled_indices_single <- sample(n_total)
train_df <- iris_data_full[shuffled_indices_single[1:n_train_loop], ]
calib_df <- iris_data_full[shuffled_indices_single[(n_train_loop + 1):(n_train_loop + n_calib_loop)], ]
test_df <- iris_data_full[shuffled_indices_single[(n_train_loop + n_calib_loop + 1):n_total], ]
cat(paste0("INFO: Single run dataset sizes: Train=", nrow(train_df), 
           ", Calib=", nrow(calib_df), ", Test=", nrow(test_df), "\n"))

# --- 4.3 Model Training for Single Run ---
quantile_models <- train_quantile_models(REG_FORMULA, train_df, alpha = ALPHA_CONF)

# --- 4.4 Calibration for Single Run ---
non_conf_scores <- get_non_conformity_scores_quantile(quantile_models, calib_df, TARGET_VARIABLE)
q_hat <- calculate_q_hat(non_conf_scores, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Single run calibration complete. q_hat = ", round(q_hat, 4), "\n"))

# --- 4.5 Prediction for Single Run ---
prediction_intervals <- create_prediction_intervals_quantile(quantile_models, test_df, q_hat)

# --- 4.6 Save Detailed CSV of Test Intervals ---
intervals_df <- data.frame(
  SampleID = 1:nrow(test_df),
  TrueValue = test_df[[TARGET_VARIABLE]],
  LowerBound = prediction_intervals$lower_bound,
  UpperBound = prediction_intervals$upper_bound
)
intervals_df$IntervalWidth <- intervals_df$UpperBound - intervals_df$LowerBound
intervals_df$Covered <- (intervals_df$TrueValue >= intervals_df$LowerBound) & (intervals_df$TrueValue <= intervals_df$UpperBound)
write.csv(intervals_df, file.path(TABLES_DIR, "detailed_test_intervals_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Detailed test intervals saved to '", file.path(TABLES_DIR, "detailed_test_intervals_BASESEED_RUN.csv"), "'\n"))

# --- 4.7 Evaluate and Save All Metrics for Single Run ---
cat("\nINFO: --- Evaluation Results for Single BASE_SEED Run ---\n")
# ---- 4.7.1 Single-Run Marginal Coverage & Width Summary ----
single_run_coverage <- mean(intervals_df$Covered, na.rm = TRUE)
single_run_avg_width <- mean(intervals_df$IntervalWidth, na.rm = TRUE)
cat(paste0("RESULT (BASE_SEED Run): Empirical Coverage: ", round(single_run_coverage, 3), "\n"))
cat(paste0("RESULT (BASE_SEED Run): Average Interval Width: ", round(single_run_avg_width, 3), "\n"))

summary_df <- data.frame(
  method = METHOD_NAME_SUFFIX, alpha = ALPHA_CONF, target_coverage = 1 - ALPHA_CONF,
  empirical_coverage = single_run_coverage, avg_width = single_run_avg_width, q_hat = q_hat
)
summary_filename <- file.path(TABLES_DIR, "summary_BASESEED_RUN.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)
cat(paste0("INFO: Single run summary saved to '", summary_filename, "'\n"))

# ---- 4.7.2 Single-Run Interval Widths (Summary, Raw, Histogram) ----
cat("RESULT (BASE_SEED Run): Interval Width Statistics:\n"); print(summary(intervals_df$IntervalWidth))

width_summary_df <- data.frame(
  Min = min(intervals_df$IntervalWidth, na.rm=TRUE), 
  Q1 = quantile(intervals_df$IntervalWidth, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(intervals_df$IntervalWidth, na.rm=TRUE), 
  Mean = single_run_avg_width,
  Q3 = quantile(intervals_df$IntervalWidth, 0.75, na.rm=TRUE, names=FALSE), 
  Max = max(intervals_df$IntervalWidth, na.rm=TRUE)
)
width_summary_filename <- file.path(TABLES_DIR, "width_summary_BASESEED_RUN.csv")
write.csv(width_summary_df, width_summary_filename, row.names = FALSE)
width_raw_filename <- file.path(TABLES_DIR, "widths_raw_BASESEED_RUN.csv")
write.csv(data.frame(IntervalWidth = intervals_df$IntervalWidth), width_raw_filename, row.names = FALSE)
cat(paste0("INFO: Width summary and raw widths saved to '", TABLES_DIR, "/'\n"))

width_hist_filename <- file.path(PLOTS_DIR, "histogram_widths_BASESEED_RUN.png")
png(width_hist_filename, width=800, height=600)
hist(intervals_df$IntervalWidth, 
     main = "Interval Width Distribution (BASE_SEED Run)",
     xlab = "Interval Width",
     col = "lightblue",
     border = "black")
dev.off()
cat(paste0("INFO: Interval width histogram saved to '", width_hist_filename, "'\n"))

# ---- 4.7.3 Single-Run FSC (Feature-Stratified Coverage) ----
cat("\nINFO: --- Evaluating FSC for Single BASE_SEED Run ---\n")
fsc_feature_name <- "Sepal.Width" 
cat(paste0("INFO: Calculating FSC stratified by '", fsc_feature_name, "'...\n"))

feature_groups <- cut(test_df[[fsc_feature_name]], breaks = 4, include.lowest = TRUE, ordered_result = TRUE)
fsc_df <- data.frame(feature_group = feature_groups, covered = intervals_df$Covered)

fsc_results_df <- fsc_df %>%
  group_by(feature_group, .drop = FALSE) %>%
  summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
  filter(count > 0)

cat("RESULT (BASE_SEED Run): FSC Results:\n")
print(fsc_results_df)

fsc_table_filename <- file.path(TABLES_DIR, paste0("fsc_by_", fsc_feature_name, "_BASESEED_RUN.csv"))
write.csv(fsc_results_df, fsc_table_filename, row.names = FALSE)
cat(paste0("INFO: FSC results table saved to '", fsc_table_filename, "'\n"))

fsc_plot_filename <- file.path(PLOTS_DIR, paste0("plot_fsc_", fsc_feature_name, "_BASESEED_RUN.png"))
png(fsc_plot_filename, width=800, height=600)
p_fsc <- plot_conditional_coverage(fsc_results_df, 
                                   group_col_name = "feature_group", 
                                   coverage_col_name = "coverage", 
                                   desired_coverage = 1 - ALPHA_CONF, 
                                   main_title = paste0("FSC by ", fsc_feature_name, " (", METHOD_NAME_SUFFIX, ")"))
print(p_fsc)
dev.off()
cat(paste0("INFO: FSC plot saved to '", fsc_plot_filename, "'\n"))

# ---- 4.7.4 Single-Run SSC (Set-Stratified Coverage) ----
cat("\nINFO: --- Evaluating SSC for Single BASE_SEED Run ---\n")
cat("INFO: Calculating SSC stratified by interval width...\n")

width_groups <- cut(intervals_df$IntervalWidth,
                    breaks = quantile(intervals_df$IntervalWidth, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
                    labels = c("Stretto", "Medio", "Largo"),
                    include.lowest = TRUE,
                    ordered_result = TRUE)

ssc_df <- data.frame(size_group = width_groups, covered = intervals_df$Covered)

ssc_results_df <- ssc_df %>%
  group_by(size_group, .drop = FALSE) %>%
  summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
  filter(count > 0)

cat("RESULT (BASE_SEED Run): SSC Results:\n")
print(ssc_results_df)

ssc_table_filename <- file.path(TABLES_DIR, "ssc_BASESEED_RUN.csv")
write.csv(ssc_results_df, ssc_table_filename, row.names = FALSE)
cat(paste0("INFO: SSC results table saved to '", ssc_table_filename, "'\n"))

ssc_plot_filename <- file.path(PLOTS_DIR, "plot_ssc_BASESEED_RUN.png")
png(ssc_plot_filename, width=800, height=600)
p_ssc <- plot_conditional_coverage(ssc_results_df, 
                                   group_col_name = "size_group", 
                                   coverage_col_name = "coverage", 
                                   desired_coverage = 1 - ALPHA_CONF, 
                                   main_title = paste0("SSC by Interval Width (", METHOD_NAME_SUFFIX, ")"))
print(p_ssc)
dev.off()
cat(paste0("INFO: SSC plot saved to '", ssc_plot_filename, "'\n"))

cat("\nINFO: --- End of Conformalized Quantile Regression Experiment ---\n")