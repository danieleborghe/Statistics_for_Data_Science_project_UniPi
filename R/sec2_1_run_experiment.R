# R/sec2_1_run_experiment.R
#
# Purpose:
# This script executes the Adaptive Prediction Sets experiment (Paper Section 2.1).
# It includes multiple runs for marginal coverage histogram and a single detailed
# evaluation run using BASE_SEED.
#
# Structure:
#   0. Setup: Create Results Directories, Load Libraries
#   1. Experiment Settings: Define parameters like ALPHA, N_RUNS, seeds, data proportions.
#   2. Multiple Runs for Marginal Coverage Distribution:
#      2.1 Load Full Dataset (once)
#      2.2 Define Loop-Specific Data Split Parameters
#      2.3 Start N_RUNS Loop
#          - Iteration X: Setup (Seed, Logging)
#          - Iteration X: Data Splitting
#          - Iteration X: Model Training
#          - Iteration X: Calibration (Adaptive Scores)
#          - Iteration X: Prediction (Adaptive Sets)
#          - Iteration X: Store Empirical Coverage
#   3. Analyze and Save Marginal Coverage Distribution from N_RUNS:
#      3.1 Calculate and Print Summary Statistics (Mean, SD)
#      3.2 Save Raw Coverage Values to CSV
#      3.3 Plot and Save Histogram of Coverages
#   4. Detailed Single-Run Evaluation (using BASE_SEED):
#      4.1 Setup for Single Detailed Run (Set BASE_SEED)
#      4.2 Data Splitting for Single Run
#      4.3 Model Training for Single Run
#      4.4 Calibration for Single Run (Adaptive Scores)
#      4.5 Prediction for Single Run (Adaptive Sets)
#      4.6 Save Detailed CSV of Predictions for Single Run
#      4.7 Evaluate and Save All Metrics for Single Run:
#          4.7.1 Single-Run Marginal Coverage
#          4.7.2 Single-Run Set Sizes (Summary, Raw, Histogram)
#          4.7.3 Single-Run FSC (Table, Plot)
#          4.7.4 Single-Run SSC (Table, Plot)

cat("INFO: Sourcing shared R scripts for Experiment (Section 2.1 - Adaptive)...\n")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")

cat("INFO: --- Starting Experiment: Adaptive Prediction Sets (Section 2.1) with Multiple Runs for Coverage ---\n")

# --- 0. Setup: Load Libraries ---
all_required_packages <- c("e1071", "dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# --- 0. Setup: Create Results Directories ---
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section2_1_adaptive" 
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME_SUFFIX)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/ (subfolders: ", METHOD_NAME_SUFFIX, ")'\n"))

# --- 1. Experiment Settings ---
cat("INFO: Defining experiment settings for Adaptive Prediction Sets...\n")
BASE_SEED <- 43 # Different base seed for this experiment
ALPHA_CONF <- 0.1 
PROP_TRAIN <- 0.7 
PROP_CALIB <- 0.1 
N_RUNS <- 100 
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", N_RUNS_FOR_COVERAGE_HISTOGRAM=", N_RUNS, "\n"))
cat(paste0("INFO: PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))
cat(paste0("INFO: BASE_SEED for detailed single run analysis: ", BASE_SEED, "\n"))

# --- 2. Multiple Runs for Marginal Coverage Distribution ---
all_empirical_coverages_adaptive <- numeric(N_RUNS)
cat(paste0("INFO: Starting ", N_RUNS, " runs to collect marginal coverage data (Adaptive Method)...\n"))

# --- 2.1 Load Full Dataset (once) ---
iris_data_full <- load_iris_for_classification() 
n_total <- nrow(iris_data_full)

# --- 2.2 Define Loop-Specific Data Split Parameters ---
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop

# --- 2.3 Start N_RUNS Loop for Marginal Coverage ---
for (run_iter in 1:N_RUNS) {
  # ------ Iteration X: Setup (Seed, Logging) ------
  current_run_seed <- BASE_SEED + run_iter 
  set.seed(current_run_seed)
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: --- Run ", run_iter, "/", N_RUNS, " (Seed: ", current_run_seed, ") (Adaptive) ---\n"))
  }
  
  # ------ Iteration X: Data Splitting ------
  shuffled_indices_for_run <- sample(n_total)
  current_data_for_run <- iris_data_full[shuffled_indices_for_run, ]
  train_df_iter <- current_data_for_run[1:n_train_loop, ]
  calib_df_iter <- current_data_for_run[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
  test_df_iter <- current_data_for_run[(n_train_loop + n_calib_loop + 1):n_total, ]
  
  # ------ Iteration X: Model Training ------
  svm_formula <- Species ~ . 
  svm_model_iter <- train_svm_model(svm_formula, train_df_iter)
  
  # ------ Iteration X: Calibration (Adaptive Scores) ------
  calib_probs_iter <- predict_svm_probabilities(svm_model_iter, calib_df_iter)
  calib_true_labels_iter <- calib_df_iter$Species
  non_conf_scores_iter <- get_non_conformity_scores_adaptive(calib_probs_iter, calib_true_labels_iter) 
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iteration X: Prediction (Adaptive Sets) ------
  test_probs_iter <- predict_svm_probabilities(svm_model_iter, test_df_iter)
  test_true_labels_iter <- test_df_iter$Species
  prediction_sets_iter <- create_prediction_sets_adaptive(test_probs_iter, q_hat_iter) 
  
  # ------ Iteration X: Store Empirical Coverage ------
  current_coverage <- calculate_empirical_coverage(prediction_sets_iter, test_true_labels_iter)
  all_empirical_coverages_adaptive[run_iter] <- current_coverage
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: Run ", run_iter, " coverage (Adaptive): ", round(current_coverage, 3), "\n"))
  }
}
cat("INFO: All ", N_RUNS, " runs for marginal coverage completed (Adaptive Method).\n")

# --- 3. Analyze and Save Marginal Coverage Distribution from N_RUNS ---
cat("\nINFO: --- Analyzing Marginal Coverage Distribution (Adaptive Method) ---\n")
# --- 3.1 Calculate and Print Summary Statistics ---
mean_empirical_coverage_adaptive <- mean(all_empirical_coverages_adaptive, na.rm = TRUE)
sd_empirical_coverage_adaptive <- sd(all_empirical_coverages_adaptive, na.rm = TRUE)
cat(paste0("RESULT: Mean Empirical Marginal Coverage (Adaptive) over ", N_RUNS, " runs: ", round(mean_empirical_coverage_adaptive, 3), "\n"))
cat(paste0("RESULT: Std. Dev. of Empirical Marginal Coverage (Adaptive): ", round(sd_empirical_coverage_adaptive, 3), "\n"))
cat(paste0("RESULT: Target Coverage was >= ", 1 - ALPHA_CONF, "\n"))

# --- 3.2 Save Raw Coverage Values to CSV ---
coverage_distribution_df_adaptive <- data.frame(run_iteration = 1:N_RUNS, empirical_coverage = all_empirical_coverages_adaptive)
coverage_dist_filename <- file.path(TABLES_DIR, "coverage_distribution_adaptive.csv")
write.csv(coverage_distribution_df_adaptive, coverage_dist_filename, row.names = FALSE)
cat(paste0("INFO: Distribution of ", N_RUNS, " empirical coverages (Adaptive) saved to '", coverage_dist_filename, "'\n"))

# --- 3.3 Plot and Save Histogram of Coverages ---
coverage_hist_filename <- file.path(PLOTS_DIR, "histogram_marginal_coverage_adaptive.png")
png(coverage_hist_filename, width=800, height=600)
plot_coverage_histogram(all_empirical_coverages_adaptive, 
                        alpha_conf = ALPHA_CONF, 
                        n_runs = N_RUNS, 
                        method_name = "Adaptive Prediction Sets")
dev.off()
cat(paste0("INFO: Histogram of marginal coverage (Adaptive) saved to '", coverage_hist_filename, "'\n"))

# --- 4. Detailed Single-Run Evaluation (using BASE_SEED for reproducibility) ---
cat("\nINFO: --- Detailed Evaluation for a Single Reproducible Run (Adaptive Method - using BASE_SEED) ---\n")
# --- 4.1 Setup for Single Detailed Run (Set BASE_SEED) ---
set.seed(BASE_SEED) 
cat(paste0("INFO: Performing single data split with BASE_SEED = ", BASE_SEED, " for detailed analysis (Adaptive)...\n"))

# --- 4.2 Data Splitting for Single Run ---
single_run_shuffled_indices <- sample(n_total)
single_run_data <- iris_data_full[single_run_shuffled_indices, ]
single_run_train_df <- single_run_data[1:n_train_loop, ]
single_run_calib_df <- single_run_data[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
single_run_test_df <- single_run_data[(n_train_loop + n_calib_loop + 1):n_total, ]
cat(paste0("INFO: Single run dataset sizes (Adaptive): Train=", nrow(single_run_train_df), 
           ", Calib=", nrow(single_run_calib_df), ", Test=", nrow(single_run_test_df), "\n"))

# --- 4.3 Model Training for Single Run ---
single_run_svm_model <- train_svm_model(svm_formula, single_run_train_df) # svm_formula defined earlier

# --- 4.4 Calibration for Single Run (Adaptive Scores) ---
single_run_calib_probs <- predict_svm_probabilities(single_run_svm_model, single_run_calib_df)
single_run_calib_true_labels <- single_run_calib_df$Species
single_run_non_conf_scores <- get_non_conformity_scores_adaptive(single_run_calib_probs, single_run_calib_true_labels) 
single_run_q_hat_adaptive <- calculate_q_hat(single_run_non_conf_scores, ALPHA_CONF, n_calib = nrow(single_run_calib_df))
cat(paste0("INFO: Single run calibration complete. q_hat (adaptive) = ", round(single_run_q_hat_adaptive, 4), "\n"))

# --- 4.5 Prediction for Single Run (Adaptive Sets) ---
single_run_test_probs <- predict_svm_probabilities(single_run_svm_model, single_run_test_df)
single_run_test_true_labels <- single_run_test_df$Species
single_run_prediction_sets <- create_prediction_sets_adaptive(single_run_test_probs, single_run_q_hat_adaptive) 

# --- 4.6 Save Detailed CSV of Predictions for Single Run ---
save_detailed_test_predictions(
  test_true_labels = single_run_test_true_labels,
  prediction_sets_list = single_run_prediction_sets,
  test_probs_matrix = single_run_test_probs,
  output_directory = TABLES_DIR,
  base_filename = "detailed_test_predictions_adaptive_BASESEED_RUN.csv"
)

# --- 4.7 Evaluate and Save All Metrics for Single Run ---
cat("\nINFO: --- Evaluation Results for Single BASE_SEED Run (Adaptive Method) ---\n")
# ---- 4.7.1 Single-Run Marginal Coverage ----
single_run_empirical_cov <- calculate_empirical_coverage(single_run_prediction_sets, single_run_test_true_labels)
cat(paste0("RESULT (BASE_SEED Run): Empirical Marginal Coverage (Adaptive): ", round(single_run_empirical_cov, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
single_run_coverage_summary <- data.frame(
  method = "Adaptive_Section2.1_BASESEED_Run", alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF, empirical_coverage = single_run_empirical_cov,
  q_hat = single_run_q_hat_adaptive 
)
write.csv(single_run_coverage_summary, file.path(TABLES_DIR, "coverage_summary_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary for BASE_SEED RUN (Adaptive) saved to '", TABLES_DIR, "/coverage_summary_adaptive_BASESEED_RUN.csv'\n"))

# ---- 4.7.2 Single-Run Set Sizes (Summary, Raw, Histogram) ----
single_run_set_sizes <- get_set_sizes(single_run_prediction_sets)
cat("RESULT (BASE_SEED Run): Set Size Statistics (Adaptive):\n"); print(summary(single_run_set_sizes))
single_run_avg_set_size <- mean(single_run_set_sizes, na.rm=TRUE)
cat(paste0("RESULT (BASE_SEED Run): Average set size (Adaptive): ", round(single_run_avg_set_size, 3), "\n"))
single_run_set_size_summary_df <- data.frame(
  Min = min(single_run_set_sizes, na.rm=TRUE), Q1 = quantile(single_run_set_sizes, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(single_run_set_sizes, na.rm=TRUE), Mean = single_run_avg_set_size,
  Q3 = quantile(single_run_set_sizes, 0.75, na.rm=TRUE, names=FALSE), Max = max(single_run_set_sizes, na.rm=TRUE)
)
write.csv(single_run_set_size_summary_df, file.path(TABLES_DIR, "set_size_summary_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
write.csv(data.frame(set_size = single_run_set_sizes), file.path(TABLES_DIR, "set_sizes_raw_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes for BASE_SEED RUN (Adaptive) saved to '", TABLES_DIR, "/'\n"))

single_run_plot_filename_set_size <- file.path(PLOTS_DIR, "histogram_set_sizes_adaptive_BASESEED_RUN.png")
png(single_run_plot_filename_set_size, width=800, height=600)
plot_set_size_histogram(single_run_set_sizes, main_title = "Set Sizes (Adaptive Conformal - Iris - BASE_SEED Run)")
dev.off()
cat(paste0("INFO: Set size histogram for BASE_SEED RUN (Adaptive) saved to '", single_run_plot_filename_set_size, "'\n"))

# ---- 4.7.3 Single-Run FSC (Table, Plot) ----
single_run_fsc_feature_name <- "Petal.Width" 
single_run_fsc_results <- calculate_fsc(single_run_prediction_sets, single_run_test_true_labels, 
                                        single_run_test_df[[single_run_fsc_feature_name]], feature_name = single_run_fsc_feature_name,
                                        num_bins_for_continuous = 4)
cat(paste0("RESULT (BASE_SEED Run): FSC (Adaptive - by ", single_run_fsc_feature_name, "):\n"))
if(!is.na(single_run_fsc_results$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(single_run_fsc_results$min_coverage, 3), "\n"))
  print(single_run_fsc_results$coverage_by_group)
  write.csv(single_run_fsc_results$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",single_run_fsc_feature_name,"_adaptive_BASESEED_RUN.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table for BASE_SEED RUN (Adaptive) saved to '", TABLES_DIR, "/fsc_by_",single_run_fsc_feature_name,"_adaptive_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_fsc <- file.path(PLOTS_DIR, paste0("plot_fsc_",single_run_fsc_feature_name,"_adaptive_BASESEED_RUN.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_fsc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_fsc, width=800, height=600)
    plot_conditional_coverage(single_run_fsc_results$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Adaptive - ", single_run_fsc_feature_name, " - Iris - BASE_SEED Run)"))
    dev.off()
    cat(paste0("INFO: FSC plot for BASE_SEED RUN (Adaptive) saved to '", single_run_plot_filename_fsc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC (BASE_SEED Run - Adaptive): NA or no groups.\n") }

# ---- 4.7.4 Single-Run SSC (Table, Plot) ----
single_run_ssc_bins <- max(1, min(length(unique(single_run_set_sizes)), length(levels(iris_data_full$Species))))
single_run_ssc_results <- calculate_ssc(single_run_prediction_sets, single_run_test_true_labels, num_bins_for_size = single_run_ssc_bins)
cat("RESULT (BASE_SEED Run): SSC (Adaptive):\n")
if(!is.na(single_run_ssc_results$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(single_run_ssc_results$min_coverage, 3), "\n"))
  print(single_run_ssc_results$coverage_by_group)
  write.csv(single_run_ssc_results$coverage_by_group, file.path(TABLES_DIR, "ssc_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table for BASE_SEED RUN (Adaptive) saved to '", TABLES_DIR, "/ssc_adaptive_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_ssc <- file.path(PLOTS_DIR, "plot_ssc_adaptive_BASESEED_RUN.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_ssc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_ssc, width=800, height=600)
    plot_conditional_coverage(single_run_ssc_results$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Adaptive - Iris - BASE_SEED Run)")
    dev.off()
    cat(paste0("INFO: SSC plot for BASE_SEED RUN (Adaptive) saved to '", single_run_plot_filename_ssc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC (BASE_SEED Run - Adaptive): NA or no groups.\n") }

cat("INFO: --- End of Adaptive Prediction Sets Experiment ---\n")