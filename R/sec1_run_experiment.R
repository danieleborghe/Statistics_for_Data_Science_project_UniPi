# R/sec1_run_experiment.R
#
# Purpose:
# This script executes the Basic Conformal Prediction experiment (Paper Section 1).
# It includes multiple runs to generate a distribution of marginal coverage values
# and plots a histogram. Detailed evaluations (set sizes, FSC, SSC, detailed predictions CSV)
# are performed and saved based on a single, reproducible run using BASE_SEED.
#
# Structure:
#   0. Setup: Create Results Directories, Load Libraries
#   1. Experiment Settings: Define parameters like ALPHA, N_RUNS, seeds.
#   2. Multiple Runs for Marginal Coverage Distribution:
#      2.1 Load Full Dataset (once)
#      2.2 Define Loop-Specific Data Split Parameters
#      2.3 Start N_RUNS Loop
#          - Iteration X: Data Splitting
#          - Iteration X: Model Training
#          - Iteration X: Calibration
#          - Iteration X: Prediction
#          - Iteration X: Store Coverage
#   3. Analyze and Save Marginal Coverage Distribution from N_RUNS:
#      3.1 Calculate and Print Summary Statistics
#      3.2 Save Raw Coverage Values
#      3.3 Plot and Save Histogram of Coverages
#   4. Detailed Single-Run Evaluation (using BASE_SEED):
#      4.1 Setup for Single Detailed Run
#      4.2 Data Splitting for Single Run
#      4.3 Model Training for Single Run
#      4.4 Calibration for Single Run
#      4.5 Prediction for Single Run
#      4.6 Save Detailed CSV for Single Run
#      4.7 Evaluate and Save Metrics for Single Run (Coverage, Set Sizes, FSC, SSC)

cat("INFO: Sourcing shared R scripts for Experiment (Section 1 - Basic)...\n")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")

cat("INFO: --- Starting Experiment: Basic Conformal Prediction (Section 1) with Multiple Runs for Coverage ---\n")

# --- 0. Setup: Load Libraries ---
all_required_packages <- c("e1071", "dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# --- 0. Setup: Create Results Directories ---
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section1_basic" 
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME_SUFFIX)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/ (subfolders: ", METHOD_NAME_SUFFIX, ")'\n"))

# --- 1. Experiment Settings ---
cat("INFO: Defining experiment settings...\n")
BASE_SEED <- 42       # Seed for the detailed single run analysis AND base for multi-run seeds
ALPHA_CONF <- 0.1      # Significance level for conformal prediction (target 1-ALPHA_CONF coverage)
PROP_TRAIN <- 0.7      # Proportion of data for training the underlying SVM model
PROP_CALIB <- 0.1      # Proportion of data for the conformal calibration set
N_RUNS <- 100          # Number of runs for marginal coverage distribution
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", N_RUNS_FOR_COVERAGE_HISTOGRAM=", N_RUNS, "\n"))
cat(paste0("INFO: PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))
cat(paste0("INFO: BASE_SEED for detailed single run analysis: ", BASE_SEED, "\n"))

# --- 2. Multiple Runs for Marginal Coverage Distribution ---
all_empirical_coverages_basic <- numeric(N_RUNS)
cat(paste0("INFO: Starting ", N_RUNS, " runs to collect marginal coverage data (Basic Method)...\n"))

# --- 2.1 Load Full Dataset (once) ---
iris_data_full <- load_iris_for_classification() 
n_total <- nrow(iris_data_full)

# --- 2.2 Define Loop-Specific Data Split Parameters ---
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop # Ensure all data is used

# --- 2.3 Start N_RUNS Loop for Marginal Coverage ---
for (run_iter in 1:N_RUNS) {
  current_run_seed <- BASE_SEED + run_iter 
  set.seed(current_run_seed) # Set seed for this specific iteration's split
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: --- Run ", run_iter, "/", N_RUNS, " (Seed: ", current_run_seed, ") ---\n"))
  }
  
  # ------ Iteration: Data Splitting ------
  shuffled_indices_for_run <- sample(n_total) # New shuffle for each run
  current_data_for_run <- iris_data_full[shuffled_indices_for_run, ]
  
  train_df_iter <- current_data_for_run[1:n_train_loop, ]
  calib_df_iter <- current_data_for_run[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
  test_df_iter <- current_data_for_run[(n_train_loop + n_calib_loop + 1):n_total, ]
  
  # ------ Iteration: Model Training ------
  svm_formula <- Species ~ . 
  svm_model_iter <- train_svm_model(svm_formula, train_df_iter)
  
  # ------ Iteration: Calibration ------
  calib_probs_iter <- predict_svm_probabilities(svm_model_iter, calib_df_iter)
  calib_true_labels_iter <- calib_df_iter$Species
  non_conf_scores_iter <- get_non_conformity_scores_basic(calib_probs_iter, calib_true_labels_iter)
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iteration: Prediction on Test Set ------
  test_probs_iter <- predict_svm_probabilities(svm_model_iter, test_df_iter)
  test_true_labels_iter <- test_df_iter$Species
  prediction_sets_iter <- create_prediction_sets_basic(test_probs_iter, q_hat_iter)
  
  # ------ Iteration: Store Empirical Coverage ------
  current_coverage <- calculate_empirical_coverage(prediction_sets_iter, test_true_labels_iter)
  all_empirical_coverages_basic[run_iter] <- current_coverage
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: Run ", run_iter, " coverage (Basic): ", round(current_coverage, 3), "\n"))
  }
}
cat("INFO: All ", N_RUNS, " runs for marginal coverage completed (Basic Method).\n")

# --- 3. Analyze and Save Marginal Coverage Distribution from N_RUNS ---
cat("\nINFO: --- Analyzing Marginal Coverage Distribution (Basic Method) ---\n")
# --- 3.1 Calculate and Print Summary Statistics ---
mean_empirical_coverage <- mean(all_empirical_coverages_basic, na.rm = TRUE)
sd_empirical_coverage <- sd(all_empirical_coverages_basic, na.rm = TRUE)
cat(paste0("RESULT: Mean Empirical Marginal Coverage over ", N_RUNS, " runs: ", round(mean_empirical_coverage, 3), "\n"))
cat(paste0("RESULT: Std. Dev. of Empirical Marginal Coverage: ", round(sd_empirical_coverage, 3), "\n"))
cat(paste0("RESULT: Target Coverage was >= ", 1 - ALPHA_CONF, "\n"))

# --- 3.2 Save Raw Coverage Values ---
coverage_distribution_df_basic <- data.frame(run_iteration = 1:N_RUNS, empirical_coverage = all_empirical_coverages_basic)
coverage_dist_filename <- file.path(TABLES_DIR, "coverage_distribution_basic.csv")
write.csv(coverage_distribution_df_basic, coverage_dist_filename, row.names = FALSE)
cat(paste0("INFO: Distribution of ", N_RUNS, " empirical coverages saved to '", coverage_dist_filename, "'\n"))

# --- 3.3 Plot and Save Histogram of Coverages ---
coverage_hist_filename <- file.path(PLOTS_DIR, "histogram_marginal_coverage_basic.png")
png(coverage_hist_filename, width=800, height=600)
plot_coverage_histogram(all_empirical_coverages_basic, 
                        alpha_conf = ALPHA_CONF, 
                        n_runs = N_RUNS, 
                        method_name = "Basic Conformal Method")
dev.off()
cat(paste0("INFO: Histogram of marginal coverage saved to '", coverage_hist_filename, "'\n"))


# --- 4. Detailed Single-Run Evaluation (using BASE_SEED for reproducibility) ---
cat("\nINFO: --- Detailed Evaluation for a Single Reproducible Run (Basic Method - using BASE_SEED) ---\n")
# --- 4.1 Setup for Single Detailed Run ---
set.seed(BASE_SEED) 
cat(paste0("INFO: Performing single data split with BASE_SEED = ", BASE_SEED, " for detailed analysis...\n"))

# --- 4.2 Data Splitting for Single Run ---
single_run_shuffled_indices <- sample(n_total) # New shuffle based on BASE_SEED
single_run_data <- iris_data_full[single_run_shuffled_indices, ]

# Use n_train_loop, n_calib_loop, n_test_loop for consistent proportions
single_run_train_df <- single_run_data[1:n_train_loop, ]
single_run_calib_df <- single_run_data[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
single_run_test_df <- single_run_data[(n_train_loop + n_calib_loop + 1):n_total, ]
cat(paste0("INFO: Single run dataset sizes: Train=", nrow(single_run_train_df), 
           ", Calib=", nrow(single_run_calib_df), ", Test=", nrow(single_run_test_df), "\n"))

# --- 4.3 Model Training for Single Run ---
single_run_svm_model <- train_svm_model(svm_formula, single_run_train_df) # Formula is already defined

# --- 4.4 Calibration for Single Run ---
single_run_calib_probs <- predict_svm_probabilities(single_run_svm_model, single_run_calib_df)
single_run_calib_true_labels <- single_run_calib_df$Species
single_run_non_conf_scores <- get_non_conformity_scores_basic(single_run_calib_probs, single_run_calib_true_labels)
single_run_q_hat_basic <- calculate_q_hat(single_run_non_conf_scores, ALPHA_CONF, n_calib = nrow(single_run_calib_df))
cat(paste0("INFO: Single run calibration complete. q_hat (basic) = ", round(single_run_q_hat_basic, 4), "\n"))

# --- 4.5 Prediction for Single Run ---
single_run_test_probs <- predict_svm_probabilities(single_run_svm_model, single_run_test_df)
single_run_test_true_labels <- single_run_test_df$Species
single_run_prediction_sets <- create_prediction_sets_basic(single_run_test_probs, single_run_q_hat_basic)

# --- 4.6 Save Detailed CSV for Single Run ---
save_detailed_test_predictions(
  test_true_labels = single_run_test_true_labels,
  prediction_sets_list = single_run_prediction_sets,
  test_probs_matrix = single_run_test_probs,
  output_directory = TABLES_DIR,
  base_filename = "detailed_test_predictions_basic_BASESEED_RUN.csv"
)

# --- 4.7 Evaluate and Save Metrics for Single Run (Coverage, Set Sizes, FSC, SSC) ---
cat("\nINFO: --- Evaluation Results for Single BASE_SEED Run (Basic Method) ---\n")
# ---- 4.7.1 Single-Run Marginal Coverage ----
single_run_empirical_cov <- calculate_empirical_coverage(single_run_prediction_sets, single_run_test_true_labels)
cat(paste0("RESULT (BASE_SEED Run): Empirical Marginal Coverage (Basic): ", round(single_run_empirical_cov, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
single_run_coverage_summary <- data.frame(
  method = "Basic_Section1_BASESEED_Run", alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF, empirical_coverage = single_run_empirical_cov,
  q_hat = single_run_q_hat_basic 
)
write.csv(single_run_coverage_summary, file.path(TABLES_DIR, "coverage_summary_basic_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary for BASE_SEED RUN saved to '", TABLES_DIR, "/coverage_summary_basic_BASESEED_RUN.csv'\n"))

# ---- 4.7.2 Single-Run Set Sizes ----
single_run_set_sizes <- get_set_sizes(single_run_prediction_sets)
cat("RESULT (BASE_SEED Run): Set Size Statistics (Basic):\n"); print(summary(single_run_set_sizes))
single_run_avg_set_size <- mean(single_run_set_sizes, na.rm=TRUE)
cat(paste0("RESULT (BASE_SEED Run): Average set size (Basic): ", round(single_run_avg_set_size, 3), "\n"))
single_run_set_size_summary_df <- data.frame(
  Min = min(single_run_set_sizes, na.rm=TRUE), Q1 = quantile(single_run_set_sizes, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(single_run_set_sizes, na.rm=TRUE), Mean = single_run_avg_set_size,
  Q3 = quantile(single_run_set_sizes, 0.75, na.rm=TRUE, names=FALSE), Max = max(single_run_set_sizes, na.rm=TRUE)
)
write.csv(single_run_set_size_summary_df, file.path(TABLES_DIR, "set_size_summary_basic_BASESEED_RUN.csv"), row.names = FALSE)
write.csv(data.frame(set_size = single_run_set_sizes), file.path(TABLES_DIR, "set_sizes_raw_basic_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes for BASE_SEED RUN saved to '", TABLES_DIR, "/'\n"))

single_run_plot_filename_set_size <- file.path(PLOTS_DIR, "histogram_set_sizes_basic_BASESEED_RUN.png")
png(single_run_plot_filename_set_size, width=800, height=600)
plot_set_size_histogram(single_run_set_sizes, main_title = "Set Sizes (Basic Conformal - Iris - BASE_SEED Run)")
dev.off()
cat(paste0("INFO: Set size histogram for BASE_SEED RUN saved to '", single_run_plot_filename_set_size, "'\n"))

# ---- 4.7.3 Single-Run FSC ----
single_run_fsc_feature_name <- "Sepal.Length" 
single_run_fsc_results <- calculate_fsc(single_run_prediction_sets, single_run_test_true_labels, 
                                        single_run_test_df[[single_run_fsc_feature_name]], feature_name = single_run_fsc_feature_name,
                                        num_bins_for_continuous = 4)
cat(paste0("RESULT (BASE_SEED Run): FSC (Basic - by ", single_run_fsc_feature_name, "):\n"))
if(!is.na(single_run_fsc_results$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(single_run_fsc_results$min_coverage, 3), "\n"))
  print(single_run_fsc_results$coverage_by_group)
  write.csv(single_run_fsc_results$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",single_run_fsc_feature_name,"_basic_BASESEED_RUN.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table for BASE_SEED RUN saved to '", TABLES_DIR, "/fsc_by_",single_run_fsc_feature_name,"_basic_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_fsc <- file.path(PLOTS_DIR, paste0("plot_fsc_",single_run_fsc_feature_name,"_basic_BASESEED_RUN.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_fsc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_fsc, width=800, height=600)
    plot_conditional_coverage(single_run_fsc_results$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Basic - ", single_run_fsc_feature_name, " - Iris - BASE_SEED Run)"))
    dev.off()
    cat(paste0("INFO: FSC plot for BASE_SEED RUN saved to '", single_run_plot_filename_fsc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC (BASE_SEED Run): NA or no groups.\n") }

# ---- 4.7.4 Single-Run SSC ----
single_run_ssc_bins <- max(1, min(length(unique(single_run_set_sizes)), length(levels(iris_data_full$Species))))
single_run_ssc_results <- calculate_ssc(single_run_prediction_sets, single_run_test_true_labels, num_bins_for_size = single_run_ssc_bins)
cat("RESULT (BASE_SEED Run): SSC (Basic):\n")
if(!is.na(single_run_ssc_results$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(single_run_ssc_results$min_coverage, 3), "\n"))
  print(single_run_ssc_results$coverage_by_group)
  write.csv(single_run_ssc_results$coverage_by_group, file.path(TABLES_DIR, "ssc_basic_BASESEED_RUN.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table for BASE_SEED RUN saved to '", TABLES_DIR, "/ssc_basic_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_ssc <- file.path(PLOTS_DIR, "plot_ssc_basic_BASESEED_RUN.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_ssc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_ssc, width=800, height=600)
    plot_conditional_coverage(single_run_ssc_results$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Basic - Iris - BASE_SEED Run)")
    dev.off()
    cat(paste0("INFO: SSC plot for BASE_SEED RUN saved to '", single_run_plot_filename_ssc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC (BASE_SEED Run): NA or no groups.\n") }

cat("INFO: --- End of Basic Conformal Prediction Experiment ---\n")