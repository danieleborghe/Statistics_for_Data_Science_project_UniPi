# R/sec2_4_run_experiment.R
#
# Purpose:
# This script executes the Conformalizing Bayes experiment (Paper Section 2.4).
# It includes multiple runs for marginal coverage histogram and a single detailed
# evaluation run using BASE_SEED.
#
# Structure:
#   0. Setup: Create Results Directories, Load Libraries
#   1. Experiment Settings
#   2. Multiple Runs for Marginal Coverage Distribution
#      (Loop: Split, Train, Calibrate (Bayes), Predict (Bayes), Store Coverage)
#   3. Analyze and Save Marginal Coverage Distribution
#   4. Detailed Single-Run Evaluation (using BASE_SEED)
#      (Split, Train, Calibrate (Bayes), Predict (Bayes), Save Detailed CSV, Evaluate Metrics)

cat("INFO: Sourcing shared R scripts for Experiment (Section 2.4 - Conformalizing Bayes)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R") 
source("R/evaluation_utils.R")
source("R/utils.R")

if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  cat("INFO: ggplot2 package loaded.\n")
}

cat("INFO: --- Starting Experiment: Conformalizing Bayes (Section 2.4) with Multiple Runs for Coverage ---\n")

# --- 0. Setup: Create Results Directories ---
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section2_4_bayes" 
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME_SUFFIX)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/ (subfolders: ", METHOD_NAME_SUFFIX, ")'\n"))

# --- 1. Experiment Settings ---
cat("INFO: Defining experiment settings...\n")
BASE_SEED <- 44 # Different base seed
ALPHA_CONF <- 0.1 
PROP_TRAIN <- 0.7 
PROP_CALIB <- 0.1 
N_RUNS <- 100 
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", N_RUNS_FOR_COVERAGE_HISTOGRAM=", N_RUNS, "\n"))
cat(paste0("INFO: PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))
cat(paste0("INFO: BASE_SEED for detailed single run analysis: ", BASE_SEED, "\n"))

# --- 2. Multiple Runs for Marginal Coverage Distribution ---
all_empirical_coverages_bayes <- numeric(N_RUNS)
cat(paste0("INFO: Starting ", N_RUNS, " runs to collect marginal coverage data (Bayes Method)...\n"))

iris_data_full <- load_iris_data() 
n_total <- nrow(iris_data_full)
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop

for (run_iter in 1:N_RUNS) {
  current_run_seed <- BASE_SEED + run_iter 
  set.seed(current_run_seed)
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: --- Run ", run_iter, "/", N_RUNS, " (Seed: ", current_run_seed, ") ---\n"))
  }
  
  shuffled_indices_for_run <- sample(n_total)
  current_data_for_run <- iris_data_full[shuffled_indices_for_run, ]
  train_df_iter <- current_data_for_run[1:n_train_loop, ]
  calib_df_iter <- current_data_for_run[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
  test_df_iter <- current_data_for_run[(n_train_loop + n_calib_loop + 1):n_total, ]
  
  svm_formula <- Species ~ . 
  svm_model_iter <- train_svm_model(svm_formula, train_df_iter)
  
  calib_probs_iter <- predict_svm_probabilities(svm_model_iter, calib_df_iter)
  calib_true_labels_iter <- calib_df_iter$Species
  # Use BAYES scores
  non_conf_scores_iter <- get_non_conformity_scores_bayes(calib_probs_iter, calib_true_labels_iter) 
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  test_probs_iter <- predict_svm_probabilities(svm_model_iter, test_df_iter)
  test_true_labels_iter <- test_df_iter$Species
  # Use BAYES prediction sets
  prediction_sets_iter <- create_prediction_sets_bayes(test_probs_iter, q_hat_iter) 
  
  current_coverage <- calculate_empirical_coverage(prediction_sets_iter, test_true_labels_iter)
  all_empirical_coverages_bayes[run_iter] <- current_coverage
  
  if (run_iter == 1 || run_iter %% (N_RUNS/10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: Run ", run_iter, " coverage (Bayes): ", round(current_coverage, 3), 
               ", q_hat_iter: ", round(q_hat_iter, 4), "\n"))
  }
}
cat("INFO: All ", N_RUNS, " runs for marginal coverage completed (Bayes Method).\n")

# --- 3. Analyze and Save Marginal Coverage Distribution ---
cat("\nINFO: --- Analyzing Marginal Coverage Distribution (Conformalizing Bayes Method) ---\n")
mean_empirical_coverage_bayes <- mean(all_empirical_coverages_bayes, na.rm = TRUE)
sd_empirical_coverage_bayes <- sd(all_empirical_coverages_bayes, na.rm = TRUE)
cat(paste0("RESULT: Mean Empirical Marginal Coverage (Bayes) over ", N_RUNS, " runs: ", round(mean_empirical_coverage_bayes, 3), "\n"))
cat(paste0("RESULT: Std. Dev. of Empirical Marginal Coverage (Bayes): ", round(sd_empirical_coverage_bayes, 3), "\n"))
cat(paste0("RESULT: Target Coverage was >= ", 1 - ALPHA_CONF, "\n"))

coverage_distribution_df_bayes <- data.frame(run_iteration = 1:N_RUNS, empirical_coverage = all_empirical_coverages_bayes)
coverage_dist_filename <- file.path(TABLES_DIR, "coverage_distribution_bayes.csv")
write.csv(coverage_distribution_df_bayes, coverage_dist_filename, row.names = FALSE)
cat(paste0("INFO: Distribution of ", N_RUNS, " empirical coverages saved to '", coverage_dist_filename, "'\n"))

coverage_hist_filename <- file.path(PLOTS_DIR, "histogram_marginal_coverage_bayes.png")
png(coverage_hist_filename, width=800, height=600)
plot_coverage_histogram(all_empirical_coverages_bayes, 
                        alpha_conf = ALPHA_CONF, 
                        n_runs = N_RUNS, 
                        method_name = "Conformalizing Bayes Method")
dev.off()
cat(paste0("INFO: Histogram of marginal coverage saved to '", coverage_hist_filename, "'\n"))

# --- 4. Detailed Single-Run Evaluation (using BASE_SEED for reproducibility) ---
cat("\nINFO: --- Detailed Evaluation for a Single Reproducible Run (Conformalizing Bayes - using BASE_SEED) ---\n")
set.seed(BASE_SEED) 

cat(paste0("INFO: Performing single data split with BASE_SEED = ", BASE_SEED, " for detailed analysis (Bayes)...\n"))
single_run_shuffled_indices <- sample(n_total)
single_run_data <- iris_data_full[single_run_shuffled_indices, ]
single_run_train_df <- single_run_data[1:n_train_loop, ]
single_run_calib_df <- single_run_data[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
single_run_test_df <- single_run_data[(n_train_loop + n_calib_loop + 1):n_total, ]
cat(paste0("INFO: Single run dataset sizes (Bayes): Train=", nrow(single_run_train_df), 
           ", Calib=", nrow(single_run_calib_df), ", Test=", nrow(single_run_test_df), "\n"))

single_run_svm_model <- train_svm_model(svm_formula, single_run_train_df)

single_run_calib_probs <- predict_svm_probabilities(single_run_svm_model, single_run_calib_df)
single_run_calib_true_labels <- single_run_calib_df$Species
# Use BAYES scores
single_run_non_conf_scores <- get_non_conformity_scores_bayes(single_run_calib_probs, single_run_calib_true_labels) 
single_run_q_hat_bayes <- calculate_q_hat(single_run_non_conf_scores, ALPHA_CONF, n_calib = nrow(single_run_calib_df))
cat(paste0("INFO: Single run calibration complete. q_hat (Bayes) = ", round(single_run_q_hat_bayes, 4), "\n"))
cat(paste0("INFO: Single run prediction threshold will be -q_hat = ", round(-single_run_q_hat_bayes, 4), "\n"))

single_run_test_probs <- predict_svm_probabilities(single_run_svm_model, single_run_test_df)
single_run_test_true_labels <- single_run_test_df$Species
# Use BAYES prediction sets
single_run_prediction_sets <- create_prediction_sets_bayes(single_run_test_probs, single_run_q_hat_bayes) 

save_detailed_test_predictions(
  test_true_labels = single_run_test_true_labels,
  prediction_sets_list = single_run_prediction_sets,
  test_probs_matrix = single_run_test_probs,
  output_directory = TABLES_DIR,
  base_filename = "detailed_test_predictions_bayes_BASESEED_RUN.csv"
)

single_run_empirical_cov_bayes <- calculate_empirical_coverage(single_run_prediction_sets, single_run_test_true_labels)
cat(paste0("RESULT (BASE_SEED Run): Empirical Marginal Coverage (Bayes): ", round(single_run_empirical_cov_bayes, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
single_run_coverage_summary <- data.frame(
  method = "ConformalizingBayes_Section2.4_BASESEED_Run", alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF, empirical_coverage = single_run_empirical_cov_bayes,
  q_hat = single_run_q_hat_bayes, prediction_threshold = -single_run_q_hat_bayes
)
write.csv(single_run_coverage_summary, file.path(TABLES_DIR, "coverage_summary_bayes_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary for BASE_SEED RUN saved to '", TABLES_DIR, "/coverage_summary_bayes_BASESEED_RUN.csv'\n"))

single_run_set_sizes_bayes <- get_set_sizes(single_run_prediction_sets)
cat("RESULT (BASE_SEED Run): Set Size Statistics (Bayes):\n"); print(summary(single_run_set_sizes_bayes))
single_run_avg_set_size_bayes <- mean(single_run_set_sizes_bayes, na.rm=TRUE)
cat(paste0("RESULT (BASE_SEED Run): Average set size (Bayes): ", round(single_run_avg_set_size_bayes, 3), "\n"))
single_run_set_size_summary_df <- data.frame(
  Min = min(single_run_set_sizes_bayes, na.rm=TRUE), Q1 = quantile(single_run_set_sizes_bayes, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(single_run_set_sizes_bayes, na.rm=TRUE), Mean = single_run_avg_set_size_bayes,
  Q3 = quantile(single_run_set_sizes_bayes, 0.75, na.rm=TRUE, names=FALSE), Max = max(single_run_set_sizes_bayes, na.rm=TRUE)
)
write.csv(single_run_set_size_summary_df, file.path(TABLES_DIR, "set_size_summary_bayes_BASESEED_RUN.csv"), row.names = FALSE)
write.csv(data.frame(set_size = single_run_set_sizes_bayes), file.path(TABLES_DIR, "set_sizes_raw_bayes_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes for BASE_SEED RUN saved to '", TABLES_DIR, "/'\n"))

single_run_plot_filename_set_size <- file.path(PLOTS_DIR, "histogram_set_sizes_bayes_BASESEED_RUN.png")
png(single_run_plot_filename_set_size, width=800, height=600)
plot_set_size_histogram(single_run_set_sizes_bayes, main_title = "Set Sizes (Conformalizing Bayes - Iris - BASE_SEED Run)")
dev.off()
cat(paste0("INFO: Set size histogram for BASE_SEED RUN saved to '", single_run_plot_filename_set_size, "'\n"))

single_run_fsc_feature_name <- "Sepal.Width" 
single_run_fsc_results <- calculate_fsc(single_run_prediction_sets, single_run_test_true_labels, 
                                        single_run_test_df[[single_run_fsc_feature_name]], feature_name = single_run_fsc_feature_name,
                                        num_bins_for_continuous = 4)
cat(paste0("RESULT (BASE_SEED Run): FSC (Bayes - by ", single_run_fsc_feature_name, "):\n"))
if(!is.na(single_run_fsc_results$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(single_run_fsc_results$min_coverage, 3), "\n"))
  print(single_run_fsc_results$coverage_by_group)
  write.csv(single_run_fsc_results$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",single_run_fsc_feature_name,"_bayes_BASESEED_RUN.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table for BASE_SEED RUN saved to '", TABLES_DIR, "/fsc_by_",single_run_fsc_feature_name,"_bayes_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_fsc <- file.path(PLOTS_DIR, paste0("plot_fsc_",single_run_fsc_feature_name,"_bayes_BASESEED_RUN.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_fsc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_fsc, width=800, height=600)
    plot_conditional_coverage(single_run_fsc_results$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Bayes - ", single_run_fsc_feature_name, " - Iris - BASE_SEED Run)"))
    dev.off()
    cat(paste0("INFO: FSC plot for BASE_SEED RUN saved to '", single_run_plot_filename_fsc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC (BASE_SEED Run - Bayes): NA or no groups.\n") }

single_run_ssc_bins <- max(1, min(length(unique(single_run_set_sizes_bayes)), length(levels(iris_data_full$Species))))
single_run_ssc_results <- calculate_ssc(single_run_prediction_sets, single_run_test_true_labels, num_bins_for_size = single_run_ssc_bins)
cat("RESULT (BASE_SEED Run): SSC (Bayes):\n")
if(!is.na(single_run_ssc_results$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(single_run_ssc_results$min_coverage, 3), "\n"))
  print(single_run_ssc_results$coverage_by_group)
  write.csv(single_run_ssc_results$coverage_by_group, file.path(TABLES_DIR, "ssc_bayes_BASESEED_RUN.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table for BASE_SEED RUN saved to '", TABLES_DIR, "/ssc_bayes_BASESEED_RUN.csv'\n"))
  
  single_run_plot_filename_ssc <- file.path(PLOTS_DIR, "plot_ssc_bayes_BASESEED_RUN.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_ssc_results$coverage_by_group) > 0){
    png(single_run_plot_filename_ssc, width=800, height=600)
    plot_conditional_coverage(single_run_ssc_results$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Conformalizing Bayes - Iris - BASE_SEED Run)")
    dev.off()
    cat(paste0("INFO: SSC plot for BASE_SEED RUN saved to '", single_run_plot_filename_ssc, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC (BASE_SEED Run - Bayes): NA or no groups.\n") }

cat("INFO: --- End of Conformalizing Bayes Experiment ---\n")