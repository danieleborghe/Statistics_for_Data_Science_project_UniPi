# R/sec1_run_experiment.R
#
# Purpose:
# This script executes the full experiment for the Basic Conformal Prediction
# method (described in Section 1 of the reference paper). It uses the Iris dataset
# and an SVM model. Results, including tables and plots, are saved.
#
# Functions:
#   - No user-defined functions in this script; it's a main execution script.

cat("INFO: Sourcing shared R scripts for Experiment (Section 1)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")

# Ensure ggplot2 is loaded for saving plots if it's used in evaluation utils
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
}

cat("INFO: --- Starting Experiment: Basic Conformal Prediction (Section 1) ---\n")

# --- 0. Create Results Directories ---
RESULTS_DIR <- "results"
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "section1_basic")
TABLES_DIR <- file.path(RESULTS_DIR, "tables", "section1_basic")
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/'\n"))

# --- 1. Experiment Settings ---
set.seed(1234) 
ALPHA_CONF <- 0.1 
PROP_TRAIN <- 0.6 
PROP_CALIB <- 0.2 
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))

# --- 2. Load and Prepare Data ---
iris_data <- load_iris_data() 
n_total <- nrow(iris_data)

# --- 3. Data Splitting ---
n_train <- floor(PROP_TRAIN * n_total)
n_calib <- floor(PROP_CALIB * n_total)
n_test <- n_total - n_train - n_calib
train_indices <- 1:n_train
calib_indices <- (n_train + 1):(n_train + n_calib)
test_indices <- (n_train + n_calib + 1):n_total
train_df <- iris_data[train_indices, ]
calib_df <- iris_data[calib_indices, ]
test_df <- iris_data[test_indices, ]
cat(paste0("INFO: Dataset sizes: Train=", nrow(train_df), ", Calib=", nrow(calib_df), ", Test=", nrow(test_df), "\n"))

# --- 4. Train SVM Model ---
svm_formula <- Species ~ . 
svm_model <- train_svm_model(svm_formula, train_df)

# --- 5. Calibration ---
cat("INFO: Starting calibration phase for Basic Conformal Prediction...\n")
calib_probs <- predict_svm_probabilities(svm_model, calib_df)
calib_true_labels <- calib_df$Species
non_conf_scores_calib_basic <- get_non_conformity_scores_basic(calib_probs, calib_true_labels)
q_hat_basic <- calculate_q_hat(non_conf_scores_calib_basic, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Calibration complete. q_hat (basic) = ", round(q_hat_basic, 4), "\n"))

# --- 6. Prediction and Evaluation on Test Set ---
cat("INFO: Predicting and evaluating Basic Conformal Prediction on test set...\n")
test_probs <- predict_svm_probabilities(svm_model, test_df)
test_true_labels <- test_df$Species
prediction_sets_test_basic <- create_prediction_sets_basic(test_probs, q_hat_basic)

cat("\nINFO: --- Evaluation Results: Basic Method (Section 1) ---\n")
# Marginal Coverage
empirical_cov_basic <- calculate_empirical_coverage(prediction_sets_test_basic, test_true_labels)
cat(paste0("RESULT: Empirical Marginal Coverage (Basic): ", round(empirical_cov_basic, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
# Save marginal coverage
coverage_summary_basic <- data.frame(
  method = "Basic_Section1",
  alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF,
  empirical_coverage = empirical_cov_basic
)
write.csv(coverage_summary_basic, file.path(TABLES_DIR, "coverage_summary_basic.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary saved to '", TABLES_DIR, "/coverage_summary_basic.csv'\n"))

# Set Sizes
set_sizes_basic <- get_set_sizes(prediction_sets_test_basic)
cat("RESULT: Set Size Statistics (Basic):\n"); print(summary(set_sizes_basic))
avg_set_size_basic <- mean(set_sizes_basic, na.rm=TRUE)
cat(paste0("RESULT: Average set size (Basic): ", round(avg_set_size_basic, 3), "\n"))
# Save set size summary
set_size_summary_df_basic <- data.frame(
  Min = min(set_sizes_basic, na.rm=TRUE),
  Q1 = quantile(set_sizes_basic, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(set_sizes_basic, na.rm=TRUE),
  Mean = avg_set_size_basic,
  Q3 = quantile(set_sizes_basic, 0.75, na.rm=TRUE, names=FALSE),
  Max = max(set_sizes_basic, na.rm=TRUE)
)
write.csv(set_size_summary_df_basic, file.path(TABLES_DIR, "set_size_summary_basic.csv"), row.names = FALSE)
write.csv(data.frame(set_size = set_sizes_basic), file.path(TABLES_DIR, "set_sizes_raw_basic.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes saved to '", TABLES_DIR, "/'\n"))

# Plot and Save set size histogram
plot_filename_set_size_basic <- file.path(PLOTS_DIR, "histogram_set_sizes_basic.png")
png(plot_filename_set_size_basic, width=800, height=600)
plot_set_size_histogram(set_sizes_basic, main_title = "Set Sizes (Basic Conformal - Iris)")
dev.off()
cat(paste0("INFO: Set size histogram saved to '", plot_filename_set_size_basic, "'\n"))

# FSC (Feature-Stratified Coverage)
fsc_feature_name_sec1 <- "Sepal.Length" 
fsc_results_basic <- calculate_fsc(prediction_sets_test_basic, test_true_labels, 
                                   test_df[[fsc_feature_name_sec1]], feature_name = fsc_feature_name_sec1,
                                   num_bins_for_continuous = 4)
cat(paste0("RESULT: FSC (Basic - by ", fsc_feature_name_sec1, "):\n"))
if(!is.na(fsc_results_basic$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(fsc_results_basic$min_coverage, 3), "\n"))
  print(fsc_results_basic$coverage_by_group)
  write.csv(fsc_results_basic$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",fsc_feature_name_sec1,"_basic.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table saved to '", TABLES_DIR, "/fsc_by_",fsc_feature_name_sec1,"_basic.csv'\n"))
  
  plot_filename_fsc_basic <- file.path(PLOTS_DIR, paste0("plot_fsc_",fsc_feature_name_sec1,"_basic.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(fsc_results_basic$coverage_by_group) > 0){
    png(plot_filename_fsc_basic, width=800, height=600)
    plot_conditional_coverage(fsc_results_basic$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Basic - ", fsc_feature_name_sec1, " - Iris)"))
    dev.off()
    cat(paste0("INFO: FSC plot saved to '", plot_filename_fsc_basic, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC: NA or no groups.\n") }

# SSC (Set-Stratified Coverage)
ssc_bins_basic <- max(1, min(length(unique(set_sizes_basic)), length(levels(iris_data$Species))))
ssc_results_basic <- calculate_ssc(prediction_sets_test_basic, test_true_labels, num_bins_for_size = ssc_bins_basic)
cat("RESULT: SSC (Basic):\n")
if(!is.na(ssc_results_basic$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(ssc_results_basic$min_coverage, 3), "\n"))
  print(ssc_results_basic$coverage_by_group)
  write.csv(ssc_results_basic$coverage_by_group, file.path(TABLES_DIR, "ssc_basic.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table saved to '", TABLES_DIR, "/ssc_basic.csv'\n"))
  
  plot_filename_ssc_basic <- file.path(PLOTS_DIR, "plot_ssc_basic.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(ssc_results_basic$coverage_by_group) > 0){
    png(plot_filename_ssc_basic, width=800, height=600)
    plot_conditional_coverage(ssc_results_basic$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Basic - Iris)")
    dev.off()
    cat(paste0("INFO: SSC plot saved to '", plot_filename_ssc_basic, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC: NA or no groups.\n") }

cat("INFO: --- End of Basic Conformal Prediction Experiment ---\n")