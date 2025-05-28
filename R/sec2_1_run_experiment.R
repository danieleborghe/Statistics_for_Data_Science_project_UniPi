# R/sec2_1_run_experiment.R
#
# Purpose:
# This script executes the full experiment for the Adaptive Prediction Sets method
# (described in Section 2.1 of the reference paper). It uses the Iris dataset
# and an SVM model. Results, including tables and plots, are saved.
#
# Functions:
#   - No user-defined functions in this script; it's a main execution script.

cat("INFO: Sourcing shared R scripts for Experiment (Section 2.1)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")

# Ensure ggplot2 is loaded for saving plots if it's used in evaluation utils
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
}

cat("INFO: --- Starting Experiment: Adaptive Prediction Sets (Section 2.1) ---\n")

# --- 0. Create Results Directories ---
RESULTS_DIR <- "results"
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "section2_1_adaptive")
TABLES_DIR <- file.path(RESULTS_DIR, "tables", "section2_1_adaptive")
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results will be saved in '", RESULTS_DIR, "/'\n"))

# --- 1. Experiment Settings ---
set.seed(5678) 
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
cat("INFO: Starting calibration phase for Adaptive Prediction Sets...\n")
calib_probs <- predict_svm_probabilities(svm_model, calib_df)
calib_true_labels <- calib_df$Species
non_conf_scores_calib_adaptive <- get_non_conformity_scores_adaptive(calib_probs, calib_true_labels)
q_hat_adaptive <- calculate_q_hat(non_conf_scores_calib_adaptive, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Calibration complete. q_hat (adaptive) = ", round(q_hat_adaptive, 4), "\n"))

# --- 6. Prediction and Evaluation on Test Set ---
cat("INFO: Predicting and evaluating Adaptive Prediction Sets on test set...\n")
test_probs <- predict_svm_probabilities(svm_model, test_df)
test_true_labels <- test_df$Species
prediction_sets_test_adaptive <- create_prediction_sets_adaptive(test_probs, q_hat_adaptive)

cat("\nINFO: --- Evaluation Results: Adaptive Method (Section 2.1) ---\n")
# Marginal Coverage
empirical_cov_adaptive <- calculate_empirical_coverage(prediction_sets_test_adaptive, test_true_labels)
cat(paste0("RESULT: Empirical Marginal Coverage (Adaptive): ", round(empirical_cov_adaptive, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
# Save marginal coverage
coverage_summary_adaptive <- data.frame(
  method = "Adaptive_Section2.1",
  alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF,
  empirical_coverage = empirical_cov_adaptive
)
write.csv(coverage_summary_adaptive, file.path(TABLES_DIR, "coverage_summary_adaptive.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary saved to '", TABLES_DIR, "/coverage_summary_adaptive.csv'\n"))


# Set Sizes
set_sizes_adaptive <- get_set_sizes(prediction_sets_test_adaptive)
cat("RESULT: Set Size Statistics (Adaptive):\n"); print(summary(set_sizes_adaptive))
avg_set_size_adaptive <- mean(set_sizes_adaptive, na.rm=TRUE)
cat(paste0("RESULT: Average set size (Adaptive): ", round(avg_set_size_adaptive, 3), "\n"))
# Save set size summary
set_size_summary_df_adaptive <- data.frame(
  Min = min(set_sizes_adaptive, na.rm=TRUE),
  Q1 = quantile(set_sizes_adaptive, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(set_sizes_adaptive, na.rm=TRUE),
  Mean = avg_set_size_adaptive,
  Q3 = quantile(set_sizes_adaptive, 0.75, na.rm=TRUE, names=FALSE),
  Max = max(set_sizes_adaptive, na.rm=TRUE)
)
write.csv(set_size_summary_df_adaptive, file.path(TABLES_DIR, "set_size_summary_adaptive.csv"), row.names = FALSE)
write.csv(data.frame(set_size = set_sizes_adaptive), file.path(TABLES_DIR, "set_sizes_raw_adaptive.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes saved to '", TABLES_DIR, "/'\n"))

# Plot and Save set size histogram
plot_filename_set_size_adaptive <- file.path(PLOTS_DIR, "histogram_set_sizes_adaptive.png")
png(plot_filename_set_size_adaptive, width=800, height=600)
plot_set_size_histogram(set_sizes_adaptive, main_title = "Set Sizes (Adaptive Conformal - Iris)")
dev.off()
cat(paste0("INFO: Set size histogram saved to '", plot_filename_set_size_adaptive, "'\n"))

# FSC (Feature-Stratified Coverage)
fsc_feature_name_sec2_1 <- "Petal.Width" 
fsc_results_adaptive <- calculate_fsc(prediction_sets_test_adaptive, test_true_labels, 
                                      test_df[[fsc_feature_name_sec2_1]], feature_name = fsc_feature_name_sec2_1,
                                      num_bins_for_continuous = 4)
cat(paste0("RESULT: FSC (Adaptive - by ", fsc_feature_name_sec2_1, "):\n"))
if(!is.na(fsc_results_adaptive$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(fsc_results_adaptive$min_coverage, 3), "\n"))
  print(fsc_results_adaptive$coverage_by_group)
  write.csv(fsc_results_adaptive$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",fsc_feature_name_sec2_1,"_adaptive.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table saved to '", TABLES_DIR, "/fsc_by_",fsc_feature_name_sec2_1,"_adaptive.csv'\n"))
  
  plot_filename_fsc_adaptive <- file.path(PLOTS_DIR, paste0("plot_fsc_",fsc_feature_name_sec2_1,"_adaptive.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(fsc_results_adaptive$coverage_by_group) > 0){
    png(plot_filename_fsc_adaptive, width=800, height=600)
    plot_conditional_coverage(fsc_results_adaptive$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Adaptive - ", fsc_feature_name_sec2_1, " - Iris)"))
    dev.off()
    cat(paste0("INFO: FSC plot saved to '", plot_filename_fsc_adaptive, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC: NA or no groups.\n") }

# SSC (Set-Stratified Coverage)
ssc_bins_adaptive <- max(1, min(length(unique(set_sizes_adaptive)), length(levels(iris_data$Species))))
ssc_results_adaptive <- calculate_ssc(prediction_sets_test_adaptive, test_true_labels, num_bins_for_size = ssc_bins_adaptive)
cat("RESULT: SSC (Adaptive):\n")
if(!is.na(ssc_results_adaptive$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(ssc_results_adaptive$min_coverage, 3), "\n"))
  print(ssc_results_adaptive$coverage_by_group)
  write.csv(ssc_results_adaptive$coverage_by_group, file.path(TABLES_DIR, "ssc_adaptive.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table saved to '", TABLES_DIR, "/ssc_adaptive.csv'\n"))
  
  plot_filename_ssc_adaptive <- file.path(PLOTS_DIR, "plot_ssc_adaptive.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(ssc_results_adaptive$coverage_by_group) > 0){
    png(plot_filename_ssc_adaptive, width=800, height=600)
    plot_conditional_coverage(ssc_results_adaptive$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Adaptive - Iris)")
    dev.off()
    cat(paste0("INFO: SSC plot saved to '", plot_filename_ssc_adaptive, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC: NA or no groups.\n") }

cat("INFO: --- End of Adaptive Prediction Sets Experiment ---\n")