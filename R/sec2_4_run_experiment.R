# R/sec2_4_run_experiment.R
#
# Purpose:
# This script executes the full experiment for the Conformalizing Bayes method (described in Section 2.4 of the reference paper). 
# It uses the Iris dataset and an SVM model (whose probability outputs are treated as posterior predictive mass).
# Results, including tables and plots, are saved.

cat("INFO: Sourcing shared R scripts for Experiment (Section 2.4 - Conformalizing Bayes)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R") # Make sure this now includes Bayes functions
source("R/evaluation_utils.R")

# Ensure ggplot2 is loaded for saving plots if it's used in evaluation utils
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
}

cat("INFO: --- Starting Experiment: Conformalizing Bayes (Section 2.4) ---\n")

# --- 0. Create Results Directories ---
RESULTS_DIR <- "results"
METHOD_NAME <- "section2_4_bayes"
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME)
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Results for ", METHOD_NAME, " will be saved in '", RESULTS_DIR, "/'\n"))

# --- 1. Experiment Settings ---
set.seed(9012) # Yet another seed, or choose systematically
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
svm_model <- train_svm_model(svm_formula, train_df) # The SVM's probabilities act as f_hat(X)_y

# --- 5. Calibration for Conformalizing Bayes ---
cat("INFO: Starting calibration phase for Conformalizing Bayes...\n")
calib_probs <- predict_svm_probabilities(svm_model, calib_df)
calib_true_labels <- calib_df$Species

# Calculate non-conformity scores using the Bayes method s_i = -P(Y_i|X_i)
non_conf_scores_calib_bayes <- get_non_conformity_scores_bayes(calib_probs, calib_true_labels)

# Calculate q_hat (will likely be negative)
q_hat_bayes <- calculate_q_hat(non_conf_scores_calib_bayes, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Calibration complete. q_hat (Bayes) = ", round(q_hat_bayes, 4), "\n"))
cat(paste0("INFO: Prediction threshold will be -q_hat = ", round(-q_hat_bayes, 4), "\n"))


# --- 6. Prediction and Evaluation on Test Set ---
cat("INFO: Predicting and evaluating Conformalizing Bayes on test set...\n")
test_probs <- predict_svm_probabilities(svm_model, test_df)
test_true_labels <- test_df$Species

# Create prediction sets: T(X) = {y : f_hat(X)_y > -q_hat_bayes}
prediction_sets_test_bayes <- create_prediction_sets_bayes(test_probs, q_hat_bayes)

cat("\nINFO: --- Evaluation Results: Conformalizing Bayes (Section 2.4) ---\n")
# Marginal Coverage
empirical_cov_bayes <- calculate_empirical_coverage(prediction_sets_test_bayes, test_true_labels)
cat(paste0("RESULT: Empirical Marginal Coverage (Bayes): ", round(empirical_cov_bayes, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))
# Save marginal coverage
coverage_summary_bayes <- data.frame(
  method = "ConformalizingBayes_Section2.4",
  alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF,
  empirical_coverage = empirical_cov_bayes,
  q_hat = q_hat_bayes,
  prediction_threshold = -q_hat_bayes
)
write.csv(coverage_summary_bayes, file.path(TABLES_DIR, "coverage_summary_bayes.csv"), row.names = FALSE)
cat(paste0("INFO: Marginal coverage summary saved to '", TABLES_DIR, "/coverage_summary_bayes.csv'\n"))

# Set Sizes
set_sizes_bayes <- get_set_sizes(prediction_sets_test_bayes)
cat("RESULT: Set Size Statistics (Bayes):\n"); print(summary(set_sizes_bayes))
avg_set_size_bayes <- mean(set_sizes_bayes, na.rm=TRUE)
cat(paste0("RESULT: Average set size (Bayes): ", round(avg_set_size_bayes, 3), "\n"))
# Save set size summary
set_size_summary_df_bayes <- data.frame(
  Min = min(set_sizes_bayes, na.rm=TRUE),
  Q1 = quantile(set_sizes_bayes, 0.25, na.rm=TRUE, names=FALSE),
  Median = median(set_sizes_bayes, na.rm=TRUE),
  Mean = avg_set_size_bayes,
  Q3 = quantile(set_sizes_bayes, 0.75, na.rm=TRUE, names=FALSE),
  Max = max(set_sizes_bayes, na.rm=TRUE)
)
write.csv(set_size_summary_df_bayes, file.path(TABLES_DIR, "set_size_summary_bayes.csv"), row.names = FALSE)
write.csv(data.frame(set_size = set_sizes_bayes), file.path(TABLES_DIR, "set_sizes_raw_bayes.csv"), row.names = FALSE)
cat(paste0("INFO: Set size summary and raw sizes saved to '", TABLES_DIR, "/'\n"))

# Plot and Save set size histogram
plot_filename_set_size_bayes <- file.path(PLOTS_DIR, "histogram_set_sizes_bayes.png")
png(plot_filename_set_size_bayes, width=800, height=600)
plot_set_size_histogram(set_sizes_bayes, main_title = "Set Sizes (Conformalizing Bayes - Iris)")
dev.off()
cat(paste0("INFO: Set size histogram saved to '", plot_filename_set_size_bayes, "'\n"))

# FSC (Feature-Stratified Coverage)
fsc_feature_name_bayes <- "Sepal.Width" # Example feature for FSC stratification
fsc_results_bayes <- calculate_fsc(prediction_sets_test_bayes, test_true_labels, 
                                   test_df[[fsc_feature_name_bayes]], feature_name = fsc_feature_name_bayes,
                                   num_bins_for_continuous = 4)
cat(paste0("RESULT: FSC (Bayes - by ", fsc_feature_name_bayes, "):\n"))
if(!is.na(fsc_results_bayes$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(fsc_results_bayes$min_coverage, 3), "\n"))
  print(fsc_results_bayes$coverage_by_group)
  write.csv(fsc_results_bayes$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_",fsc_feature_name_bayes,"_bayes.csv")), row.names = FALSE)
  cat(paste0("INFO: FSC results table saved to '", TABLES_DIR, "/fsc_by_",fsc_feature_name_bayes,"_bayes.csv'\n"))
  
  plot_filename_fsc_bayes <- file.path(PLOTS_DIR, paste0("plot_fsc_",fsc_feature_name_bayes,"_bayes.png"))
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(fsc_results_bayes$coverage_by_group) > 0){
    png(plot_filename_fsc_bayes, width=800, height=600)
    plot_conditional_coverage(fsc_results_bayes$coverage_by_group, "feature_group", "coverage", 
                              1 - ALPHA_CONF, paste0("FSC (Bayes - ", fsc_feature_name_bayes, " - Iris)"))
    dev.off()
    cat(paste0("INFO: FSC plot saved to '", plot_filename_fsc_bayes, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for FSC plot.\n")}
} else { cat("  FSC: NA or no groups.\n") }

# SSC (Set-Stratified Coverage)
ssc_bins_bayes <- max(1, min(length(unique(set_sizes_bayes)), length(levels(iris_data$Species))))
ssc_results_bayes <- calculate_ssc(prediction_sets_test_bayes, test_true_labels, num_bins_for_size = ssc_bins_bayes)
cat("RESULT: SSC (Bayes):\n")
if(!is.na(ssc_results_bayes$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(ssc_results_bayes$min_coverage, 3), "\n"))
  print(ssc_results_bayes$coverage_by_group)
  write.csv(ssc_results_bayes$coverage_by_group, file.path(TABLES_DIR, "ssc_bayes.csv"), row.names = FALSE)
  cat(paste0("INFO: SSC results table saved to '", TABLES_DIR, "/ssc_bayes.csv'\n"))
  
  plot_filename_ssc_bayes <- file.path(PLOTS_DIR, "plot_ssc_bayes.png")
  if(requireNamespace("ggplot2", quietly = TRUE) && nrow(ssc_results_bayes$coverage_by_group) > 0){
    png(plot_filename_ssc_bayes, width=800, height=600)
    plot_conditional_coverage(ssc_results_bayes$coverage_by_group, "size_group", "coverage", 
                              1 - ALPHA_CONF, "SSC (Conformalizing Bayes - Iris)")
    dev.off()
    cat(paste0("INFO: SSC plot saved to '", plot_filename_ssc_bayes, "'\n"))
  } else {cat("INFO: ggplot2 not available or no data for SSC plot.\n")}
} else { cat("  SSC: NA or no groups.\n") }

cat("INFO: --- End of Conformalizing Bayes Experiment ---\n")