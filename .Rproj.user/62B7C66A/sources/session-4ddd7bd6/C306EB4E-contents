# R/sec2_1_run_experiment.R
#
# Purpose:
# This script executes the full experiment for the Adaptive Prediction Sets method
# (described in Section 2.1 of the reference paper). It uses the Iris dataset
# and an SVM model. The process includes data loading, splitting, model training,
# conformal calibration with adaptive scores, prediction set generation, and
# evaluation of coverage and set sizes, including FSC and SSC.

cat("INFO: Sourcing shared R scripts for Experiment (Section 2.1)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")

cat("INFO: --- Starting Experiment: Adaptive Prediction Sets (Section 2.1) ---\n")

# --- 1. Experiment Settings ---
set.seed(5678) # Different seed from Sec 1 or same if comparing directly on same splits
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
# Get SVM's predicted probabilities on the calibration set
calib_probs <- predict_svm_probabilities(svm_model, calib_df)
calib_true_labels <- calib_df$Species

# Calculate adaptive non-conformity scores for the calibration set
non_conf_scores_calib_adaptive <- get_non_conformity_scores_adaptive(calib_probs, calib_true_labels)

# Calculate q_hat using the adaptive scores
q_hat_adaptive <- calculate_q_hat(non_conf_scores_calib_adaptive, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Calibration complete. q_hat (adaptive) = ", round(q_hat_adaptive, 4), "\n"))

# --- 6. Prediction and Evaluation on Test Set ---
cat("INFO: Predicting and evaluating Adaptive Prediction Sets on test set...\n")
# Get SVM's predicted probabilities on the test set
test_probs <- predict_svm_probabilities(svm_model, test_df)
test_true_labels <- test_df$Species

# Create adaptive conformal prediction sets for the test data
prediction_sets_test_adaptive <- create_prediction_sets_adaptive(test_probs, q_hat_adaptive)
cat("INFO: First few prediction sets (adaptive):\n"); print(utils::head(prediction_sets_test_adaptive))

cat("\nINFO: --- Evaluation Results: Adaptive Method (Section 2.1) ---\n")
# Calculate and print marginal coverage
empirical_cov_adaptive <- calculate_empirical_coverage(prediction_sets_test_adaptive, test_true_labels)
cat(paste0("RESULT: Empirical Marginal Coverage (Adaptive): ", round(empirical_cov_adaptive, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))

# Calculate and print set size statistics
set_sizes_adaptive <- get_set_sizes(prediction_sets_test_adaptive)
cat("RESULT: Set Size Statistics (Adaptive):\n"); print(summary(set_sizes_adaptive))
cat(paste0("RESULT: Average set size (Adaptive): ", round(mean(set_sizes_adaptive, na.rm=TRUE), 3), "\n"))
# Optional: Plot set size histogram
# plot_set_size_histogram(set_sizes_adaptive, main_title = "Set Sizes (Adaptive - Iris - Sec 2.1)")
# To save:
# dir.create("results/plots", showWarnings = FALSE, recursive = TRUE)
# png("results/plots/sec2_1_adaptive_set_sizes_iris.png"); plot_set_size_histogram(set_sizes_adaptive, "Set Sizes (Adaptive - Iris)"); dev.off()

# Calculate and print Feature-Stratified Coverage (FSC)
fsc_feature_name_sec2_1 <- "Petal.Width" # Example feature for FSC stratification
fsc_results_adaptive <- calculate_fsc(prediction_sets_test_adaptive, test_true_labels, 
                                      test_df[[fsc_feature_name_sec2_1]], feature_name = fsc_feature_name_sec2_1,
                                      num_bins_for_continuous = 4)
cat(paste0("RESULT: FSC (Adaptive - by ", fsc_feature_name_sec2_1, "):\n"))
if(!is.na(fsc_results_adaptive$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(fsc_results_adaptive$min_coverage, 3), "\n"))
  print(fsc_results_adaptive$coverage_by_group)
  # plot_conditional_coverage(fsc_results_adaptive$coverage_by_group, "feature_group", "coverage", 
  #                           1 - ALPHA_CONF, paste0("FSC (Adaptive - ", fsc_feature_name_sec2_1, " - Iris)"))
} else { cat("  FSC: NA or no groups.\n") }

# Calculate and print Set-Stratified Coverage (SSC)
ssc_bins_adaptive <- max(1, min(length(unique(set_sizes_adaptive)), length(levels(iris_data$Species))))
ssc_results_adaptive <- calculate_ssc(prediction_sets_test_adaptive, test_true_labels, num_bins_for_size = ssc_bins_adaptive)
cat("RESULT: SSC (Adaptive):\n")
if(!is.na(ssc_results_adaptive$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(ssc_results_adaptive$min_coverage, 3), "\n"))
  print(ssc_results_adaptive$coverage_by_group)
  # plot_conditional_coverage(ssc_results_adaptive$coverage_by_group, "size_group", "coverage", 
  #                           1 - ALPHA_CONF, "SSC (Adaptive - Iris)")
} else { cat("  SSC: NA or no groups.\n") }

cat("INFO: --- End of Adaptive Prediction Sets Experiment ---\n")