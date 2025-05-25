# R/sec1_run_experiment.R
#
# Purpose:
# This script executes the full experiment for the Basic Conformal Prediction method (described in Section 1 of the reference paper). 
# It uses the Iris dataset and an SVM model. 
# The process includes data loading, splitting, model training, conformal calibration, prediction set generation, and evaluation of coverage
# and set sizes, including conditional coverage metrics (FSC, SSC).

cat("INFO: Sourcing shared R scripts for Experiment (Section 1)...\n")
source("R/load_data.R")
source("R/train_svm_model.R")
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")

cat("INFO: --- Starting Experiment: Basic Conformal Prediction (Section 1) ---\n")

# --- 1. Experiment Settings ---
set.seed(1234) # For overall reproducibility of this script

ALPHA_CONF <- 0.1 # Significance level for conformal prediction (target 1-ALPHA_CONF coverage)
PROP_TRAIN <- 0.6 # Proportion of data for training the underlying SVM model
PROP_CALIB <- 0.2 # Proportion of data for the conformal calibration set
# Reminder (1 - PROP_TRAIN - PROP_CALIB) will be the test set.
cat(paste0("INFO: Settings: ALPHA_CONF=", ALPHA_CONF, ", PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))

# --- 2. Load and Prepare Data ---
iris_data <- load_iris_data() # Data is already shuffled here
n_total <- nrow(iris_data)

# --- 3. Data Splitting (Proper Training, Calibration, Test) ---
n_train <- floor(PROP_TRAIN * n_total)
n_calib <- floor(PROP_CALIB * n_total)
n_test <- n_total - n_train - n_calib

# Create indices for splitting
train_indices <- 1:n_train
calib_indices <- (n_train + 1):(n_train + n_calib)
test_indices <- (n_train + n_calib + 1):n_total

# Perform the split
train_df <- iris_data[train_indices, ]
calib_df <- iris_data[calib_indices, ]
test_df <- iris_data[test_indices, ]

cat(paste0("INFO: Dataset sizes: Train=", nrow(train_df), ", Calib=", nrow(calib_df), ", Test=", nrow(test_df), "\n"))

# --- 4. Train SVM Model ---
svm_formula <- Species ~ . # Using all other columns to predict Species
svm_model <- train_svm_model(svm_formula, train_df)

# --- 5. Calibration ---
cat("INFO: Starting calibration phase for Basic Conformal Prediction...\n")
# Get SVM's predicted probabilities on the calibration set
calib_probs <- predict_svm_probabilities(svm_model, calib_df)
calib_true_labels <- calib_df$Species # True labels for the calibration set

# Calculate basic non-conformity scores for the calibration set
non_conf_scores_calib_basic <- get_non_conformity_scores_basic(calib_probs, calib_true_labels)

# Calculate q_hat using the calibration scores
q_hat_basic <- calculate_q_hat(non_conf_scores_calib_basic, ALPHA_CONF, n_calib = nrow(calib_df))
cat(paste0("INFO: Calibration complete. q_hat (basic) = ", round(q_hat_basic, 4), "\n"))

# --- 6. Prediction and Evaluation on Test Set ---
cat("INFO: Predicting and evaluating Basic Conformal Prediction on test set...\n")
# Get SVM's predicted probabilities on the test set
test_probs <- predict_svm_probabilities(svm_model, test_df)
test_true_labels <- test_df$Species # True labels for the test set

# Create basic conformal prediction sets for the test data
prediction_sets_test_basic <- create_prediction_sets_basic(test_probs, q_hat_basic)
cat("INFO: First few prediction sets (basic):\n"); print(utils::head(prediction_sets_test_basic))

cat("\nINFO: --- Evaluation Results: Basic Method (Section 1) ---\n")
# Calculate and print marginal coverage
empirical_cov_basic <- calculate_empirical_coverage(prediction_sets_test_basic, test_true_labels)
cat(paste0("RESULT: Empirical Marginal Coverage (Basic): ", round(empirical_cov_basic, 3), 
           " (Target >= ", 1 - ALPHA_CONF, ")\n"))

# Calculate and print set size statistics
set_sizes_basic <- get_set_sizes(prediction_sets_test_basic)
cat("RESULT: Set Size Statistics (Basic):\n"); print(summary(set_sizes_basic))
cat(paste0("RESULT: Average set size (Basic): ", round(mean(set_sizes_basic, na.rm=TRUE), 3), "\n"))
# Optional: Plot set size histogram
# plot_set_size_histogram(set_sizes_basic, main_title = "Set Sizes (Basic - Iris - Sec 1)")
# To save:
# dir.create("results/plots", showWarnings = FALSE, recursive = TRUE)
# png("results/plots/sec1_basic_set_sizes_iris.png"); plot_set_size_histogram(set_sizes_basic, "Set Sizes (Basic - Iris)"); dev.off()

# Calculate and print Feature-Stratified Coverage (FSC)
fsc_feature_name_sec1 <- "Sepal.Length" # Example feature for FSC stratification
fsc_results_basic <- calculate_fsc(prediction_sets_test_basic, test_true_labels, 
                                   test_df[[fsc_feature_name_sec1]], feature_name = fsc_feature_name_sec1,
                                   num_bins_for_continuous = 4)
cat(paste0("RESULT: FSC (Basic - by ", fsc_feature_name_sec1, "):\n"))
if(!is.na(fsc_results_basic$min_coverage)) {
  cat(paste0("  Minimum FSC coverage: ", round(fsc_results_basic$min_coverage, 3), "\n"))
  print(fsc_results_basic$coverage_by_group)
  # plot_conditional_coverage(fsc_results_basic$coverage_by_group, "feature_group", "coverage", 
  #                           1 - ALPHA_CONF, paste0("FSC (Basic - ", fsc_feature_name_sec1, " - Iris)"))
} else { cat("  FSC: NA or no groups.\n") }

# Calculate and print Set-Stratified Coverage (SSC)
# Adjust num_bins for SSC based on unique set sizes, max 3 (number of classes) for Iris
ssc_bins <- max(1, min(length(unique(set_sizes_basic)), length(levels(iris_data$Species))))
ssc_results_basic <- calculate_ssc(prediction_sets_test_basic, test_true_labels, num_bins_for_size = ssc_bins)
cat("RESULT: SSC (Basic):\n")
if(!is.na(ssc_results_basic$min_coverage)) {
  cat(paste0("  Minimum SSC coverage: ", round(ssc_results_basic$min_coverage, 3), "\n"))
  print(ssc_results_basic$coverage_by_group)
  # plot_conditional_coverage(ssc_results_basic$coverage_by_group, "size_group", "coverage", 
  #                           1 - ALPHA_CONF, "SSC (Basic - Iris)")
} else { cat("  SSC: NA or no groups.\n") }

cat("INFO: --- End of Basic Conformal Prediction Experiment ---\n")