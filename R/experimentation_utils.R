# R/experimentation_utils.R
#
# Purpose:
# This script contains utility functions that can be reused across
# different parts of the R_Conformal_Iris_SVM project.
#
# Functions:
#   - check_and_load_packages(...): Checks, installs, and loads required packages.
#   - load_iris_for_classification(...): Loads and prepares Iris dataset for classification.
#   - load_iris_for_regression(...): Loads and prepares Iris dataset for regression.
#   - train_svm_model(...): Trains an SVM model for classification.
#   - predict_svm_probabilities(...): Predicts class probabilities from a trained SVM model.
#   - save_detailed_test_predictions(...): Saves a detailed CSV of test predictions.

check_and_load_packages <- function(required_packages) {
  # Purpose: Checks if a list of packages is installed, installs any that are missing,
  #          and then loads them into the current session.
  # Parameters:
  #   - required_packages: A character vector of package names.
  # Returns: None. Prints messages to the console and loads packages.
  
  cat("INFO: Starting package dependency check and load...\n")
  
  for (pkg in required_packages) {
    # Check if the package is installed
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(paste("INFO: Package '", pkg, "' not found. Attempting to install...\n", sep = ""))
      install.packages(pkg, dependencies = TRUE)
    }
    
    # Load the package
    # Using tryCatch to handle potential loading errors gracefully
    tryCatch({
      library(pkg, character.only = TRUE)
      cat(paste("INFO: Package '", pkg, "' loaded successfully.\n", sep = ""))
    }, error = function(e) {
      warning(paste("WARN: Failed to load package '", pkg, "'. Please check your installation.\nError: ", e$message, sep = ""))
    })
  }
  cat("INFO: Package dependency check and load complete.\n")
}


load_iris_for_classification <- function() {
  # Purpose: Loads the built-in Iris dataset, ensures 'Species' is a factor, and shuffles the dataset.
  # Parameters: None
  # Returns: A data frame containing the shuffled Iris dataset for classification.
  
  cat("INFO: Loading Iris dataset for classification...\n")
  data(iris) # Load the Iris dataset from R's built-in datasets
  
  # Ensure 'Species' column is treated as a factor
  iris$Species <- as.factor(iris$Species)
  
  # Shuffle the dataset for random sampling in later stages
  set.seed(123) # Set seed for reproducible shuffling
  iris_shuffled <- iris[sample(nrow(iris)), ]
  
  cat("INFO: Iris dataset loaded, 'Species' converted to factor, and data shuffled.\n")
  return(iris_shuffled)
}


load_iris_for_regression <- function() {
  # Purpose: Loads the Iris dataset and prepares it for a regression task by removing
  #          the 'Species' column and shuffling the rows.
  # Parameters: None.
  # Returns: A shuffled data frame of the Iris dataset suitable for regression.
  
  cat("INFO: Loading Iris dataset for regression...\n")
  data(iris)
  iris_regression <- iris[, -which(names(iris) == "Species")]
  
  set.seed(123)
  shuffled_iris <- iris_regression[sample(nrow(iris_regression)), ]
  
  cat("INFO: Loaded and shuffled Iris dataset for regression task.\n")
  return(shuffled_iris)
}


train_svm_model <- function(formula, training_data) {
  # Purpose: Trains an SVM model for classification using the specified formula and training data,
  #          with probability estimation enabled.
  # Parameters:
  #   - formula: An R formula specifying the model (e.g., Species ~ .).
  #   - training_data: A data frame containing the training data.
  # Returns: A trained SVM model object from the 'e1071' package.
  
  cat("INFO: Training SVM model for classification...\n")
  target_var_name <- all.vars(formula)[1]
  
  if (!is.factor(training_data[[target_var_name]])) {
    stop(paste("ERROR: Target variable '", target_var_name, "' must be a factor for SVM classification.", sep=""))
  }
  
  model <- e1071::svm(formula, data = training_data, kernel = "radial", probability = TRUE, scale = TRUE)
  cat("INFO: SVM model training complete.\n")
  return(model)
}

predict_svm_probabilities <- function(svm_model, newdata) {
  # Purpose: Predicts class probabilities for new data using a trained classification SVM model.
  # Parameters:
  #   - svm_model: A trained SVM model object from train_svm_model().
  #   - newdata: A data frame for which to predict probabilities.
  # Returns: A matrix of class probabilities, where rows are samples and columns are classes.
  
  predictions_object <- predict(svm_model, newdata, probability = TRUE)
  probabilities <- attr(predictions_object, "probabilities")
  
  if (is.null(probabilities)) {
    stop("ERROR: Could not retrieve probabilities from SVM prediction. Was the model trained with probability=TRUE?")
  }
  return(probabilities)
}


save_detailed_test_predictions <- function(test_true_labels,
                                           prediction_sets_list,
                                           test_probs_matrix,
                                           output_directory,
                                           base_filename) {
  # Purpose: Creates and saves a CSV file with detailed information for each test sample.
  # Parameters:
  #   - test_true_labels: Vector of true class labels for the test set.
  #   - prediction_sets_list: List where each element is a character vector (prediction set).
  #   - test_probs_matrix: Matrix of predicted class probabilities for the test set.
  #   - output_directory: The directory path where the CSV file will be saved.
  #   - base_filename: The base name for the CSV file.
  # Returns: Invisibly returns the full path to the saved file, or NULL if saving failed.
  
  cat(paste0("INFO: Preparing to save detailed test predictions to '", base_filename, "'...\n"))
  
  if (length(test_true_labels) != length(prediction_sets_list) ||
      length(test_true_labels) != nrow(test_probs_matrix)) {
    stop("ERROR: Input arguments to save_detailed_test_predictions have mismatched lengths/rows.")
  }
  
  pred_set_strings <- sapply(prediction_sets_list, function(s) {
    if (length(s) == 0) "" else paste(sort(s), collapse = "; ")
  })
  
  detailed_results_df <- as.data.frame(test_probs_matrix)
  names(detailed_results_df) <- paste0("Prob_", names(detailed_results_df))
  detailed_results_df <- cbind(
    data.frame(
      SampleID = 1:length(test_true_labels),
      TrueLabel = as.character(test_true_labels),
      PredictionSet = pred_set_strings
    ),
    detailed_results_df
  )
  
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  detailed_filename_path <- file.path(output_directory, base_filename)
  
  tryCatch({
    write.csv(detailed_results_df, detailed_filename_path, row.names = FALSE)
    cat(paste0("INFO: Detailed test predictions saved to '", detailed_filename_path, "'\n"))
    return(invisible(detailed_filename_path))
  }, error = function(e) {
    warning(paste0("WARN: Failed to save detailed test predictions. Error: ", e$message))
    return(invisible(NULL))
  })
}
