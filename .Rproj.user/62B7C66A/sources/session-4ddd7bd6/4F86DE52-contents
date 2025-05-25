# R/train_svm_model.R
#
# Purpose:
# This script provides functions for training a Support Vector Machine (SVM) model and for obtaining class probability predictions from the trained model.
# It utilizes the 'e1071' package.
#
# Functions:
#   - train_svm_model(formula, training_data): Trains an SVM model.
#   - predict_svm_probabilities(svm_model, newdata): Predicts class probabilities.

# Ensure e1071 is available
if (!requireNamespace("e1071", quietly = TRUE)) {
  stop("Package 'e1071' is required but not installed. Please run install_dependencies.R")
}
library(e1071) # Load the package for e1071::svm

train_svm_model <- function(formula, training_data) {
  # Purpose: Trains an SVM model using the specified formula and training data.
  #          Enables probability estimation in the SVM model.
  # Parameters:
  #   - formula: An R formula specifying the model (e.g., Species ~ .).
  #   - training_data: A data frame containing the training data.
  # Returns: A trained SVM model object.
  
  cat("INFO: Training SVM model...\n")
  target_var_name <- all.vars(formula)[1] # Get the name of the target variable from formula
  # Check if the target variable is a factor, which is required for classification
  if (!is.factor(training_data[[target_var_name]])) {
    stop(paste("ERROR: Target variable '", target_var_name, "' must be a factor for SVM classification.", sep=""))
  }
  
  # Train the SVM model with probability estimation enabled
  model <- e1071::svm(formula, data = training_data, kernel = "radial", probability = TRUE, scale = TRUE)
  cat("INFO: SVM model training complete.\n")
  return(model)
}

predict_svm_probabilities <- function(svm_model, newdata) {
  # Purpose: Predicts class probabilities for new data using a trained SVM model.
  # Parameters:
  #   - svm_model: A trained SVM model object (from train_svm_model).
  #   - newdata: A data frame containing the new data for which to predict probabilities.
  # Returns: A matrix of class probabilities, where rows are samples and columns are classes.
  
  # Predict using the SVM model; probabilities are stored as an attribute
  predictions_object <- predict(svm_model, newdata, probability = TRUE)
  probabilities <- attr(predictions_object, "probabilities")
  
  # Error handling if probabilities could not be retrieved
  if (is.null(probabilities)) {
    stop("ERROR: Could not retrieve probabilities from SVM prediction. Was the model trained with probability=TRUE?")
  }
  return(probabilities)
}