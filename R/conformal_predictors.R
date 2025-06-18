# R/conformal_predictors.R
#
# Purpose:
# This script implements the core logic for different Conformal Prediction procedures as described in the paper.
# It includes functions for calculating non-conformity scores and for creating prediction sets/intervals.
#
# Functions:
#   - calculate_q_hat(...): Calculates the q_hat quantile (shared by all methods).
#   - get_non_conformity_scores_basic(...): Calculates basic classification scores.
#   - create_prediction_sets_basic(...): Creates basic classification sets.
#   - get_non_conformity_scores_adaptive(...): Calculates adaptive classification scores.
#   - create_prediction_sets_adaptive(...): Creates adaptive classification sets.
#   - get_non_conformity_scores_bayes(...): Calculates Bayes classification scores.
#   - create_prediction_sets_bayes(...): Creates Bayes classification sets.
#   - train_quantile_models(...): Trains quantile regression models.
#   - get_non_conformity_scores_quantile(...): Calculates quantile regression scores.
#   - create_prediction_intervals_quantile(...): Creates quantile regression intervals.
#   - train_primary_and_uncertainty_models(...): Trains primary and uncertainty regression models.
#   - get_non_conformity_scores_scalar(...): Calculates scalar uncertainty regression scores.
#   - create_prediction_intervals_scalar(...): Creates scalar uncertainty regression intervals.

# --- Utility Function (Shared) ---

calculate_q_hat <- function(non_conformity_scores, alpha, n_calib) {
  # Purpose: Calculates the q_hat value (the (1-alpha) empirical quantile of non-conformity scores, adjusted for finite sample size).
  # Parameters:
  #   - non_conformity_scores: A numeric vector of scores from the calibration set.
  #   - alpha: The desired significance level (e.g., 0.1 for 90% coverage).
  #   - n_calib: The number of samples in the calibration set.
  # Returns: The calculated q_hat value.
  
  # Calculate the probability level for R's quantile function, as per paper's formula
  prob_level <- ceiling((n_calib + 1) * (1 - alpha)) / n_calib
  # Ensure prob_level is within [0, 1]
  prob_level <- min(1, max(0, prob_level))
  
  # Calculate q_hat using type=1 for the inverse of the ECDF
  q_hat <- quantile(non_conformity_scores, probs = prob_level, type = 1, names = FALSE)
  
  return(q_hat)
}


# --- Section 1: Basic Conformal Prediction (Classification) ---

get_non_conformity_scores_basic <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates basic non-conformity scores (1 - probability of true class).
  # Parameters:
  #   - predicted_probs_matrix: A matrix of predicted probabilities (rows=samples, cols=classes).
  #   - true_labels: A vector of the true class labels for the samples.
  # Returns: A numeric vector of non-conformity scores.
  
  scores <- 1 - predicted_probs_matrix[cbind(1:nrow(predicted_probs_matrix), match(true_labels, colnames(predicted_probs_matrix)))]
  return(scores)
}

create_prediction_sets_basic <- function(predicted_probs_matrix_new_data, q_hat) {
  # Purpose: Creates basic prediction sets based on the rule P(Y=y|X) >= 1 - q_hat.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of probabilities for new data.
  #   - q_hat: The calibrated q_hat threshold.
  # Returns: A list of character vectors, where each vector is a prediction set.
  
  threshold <- 1 - q_hat
  prediction_sets <- apply(predicted_probs_matrix_new_data, 1, function(row) {
    names(row)[row >= threshold]
  })
  return(prediction_sets)
}


# --- Section 2.1: Adaptive Prediction Sets (Classification) ---

get_non_conformity_scores_adaptive <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates adaptive non-conformity scores. The score is the sum of probabilities of classes,
  #          sorted by probability, up to and including the true class.
  # Parameters:
  #   - predicted_probs_matrix: Matrix of predicted probabilities.
  #   - true_labels: Vector of true class labels.
  # Returns: A numeric vector of adaptive non-conformity scores.
  
  scores <- numeric(length(true_labels))
  for (i in 1:length(true_labels)) {
    probs_i <- predicted_probs_matrix[i, ]
    true_class_name_i <- as.character(true_labels[i])
    sorted_indices <- order(probs_i, decreasing = TRUE)
    rank_true_class <- which(names(probs_i)[sorted_indices] == true_class_name_i)
    scores[i] <- sum(sort(probs_i, decreasing = TRUE)[1:rank_true_class])
  }
  return(scores)
}

create_prediction_sets_adaptive <- function(predicted_probs_matrix_new_data, q_hat) {
  # Purpose: Creates adaptive prediction sets. Classes are added to the set in order
  #          of decreasing probability until their cumulative probability sum meets or exceeds q_hat.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of probabilities for new data.
  #   - q_hat: The calibrated q_hat threshold from adaptive scores.
  # Returns: A list of character vectors, representing adaptive prediction sets.
  
  prediction_sets <- vector("list", nrow(predicted_probs_matrix_new_data))
  for (i in 1:nrow(predicted_probs_matrix_new_data)) {
    probs_i <- predicted_probs_matrix_new_data[i, ]
    sorted_probs <- sort(probs_i, decreasing = TRUE)
    cumulative_probs <- cumsum(sorted_probs)
    num_classes_to_include <- which.max(cumulative_probs >= q_hat)
    prediction_sets[[i]] <- names(sorted_probs)[1:num_classes_to_include]
  }
  return(prediction_sets)
}


# --- Section 2.4: Conformalizing Bayes (Classification) ---

get_non_conformity_scores_bayes <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates non-conformity scores for Conformalizing Bayes.
  #          The score is the negative of the model's predicted probability for the true class.
  # Parameters:
  #   - predicted_probs_matrix: A matrix of predicted probabilities.
  #   - true_labels: A vector of true class labels.
  # Returns: A numeric vector of non-conformity scores.
  
  scores <- -predicted_probs_matrix[cbind(1:nrow(predicted_probs_matrix), match(true_labels, colnames(predicted_probs_matrix)))]
  return(scores)
}

create_prediction_sets_bayes <- function(predicted_probs_matrix_new_data, q_hat_bayes) {
  # Purpose: Creates prediction sets for Conformalizing Bayes based on the rule P(y|X) > -q_hat.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of probabilities for new data.
  #   - q_hat_bayes: The calibrated q_hat threshold (typically negative).
  # Returns: A list of character vectors, where each vector is a prediction set.
  
  threshold <- -q_hat_bayes
  prediction_sets <- apply(predicted_probs_matrix_new_data, 1, function(row) {
    names(row)[row > threshold]
  })
  return(prediction_sets)
}


# --- Section 2.2: Conformalized Quantile Regression (Regression) ---

train_quantile_models <- function(formula, training_data, alpha) {
  # Purpose: Trains two quantile regression models for the lower and upper bounds.
  # Parameters:
  #   - formula: The regression formula (e.g., Y ~ .).
  #   - training_data: The data frame for training.
  #   - alpha: The significance level (e.g., 0.1).
  # Returns: A list containing the trained 'lower_model' and 'upper_model'.
  
  if (!requireNamespace("quantreg", quietly = TRUE)) stop("Package 'quantreg' is required.")
  tau_lower <- alpha / 2
  tau_upper <- 1 - (alpha / 2)
  
  model_lower <- quantreg::rq(formula, data = training_data, tau = tau_lower)
  model_upper <- quantreg::rq(formula, data = training_data, tau = tau_upper)
  
  return(list(lower_model = model_lower, upper_model = model_upper))
}

get_non_conformity_scores_quantile <- function(models, calib_data, target_variable_name) {
  # Purpose: Computes non-conformity scores for quantile regression.
  #          The score is max(lower_bound(X) - Y, Y - upper_bound(X)).
  # Parameters:
  #   - models: A list containing the trained 'lower_model' and 'upper_model'.
  #   - calib_data: The calibration data frame.
  #   - target_variable_name: String name of the target variable column.
  # Returns: A numeric vector of non-conformity scores.
  
  y_true <- calib_data[[target_variable_name]]
  q_hat_lower <- predict(models$lower_model, newdata = calib_data)
  q_hat_upper <- predict(models$upper_model, newdata = calib_data)
  
  scores <- pmax(q_hat_lower - y_true, y_true - q_hat_upper)
  return(scores)
}

create_prediction_intervals_quantile <- function(models, new_data, q_hat) {
  # Purpose: Creates conformal prediction intervals for new data using quantile regression.
  #          The interval is [lower_bound(X) - q_hat, upper_bound(X) + q_hat].
  # Parameters:
  #   - models: A list with the trained 'lower_model' and 'upper_model'.
  #   - new_data: The data for which to create intervals.
  #   - q_hat: The calibrated quantile from the calibration step.
  # Returns: A data frame with 'lower_bound' and 'upper_bound'.
  
  q_hat_lower <- predict(models$lower_model, newdata = new_data)
  q_hat_upper <- predict(models$upper_model, newdata = new_data)
  
  final_lower <- q_hat_lower - q_hat
  final_upper <- q_hat_upper + q_hat
  
  return(data.frame(lower_bound = final_lower, upper_bound = final_upper))
}


# --- Section 2.3: Conformalizing Scalar Uncertainty (Regression) ---

train_primary_and_uncertainty_models <- function(formula, training_data, target_variable_name) {
  # Purpose: Trains the primary prediction model (f_hat) and the uncertainty model (u_hat).
  #          The uncertainty model learns to predict the absolute residuals of the primary model.
  # Parameters:
  #   - formula: The regression formula for the primary model.
  #   - training_data: The data frame for training.
  #   - target_variable_name: String name of the target variable.
  # Returns: A list containing the trained 'f_model' (primary) and 'u_model' (uncertainty).
  
  if (!requireNamespace("e1071", quietly = TRUE)) stop("Package 'e1071' is required.")
  
  # Train primary model
  f_model <- e1071::svm(formula, data = training_data)
  
  # Calculate residuals on training data to create targets for the uncertainty model
  train_preds <- predict(f_model, newdata = training_data)
  train_abs_residuals <- abs(training_data[[target_variable_name]] - train_preds)
  
  u_train_data <- training_data
  u_train_data$abs_residual <- train_abs_residuals
  
  # Train uncertainty model
  u_formula_str <- paste("abs_residual ~ . -", target_variable_name)
  u_model <- e1071::svm(as.formula(u_formula_str), data = u_train_data)
  
  return(list(f_model = f_model, u_model = u_model))
}

get_non_conformity_scores_scalar <- function(models, calib_data, target_variable_name) {
  # Purpose: Computes non-conformity scores for the scalar uncertainty method.
  #          The score is |y - f_hat(x)| / u_hat(x).
  # Parameters:
  #   - models: A list containing the trained 'f_model' and 'u_model'.
  #   - calib_data: The calibration data frame.
  #   - target_variable_name: String name of the target variable.
  # Returns: A numeric vector of non-conformity scores.
  
  calib_f_preds <- predict(models$f_model, newdata = calib_data)
  calib_u_preds <- predict(models$u_model, newdata = calib_data)
  calib_abs_residuals <- abs(calib_data[[target_variable_name]] - calib_f_preds)
  
  scores <- calib_abs_residuals / calib_u_preds
  return(scores)
}

create_prediction_intervals_scalar <- function(models, new_data, q_hat) {
  # Purpose: Creates conformal prediction intervals using the scalar uncertainty method.
  #          The interval is [f_hat(x) - q_hat * u_hat(x), f_hat(x) + q_hat * u_hat(x)].
  # Parameters:
  #   - models: A list containing the trained 'f_model' and 'u_model'.
  #   - new_data: The data for which to create intervals.
  #   - q_hat: The calibrated quantile from the calibration step.
  # Returns: A data frame with 'lower_bound' and 'upper_bound'.
  
  test_f_preds <- predict(models$f_model, newdata = new_data)
  test_u_preds <- predict(models$u_model, newdata = new_data)
  
  lower_bound <- test_f_preds - q_hat * test_u_preds
  upper_bound <- test_f_preds + q_hat * test_u_preds
  
  return(data.frame(lower_bound = lower_bound, upper_bound = upper_bound))
}
