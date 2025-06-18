# R/conformal_predictors.R
#
# Purpose:
# This script implements the core logic for different Conformal Prediction procedures as described in the paper. 
# It includes functions for calculating non-conformity scores and for creating prediction sets.
# Functions:
  #   - calculate_q_hat(non_conformity_scores, alpha, n_calib): Calculates the q_hat quantile.
  #   - get_non_conformity_scores_basic(...): Calculates basic non-conformity scores.
  #   - create_prediction_sets_basic(...): Creates basic prediction sets.
  #   - get_non_conformity_scores_adaptive(...): Calculates adaptive non-conformity scores.
  #   - create_prediction_sets_adaptive(...): Creates adaptive prediction sets.
  #   - get_non_conformity_scores_bayes(...): Calculates non-conformity scores for Conformalizing Bayes.
  #   - create_prediction_sets_bayes(...): Creates prediction sets for Conformalizing Bayes.

calculate_q_hat <- function(non_conformity_scores, alpha, n_calib) {
  # Purpose: Calculates the q_hat value (the (1-alpha) empirical quantile of non-conformity scores, adjusted for finite sample size).
  # Parameters:
  #   - non_conformity_scores: A numeric vector of non-conformity scores from the calibration set.
  #   - alpha: The desired significance level (e.g., 0.1 for 90% coverage).
  #   - n_calib: The number of samples in the calibration set.
  # Returns: The calculated q_hat value.
  
  # Calculate the probability level for R's quantile function, as per paper's formula
  prob_level <- (ceiling((n_calib + 1) * (1 - alpha))) / n_calib
  # Ensure prob_level is within [0, 1]
  if (prob_level > 1) prob_level <- 1
  if (prob_level < 0) prob_level <- 0 
  
  # Calculate q_hat using type=1 for inverse of ECDF
  q_hat <- quantile(non_conformity_scores, probs = prob_level, type = 1, names = FALSE)
  
  return(q_hat)
}

# --- Section 1: Basic Conformal Prediction ---
get_non_conformity_scores_basic <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates basic non-conformity scores (1 - probability of true class).
  # Parameters:
  #   - predicted_probs_matrix: A matrix of predicted probabilities (rows=samples, cols=classes).
  #                             Colnames should be the class labels.
  #   - true_labels: A vector of true class labels (factor or character).
  # Returns: A numeric vector of non-conformity scores.
  
  scores <- numeric(length(true_labels))
  class_names <- colnames(predicted_probs_matrix) # Get class names from matrix column names
  
  for (i in 1:length(true_labels)) {
    true_class_name <- as.character(true_labels[i])
    # Get the predicted probability for the true class
    prob_true_class <- predicted_probs_matrix[i, true_class_name]
    # Calculate non-conformity score
    scores[i] <- 1 - prob_true_class
  }
  return(scores)
}

create_prediction_sets_basic <- function(predicted_probs_matrix_new_data, q_hat) {
  # Purpose: Creates basic prediction sets based on the rule P(Y=y|X) >= 1 - q_hat.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of predicted probabilities for new data.
  #   - q_hat: The calibrated q_hat threshold.
  # Returns: A list of character vectors, where each vector is a prediction set for a sample.
  
  prediction_sets <- vector("list", nrow(predicted_probs_matrix_new_data))
  class_names <- colnames(predicted_probs_matrix_new_data)
  threshold <- 1 - q_hat # Define the inclusion threshold
  
  for (i in 1:nrow(predicted_probs_matrix_new_data)) {
    # Select classes whose predicted probability meets or exceeds the threshold
    current_set <- class_names[predicted_probs_matrix_new_data[i, ] >= threshold]
    # Store the set (can be empty if no class meets the threshold)
    prediction_sets[[i]] <- if (length(current_set) == 0) character(0) else current_set
  }
  return(prediction_sets)
}

# --- Section 2.1: Adaptive Prediction Sets ---
get_non_conformity_scores_adaptive <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates adaptive non-conformity scores. The score is the sum of probabilities of classes, sorted by probability, up to and including the true class.
  # Parameters:
  #   - predicted_probs_matrix: Matrix of predicted probabilities.
  #   - true_labels: Vector of true class labels.
  # Returns: A numeric vector of adaptive non-conformity scores.
  
  scores <- numeric(length(true_labels))
  
  for (i in 1:length(true_labels)) {
    probs_i <- predicted_probs_matrix[i, ] # Probabilities for the i-th sample
    true_class_name_i <- as.character(true_labels[i])
    
    # Sort probabilities in decreasing order
    sorted_indices <- order(probs_i, decreasing = TRUE)
    sorted_probs <- probs_i[sorted_indices]
    
    # Get sorted class names, ensuring consistency
    current_colnames <- colnames(predicted_probs_matrix)
    if(is.null(current_colnames) && !is.null(names(probs_i))) current_colnames <- names(probs_i)
    if(is.null(current_colnames)) stop("ERROR: Could not determine class names from predicted_probs_matrix columns or names of probs_i.")
    sorted_class_names <- current_colnames[sorted_indices]
    
    # Find the rank (position) of the true class in the sorted list
    rank_true_class <- which(sorted_class_names == true_class_name_i)
    if (length(rank_true_class) == 0) { # Should not happen if true_class_name_i is valid
      stop(paste("ERROR: True class '", true_class_name_i, "' not found after sorting. Available: ", 
                 paste(sorted_class_names, collapse=", "), sep=""))
    }
    
    # Sum probabilities up to the rank of the true class
    scores[i] <- sum(sorted_probs[1:rank_true_class[1]])
  }
  return(scores)
}

create_prediction_sets_adaptive <- function(predicted_probs_matrix_new_data, q_hat) {
  # Purpose: Creates adaptive prediction sets. Classes are added to the set in order
  #          of decreasing probability until their cumulative probability sum meets or exceeds q_hat.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of predicted probabilities for new data.
  #   - q_hat: The calibrated q_hat threshold (from adaptive scores).
  # Returns: A list of character vectors, representing adaptive prediction sets.
  
  prediction_sets <- vector("list", nrow(predicted_probs_matrix_new_data))
  
  for (i in 1:nrow(predicted_probs_matrix_new_data)) {
    probs_i <- predicted_probs_matrix_new_data[i, ]
    
    # Sort probabilities and get corresponding class names
    sorted_indices <- order(probs_i, decreasing = TRUE)
    sorted_probs <- probs_i[sorted_indices]
    
    current_colnames <- colnames(predicted_probs_matrix_new_data)
    if(is.null(current_colnames) && !is.null(names(probs_i))) current_colnames <- names(probs_i)
    if(is.null(current_colnames)) stop("ERROR: Could not determine class names from predicted_probs_matrix_new_data columns or names of probs_i.")
    sorted_class_names <- current_colnames[sorted_indices]
    
    current_set <- character(0)
    cumulative_prob <- 0
    
    # Add classes to the set until cumulative probability reaches q_hat
    for (j in 1:length(sorted_probs)) {
      cumulative_prob <- cumulative_prob + sorted_probs[j]
      current_set <- c(current_set, sorted_class_names[j])
      if (cumulative_prob >= q_hat) {
        break # Stop once the sum meets or exceeds q_hat
      }
    }
    prediction_sets[[i]] <- current_set
  }
  return(prediction_sets)
}

# --- Section 2.4: Conformalizing Bayes ---

get_non_conformity_scores_bayes <- function(predicted_probs_matrix, true_labels) {
  # Purpose: Calculates non-conformity scores for Conformalizing Bayes.
  #          The score is the negative of the model's predicted probability (or posterior
  #          density/mass) for the true class. s_i = -f_hat(X_i)_Yi.
  # Parameters:
  #   - predicted_probs_matrix: A matrix of predicted probabilities (rows=samples, cols=classes).
  #                               Colnames should be the class labels.
  #   - true_labels: A vector of true class labels (factor or character).
  # Returns: A numeric vector of non-conformity scores.
  
  # This function is structurally similar to get_non_conformity_scores_basic,
  # but the score definition is different.
  scores <- numeric(length(true_labels))
  class_names <- colnames(predicted_probs_matrix) 
  
  for (i in 1:length(true_labels)) {
    true_class_name <- as.character(true_labels[i])
    if (!(true_class_name %in% class_names)) {
      stop(paste("ERROR: True class '", true_class_name, "' not found in probability matrix columns: ", 
                 paste(class_names, collapse=", "), sep=""))
    }
    prob_true_class <- predicted_probs_matrix[i, true_class_name]
    scores[i] <- -prob_true_class # Score is negative of the probability of the true class
  }
  return(scores)
}

create_prediction_sets_bayes <- function(predicted_probs_matrix_new_data, q_hat_bayes) {
  # Purpose: Creates prediction sets for Conformalizing Bayes.
  #          The set includes classes y for which their predicted probability (or posterior
  #          density/mass) f_hat(X)_y is greater than -q_hat_bayes.
  #          T(X) = {y : f_hat(X)_y > -q_hat_bayes}.
  # Parameters:
  #   - predicted_probs_matrix_new_data: Matrix of predicted probabilities for new data.
  #   - q_hat_bayes: The calibrated q_hat threshold (typically a negative value from Bayes scores).
  # Returns: A list of character vectors, where each vector is a prediction set for a sample.
  
  prediction_sets <- vector("list", nrow(predicted_probs_matrix_new_data))
  class_names <- colnames(predicted_probs_matrix_new_data)
  # The threshold for inclusion is -q_hat_bayes (which will be positive if q_hat_bayes is negative)
  threshold <- -q_hat_bayes 
  
  for (i in 1:nrow(predicted_probs_matrix_new_data)) {
    # Select classes whose predicted probability meets or exceeds the threshold
    current_set <- class_names[predicted_probs_matrix_new_data[i, ] > threshold]
    prediction_sets[[i]] <- if (length(current_set) == 0) character(0) else current_set
  }
  return(prediction_sets)
}

# --- Section 2.2: Conformalized Quantile Regression ---

train_quantile_models <- function(train_data, taus = c(0.05, 0.95)) {
  if (!require("quantreg")) install.packages("quantreg", dependencies = TRUE)
  library(quantreg)
  models <- lapply(taus, function(tau) {
    rq(Petal.Length ~ ., data = train_data, tau = tau)
  })
  names(models) <- paste0("tau_", taus)
  return(models)
}
compute_conformal_quantile <- function(models, calib_data, alpha = 0.1) {
  q_low  <- predict(models[[1]], newdata = calib_data)
  q_high <- predict(models[[2]], newdata = calib_data)
  scores <- pmax(q_low - calib_data$Petal.Length, calib_data$Petal.Length - q_high)
  q_conformal <- quantile(scores, probs = 1 - alpha, type = 1)
  cat("Quantile conformale q^ =", q_conformal, "\n")
  return(q_conformal)
}
predict_intervals <- function(models, test_data, q_conformal) {
  q_low  <- predict(models[[1]], newdata = test_data)
  q_high <- predict(models[[2]], newdata = test_data)
  lower <- q_low - abs(q_conformal)
  upper <- q_high + abs(q_conformal)
  return(list(lower = lower, upper = upper))
}
compute_conformal_quantile <- function(models, calib_data, alpha = 0.1) {
  q_low  <- predict(models[[1]], newdata = calib_data)
  q_high <- predict(models[[2]], newdata = calib_data)
  scores <- pmax(q_low - calib_data$Petal.Length, calib_data$Petal.Length - q_high)
  q_conformal <- quantile(scores, probs = 1 - alpha, type = 1)
  cat("Quantile conformale q^ =", q_conformal, "\n")
  return(q_conformal)
}
plot_intervals <- function(results_df, title_text) {
  if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
  library(ggplot2)
  ggplot(results_df, aes(x = index)) +
    geom_point(aes(y = true_value), color = "black", size = 1.8) +
    geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "red") +
    labs(title = title_text, x = "Osservazione nel test set", y = "Petal.Length") +
    theme_minimal()
}
evaluate_fsc <- function(test_data, covered_vec, feature_name = "SepalWidthCm") {
  if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
  library(ggplot2)
  group <- ifelse(test_data[[feature_name]] <= median(test_data[[feature_name]]), "Basso", "Alto")
  group_df <- data.frame(group = group, covered = covered_vec)
  fsc_coverage <- tapply(group_df$covered, group_df$group, mean)
  cat("\nCopertura per ampiezza intervallo (FSC):\n")
  print(round(ssc_coverage, 3))
  
  fsc_df <- data.frame(gruppo = names(fsc_coverage), copertura = as.numeric(fsc_coverage))
  ggplot(fsc_df, aes(x = gruppo, y = copertura, fill = gruppo)) +
    geom_col(width = 0.6) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
    ylim(0, 1.05) +
    labs(title = "Feature-Stratified Coverage (FSC)", x = feature_name, y = "Copertura empirica") +
    theme_minimal() +
    theme(legend.position = "none")
}

evaluate_ssc <- function(width_vec, covered_vec) {
  if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
  library(ggplot2)
  quantiles <- quantile(width_vec, probs = c(0.33, 0.66))
  interval_size <- cut(width_vec,
                       breaks = c(-Inf, quantiles[1], quantiles[2], Inf),
                       labels = c("Piccolo", "Medio", "Grande"))
  ssc_df <- data.frame(bin = interval_size, covered = covered_vec)
  ssc_coverage <- tapply(ssc_df$covered, ssc_df$bin, mean)
  cat("\nCopertura per ampiezza intervallo (SSC):\n")
  print(round(ssc_coverage, 3))
  ssc_df_plot <- data.frame(gruppo = names(ssc_coverage),
                            copertura = as.numeric(ssc_coverage))
  ggplot(ssc_df_plot, aes(x = gruppo, y = copertura, fill = gruppo)) +
    geom_col(width = 0.6) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
    ylim(0, 1.05) +
    labs(title = "Set-Stratified Coverage (SSC)", x = "Ampiezza intervallo", y = "Copertura empirica") +
    theme_minimal() +
    theme(legend.position = "none")
}


# --- Section 2.3: Scalar Uncertainty Estimates ---
train_svm_model <- function(train_data) {
  if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
  library(e1071)
  
  svm(Petal.Length ~ ., data = train_data, type = "eps-regression")
}
compute_svm_scores <- function(model, calib_data, mode = "residuals") {
  preds <- predict(model, newdata = calib_data)
  Y <- calib_data$Petal.Length
  
  if (mode == "residuals") {
    residuals <- abs(Y - preds)
    return(residuals)  
  }
  
  if (mode == "squared") {
    residuals <- (Y - preds)^2
    return(residuals)
  }
}
build_svm_intervals <- function(f_model, u_model, data, q_hat, stddev = FALSE) {
  f_pred <- predict(f_model, newdata = data)
  u_pred <- predict(u_model, newdata = data)
  
  if (stddev) {
    u_pred <- sqrt(u_pred)
  }
  
  lower <- f_pred - q_hat * u_pred
  upper <- f_pred + q_hat * u_pred
  return(list(lower = lower, upper = upper, midpoint = f_pred))
}
compute_conformal_quantile_svm <- function(residuals, u_hat, alpha = 0.1) {
  scores <- residuals / u_hat
  quantile(scores, probs = 1 - alpha, type = 1)
}
train_uncertainty_model <- function(residuals, features) {
  if (!require("e1071")) install.packages("e1071", dependencies = TRUE)
  library(e1071)
  
  # Allena un modello SVM per predire i residui
  svm(residuals ~ ., data = cbind(residuals = residuals, features))
}
