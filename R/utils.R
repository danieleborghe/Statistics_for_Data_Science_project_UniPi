# R/utils.R
#
# Purpose:
# This script contains utility functions that can be reused across
# different parts of the R_Conformal_Iris_SVM project.
#
# Functions:
#   - save_detailed_test_predictions(test_true_labels, prediction_sets_list, 
#                                    test_probs_matrix, output_directory, base_filename):
#     Saves a detailed CSV of test predictions including true labels, prediction sets,
#     and class probabilities.

save_detailed_test_predictions <- function(test_true_labels, 
                                           prediction_sets_list, 
                                           test_probs_matrix, 
                                           output_directory, 
                                           base_filename) {
  # Purpose: Creates and saves a CSV file with detailed information for each test sample,
  #          including its true label, the generated prediction set, and the predicted
  #          probabilities for all classes.
  # Parameters:
  #   - test_true_labels: Vector of true class labels for the test set.
  #   - prediction_sets_list: List where each element is a character vector (prediction set).
  #   - test_probs_matrix: Matrix of predicted class probabilities for the test set 
  #                        (rows=samples, cols=classes with colnames as class labels).
  #   - output_directory: The directory path where the CSV file will be saved.
  #   - base_filename: The base name for the CSV file (e.g., "detailed_predictions_basic.csv").
  # Returns:
  #   - Invisibly returns the full path to the saved file, or NULL if saving failed.
  
  cat(paste0("INFO: Preparing to save detailed test predictions to '", base_filename, "'...\n"))
  
  if (length(test_true_labels) != length(prediction_sets_list) || 
      length(test_true_labels) != nrow(test_probs_matrix)) {
    stop("ERROR: Input arguments to save_detailed_test_predictions have mismatched lengths/rows.")
  }
  
  # Get class names from the probability matrix colnames
  class_labels_raw <- colnames(test_probs_matrix) 
  if (is.null(class_labels_raw)) {
    warning("WARN: Column names for class probabilities are missing from test_probs_matrix. Using generic names.")
    prob_col_names <- paste0("Prob_Class", 1:ncol(test_probs_matrix))
  } else {
    prob_col_names <- paste0("Prob_", class_labels_raw) # Prepend "Prob_"
  }
  
  detailed_results_list <- vector("list", length(test_true_labels))
  
  for (i in 1:length(test_true_labels)) {
    pred_set <- prediction_sets_list[[i]]
    pred_set_string <- if (length(pred_set) == 0) "" else paste(sort(pred_set), collapse = "; ")
    
    row_data <- list(
      SampleID = i,
      TrueLabel = as.character(test_true_labels[i]),
      PredictionSet = pred_set_string
    )
    
    # Add probability columns dynamically
    for (j in 1:ncol(test_probs_matrix)) {
      row_data[[prob_col_names[j]]] <- test_probs_matrix[i, j]
    }
    
    detailed_results_list[[i]] <- as.data.frame(row_data, stringsAsFactors = FALSE)
  }
  
  # Combine all rows into a single data frame
  if (length(detailed_results_list) > 0) {
    if (requireNamespace("dplyr", quietly = TRUE)) {
      detailed_results_df <- dplyr::bind_rows(detailed_results_list)
    } else {
      warning("WARN: dplyr not available for bind_rows. Using base R rbind, which might be slower or coerce types.")
      # Base R fallback - ensure all data frames have same names for rbind
      # This part can be tricky if column names differ subtly or types are inconsistent
      # For simplicity, assuming as.data.frame inside loop handles basic cases.
      # A more robust base R version would pre-allocate a data.frame or use a more careful rbind loop.
      detailed_results_df <- do.call(rbind, lapply(detailed_results_list, function(x) data.frame(lapply(x, as.character), stringsAsFactors = FALSE)))
      # Convert probability columns back to numeric if they were coerced
      for(col_name in prob_col_names) {
        if(col_name %in% names(detailed_results_df)) {
          detailed_results_df[[col_name]] <- as.numeric(detailed_results_df[[col_name]])
        }
      }
    }
  } else {
    # Create an empty data frame with expected columns if detailed_results_list is empty
    # This prevents errors if test set is empty for some reason
    col_structure <- c(list(SampleID=integer(), TrueLabel=character(), PredictionSet=character()), 
                       stats::setNames(replicate(length(prob_col_names), numeric(), simplify = FALSE), prob_col_names))
    detailed_results_df <- data.frame(col_structure[!sapply(col_structure, is.null)], stringsAsFactors = FALSE)
    
  }
  
  # Ensure output directory exists
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  detailed_filename_path <- file.path(output_directory, base_filename)
  
  tryCatch({
    write.csv(detailed_results_df, detailed_filename_path, row.names = FALSE)
    cat(paste0("INFO: Detailed test predictions and probabilities saved to '", detailed_filename_path, "'\n"))
    return(invisible(detailed_filename_path))
  }, error = function(e) {
    warning(paste0("WARN: Failed to save detailed test predictions to '", detailed_filename_path, "'. Error: ", e$message))
    return(invisible(NULL))
  })
}