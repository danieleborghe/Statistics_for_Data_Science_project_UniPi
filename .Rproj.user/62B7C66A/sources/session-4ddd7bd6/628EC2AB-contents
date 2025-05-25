# R/evaluation_utils.R
#
# Purpose:
# This script provides functions for evaluating conformal prediction sets,
# focusing on metrics described in Section 4 of the reference paper, such as
# empirical coverage, set size analysis, Feature-Stratified Coverage (FSC),
# and Set-Stratified Coverage (SSC).
#
# Functions:
#   - calculate_empirical_coverage(prediction_sets_list, true_labels_test): Calculates marginal coverage.
#   - get_set_sizes(prediction_sets_list): Extracts prediction set sizes.
#   - plot_set_size_histogram(set_sizes, main_title): Plots histogram of set sizes.
#   - calculate_fsc(prediction_sets_list, ...): Calculates Feature-Stratified Coverage.
#   - calculate_ssc(prediction_sets_list, ...): Calculates Set-Stratified Coverage.
#   - plot_conditional_coverage(coverage_by_group_df, ...): Plots conditional coverage results.

# --- Section 4.1: Basic Checks ---
calculate_empirical_coverage <- function(prediction_sets_list, true_labels_test) {
  # Purpose: Calculates the empirical marginal coverage of the prediction sets.
  # Parameters:
  #   - prediction_sets_list: A list where each element is a character vector
  #                           representing a prediction set for one sample.
  #   - true_labels_test: A vector of the true class labels for the test samples.
  # Returns: A single numeric value representing the empirical coverage (0 to 1).
  
  covered_count <- 0
  # Iterate through each prediction set and corresponding true label
  for (i in 1:length(prediction_sets_list)) {
    # Check if the true label is present in the prediction set
    if (as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]) {
      covered_count <- covered_count + 1
    }
  }
  # Coverage is the fraction of correctly covered samples
  empirical_coverage <- covered_count / length(true_labels_test)
  return(empirical_coverage)
}

get_set_sizes <- function(prediction_sets_list) {
  # Purpose: Calculates the size (number of elements) of each prediction set.
  # Parameters:
  #   - prediction_sets_list: A list of prediction sets.
  # Returns: A numeric vector containing the size of each prediction set.
  set_sizes <- sapply(prediction_sets_list, length)
  return(set_sizes)
}

plot_set_size_histogram <- function(set_sizes, main_title = "Prediction Set Size Distribution") {
  # Purpose: Generates and displays a histogram of prediction set sizes.
  # Parameters:
  #   - set_sizes: A numeric vector of prediction set sizes.
  #   - main_title: The title for the histogram plot.
  # Returns: None (plots a histogram).
  cat(paste("INFO: Plotting set size histogram: '", main_title, "'\n", sep=""))
  hist(set_sizes, 
       main = main_title, 
       xlab = "Set Size", 
       ylab = "Frequency",
       # Define breaks to ensure each integer set size has its own bar
       breaks = seq(min(set_sizes, na.rm = TRUE)-0.5, max(set_sizes, na.rm = TRUE)+0.5, by=1),
       col = "lightblue",
       border = "black")
  cat(paste("INFO: Observed average set size: ", round(mean(set_sizes, na.rm = TRUE), 3), "\n", sep=""))
}

# --- Section 4.2: Evaluating Adaptiveness ---
calculate_fsc <- function(prediction_sets_list, true_labels_test, feature_vector, 
                          feature_name = "Feature", num_bins_for_continuous = 5) {
  # Purpose: Calculates Feature-Stratified Coverage (FSC).
  #          It groups data by a specified feature (binning if continuous)
  #          and computes coverage within each group, then reports the minimum.
  # Parameters:
  #   - prediction_sets_list: List of prediction sets.
  #   - true_labels_test: Vector of true labels.
  #   - feature_vector: Vector of the feature values to stratify by.
  #   - feature_name: A string name for the feature (for logging).
  #   - num_bins_for_continuous: Number of bins if the feature is continuous.
  # Returns: A list containing 'min_coverage' and 'coverage_by_group' (a data frame).
  
  cat(paste("INFO: Calculating Feature-Stratified Coverage (FSC) for feature '", feature_name, "'...\n", sep=""))
  
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    warning("WARN: dplyr package not installed. FSC might not be calculated robustly. Please install dplyr.")
  }
  
  df_eval <- data.frame(true_label = as.character(true_labels_test), feature_val = feature_vector)
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  # Create feature groups (binning for continuous features)
  if (is.numeric(df_eval$feature_val)) {
    unique_numeric_vals <- unique(df_eval$feature_val[!is.na(df_eval$feature_val)])
    if (length(unique_numeric_vals) > num_bins_for_continuous) {
      df_eval$feature_group <- cut(df_eval$feature_val, breaks = num_bins_for_continuous, include.lowest = TRUE, ordered_result = TRUE)
    } else {
      df_eval$feature_group <- factor(df_eval$feature_val)
    }
  } else {
    df_eval$feature_group <- as.factor(df_eval$feature_val)
  }
  
  # Calculate coverage per group using dplyr if available, otherwise fallback
  coverage_by_group_df <- tryCatch({
    suppressMessages(
      df_eval %>%
        dplyr::group_by(feature_group, .drop = FALSE) %>%
        dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = dplyr::n(), .groups = 'drop') %>%
        dplyr::filter(count > 0) # Only consider groups with actual data
    )
  }, error = function(e) {
    warning("WARN: Error during dplyr operation for FSC. Falling back to aggregate. Error: ", e$message)
    agg_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    count_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = length, drop = FALSE)
    colnames(agg_df) <- c("feature_group", "coverage"); colnames(count_df) <- c("feature_group", "count")
    merged_df <- merge(agg_df, count_df, by="feature_group")
    return(merged_df[merged_df$count > 0, ])
  })
  
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: No valid groups found for FSC or all coverages are NA.\n")
    return(list(min_coverage = NA, coverage_by_group = data.frame(feature_group=factor(), coverage=numeric(), count=integer())))
  }
  
  min_fsc <- min(coverage_by_group_df$coverage, na.rm = TRUE) # Find minimum coverage across groups
  cat(paste("INFO: Minimum FSC calculated: ", round(min_fsc, 3), "\n", sep=""))
  return(list(min_coverage = min_fsc, coverage_by_group = coverage_by_group_df))
}

calculate_ssc <- function(prediction_sets_list, true_labels_test, num_bins_for_size = 3) {
  # Purpose: Calculates Set-Stratified Coverage (SSC).
  #          It groups data by prediction set size (binning if many unique sizes)
  #          and computes coverage within each group, then reports the minimum.
  # Parameters:
  #   - prediction_sets_list: List of prediction sets.
  #   - true_labels_test: Vector of true labels.
  #   - num_bins_for_size: Number of bins to group set sizes into.
  # Returns: A list containing 'min_coverage' and 'coverage_by_group' (a data frame).
  
  cat("INFO: Calculating Set-Stratified Coverage (SSC)...\n")
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    warning("WARN: dplyr package not installed. SSC might not be calculated robustly. Please install dplyr.")
  }
  
  set_sizes <- get_set_sizes(prediction_sets_list)
  df_eval <- data.frame(true_label = as.character(true_labels_test), set_size = set_sizes)
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  # Create set size groups
  unique_sizes <- unique(df_eval$set_size[!is.na(df_eval$set_size)])
  if (length(unique_sizes) > num_bins_for_size && max(unique_sizes) > min(unique_sizes)) {
    df_eval$size_group <- cut(df_eval$set_size, breaks = num_bins_for_size, include.lowest = TRUE, ordered_result = TRUE)
  } else { # If few unique sizes, use them directly as groups
    df_eval$size_group <- factor(df_eval$set_size)
  }
  
  # Calculate coverage per size group
  coverage_by_group_df <- tryCatch({
    suppressMessages(
      df_eval %>%
        dplyr::group_by(size_group, .drop = FALSE) %>%
        dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = dplyr::n(), .groups = 'drop') %>%
        dplyr::filter(count > 0)
    )
  }, error = function(e) {
    warning("WARN: Error during dplyr operation for SSC. Falling back to aggregate. Error: ", e$message)
    agg_df <- aggregate(covered ~ size_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    count_df <- aggregate(covered ~ size_group, data = df_eval, FUN = length, drop = FALSE)
    colnames(agg_df) <- c("size_group", "coverage"); colnames(count_df) <- c("size_group", "count")
    merged_df <- merge(agg_df, count_df, by="size_group")
    return(merged_df[merged_df$count > 0, ])
  })
  
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: No valid groups found for SSC or all coverages are NA.\n")
    return(list(min_coverage = NA, coverage_by_group = data.frame(size_group=factor(), coverage=numeric(), count=integer())))
  }
  
  min_ssc <- min(coverage_by_group_df$coverage, na.rm = TRUE) # Find minimum coverage across size groups
  cat(paste("INFO: Minimum SSC calculated: ", round(min_ssc, 3), "\n", sep=""))
  return(list(min_coverage = min_ssc, coverage_by_group = coverage_by_group_df))
}

plot_conditional_coverage <- function(coverage_by_group_df, group_col_name, coverage_col_name, 
                                      desired_coverage, main_title) {
  # Purpose: Generates a bar plot for conditional coverage results (FSC or SSC).
  # Parameters:
  #   - coverage_by_group_df: Data frame from calculate_fsc/ssc (must contain group_col_name and coverage_col_name).
  #   - group_col_name: Name of the column in coverage_by_group_df representing the groups.
  #   - coverage_col_name: Name of the column representing coverage values.
  #   - desired_coverage: The target coverage level (e.g., 1-alpha) to draw as a reference line.
  #   - main_title: Title for the plot.
  # Returns: None (plots a ggplot).
  
  cat(paste("INFO: Plotting conditional coverage: '", main_title, "'\n", sep=""))
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("WARN: ggplot2 package not installed. Cannot generate conditional coverage plot. Printing table instead.")
    print(coverage_by_group_df)
    return(invisible(NULL)) # Return NULL invisibly
  }
  library(ggplot2) # Ensure ggplot2 functions are available
  
  # Basic check for valid data frame and columns
  if (nrow(coverage_by_group_df) == 0 || 
      !coverage_col_name %in% names(coverage_by_group_df) || 
      !group_col_name %in% names(coverage_by_group_df)) {
    cat("WARN: Cannot plot conditional coverage due to empty or malformed data frame.\n")
    return(invisible(NULL))
  }
  
  # Ensure the group column is treated as a factor for distinct bars
  coverage_by_group_df[[group_col_name]] <- as.factor(coverage_by_group_df[[group_col_name]])
  
  # Create the plot
  p <- ggplot(coverage_by_group_df, aes_string(x = group_col_name, y = coverage_col_name, fill = group_col_name)) +
    geom_bar(stat = "identity", color = "black", na.rm = TRUE) + # na.rm for geom_bar
    geom_hline(yintercept = desired_coverage, linetype = "dashed", color = "red", linewidth = 1) +
    labs(title = main_title, x = "Group", y = "Empirical Coverage") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), # Rotate x-axis labels for readability
          legend.position = "none") + # No legend needed if fill is same as x
    ylim(0, 1) # Standard y-axis for coverage
  print(p) # Display the plot
}