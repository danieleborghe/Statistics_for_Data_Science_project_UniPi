# R/03_evaluation_utils.R
#
# Purpose:
# This script provides functions for evaluating conformal prediction sets,
# focusing on metrics described in Section 4 of the reference paper, such as
# empirical coverage, set size analysis, Feature-Stratified Coverage (FSC),
# and Set-Stratified Coverage (SSC). It also includes a function to plot
# the distribution of marginal coverages from multiple runs.
#
# Functions:
#   - calculate_empirical_coverage(...): Calculates marginal coverage.
#   - get_set_sizes(...): Extracts prediction set sizes.
#   - plot_set_size_histogram(...): Plots histogram of set sizes.
#   - calculate_fsc(...): Calculates Feature-Stratified Coverage.
#   - calculate_ssc(...): Calculates Set-Stratified Coverage.
#   - plot_conditional_coverage(...): Plots conditional coverage results.
#   - plot_coverage_histogram(...): Plots histogram of marginal coverage distribution.

# --- Section 4.1: Basic Checks ---
calculate_empirical_coverage <- function(prediction_sets_list, true_labels_test) {
  # Purpose: Calculates the empirical marginal coverage of the prediction sets.
  # Parameters:
  #   - prediction_sets_list: A list where each element is a character vector
  #                           representing a prediction set for one sample.
  #   - true_labels_test: A vector of the true class labels for the test samples.
  # Returns: A single numeric value representing the empirical coverage (0 to 1).
  
  covered_count <- 0
  for (i in 1:length(prediction_sets_list)) {
    if (as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]) {
      covered_count <- covered_count + 1
    }
  }
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

# --- Function to Plot Histogram of Marginal Coverages ---
plot_coverage_histogram <- function(all_empirical_coverages, alpha_conf, n_runs, method_name = "") {
  # Purpose: Generates and displays a histogram of the distribution of empirical
  #          marginal coverage values obtained from multiple experiment runs.
  # Parameters:
  #   - all_empirical_coverages: A numeric vector of empirical coverage values.
  #   - alpha_conf: The significance level used (e.g., 0.1).
  #   - n_runs: The total number of runs (R) that generated the coverages.
  #   - method_name: A string identifying the conformal method (for the plot title).
  # Returns: None (plots a histogram).
  
  cat(paste0("INFO: Plotting histogram of ", n_runs, " marginal coverage values for '", method_name, "'...\n"))
  
  target_coverage <- 1 - alpha_conf
  mean_observed_coverage <- mean(all_empirical_coverages, na.rm = TRUE)
  
  hist_title <- paste("Distribution of Marginal Coverage\n(", method_name, ", N=", n_runs, " runs)", sep="")
  
  hist(all_empirical_coverages,
       main = hist_title,
       xlab = "Empirical Coverage",
       ylab = "Frequency",
       col = "skyblue",
       border = "black",
       las = 1, # Orient y-axis labels horizontally
       breaks = "Scott" # Or specify a number of breaks, e.g., 20
  )
  abline(v = target_coverage, col = "red", lwd = 2, lty = 2)
  abline(v = mean_observed_coverage, col = "darkgreen", lwd = 2, lty = 3)
  legend("topright", 
         legend = c(paste("Target Coverage (", round(target_coverage,2), ")", sep=""), 
                    paste("Mean Observed (", round(mean_observed_coverage,3), ")", sep="")),
         col = c("red", "darkgreen"), 
         lty = c(2,3), lwd = 2, cex=0.8, 
         box.lty=0) # No box around legend for cleaner look
  
  cat(paste0("INFO: Histogram for '", method_name, "' generated.\n"))
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
       breaks = seq(min(set_sizes, na.rm = TRUE)-0.5, max(set_sizes, na.rm = TRUE)+0.5, by=1),
       col = "lightblue",
       border = "black")
  cat(paste("INFO: Observed average set size: ", round(mean(set_sizes, na.rm = TRUE), 3), "\n", sep=""))
}

# --- Section 4.2: Evaluating Adaptiveness ---
calculate_fsc <- function(prediction_sets_list, true_labels_test, feature_vector, 
                          feature_name = "Feature", num_bins_for_continuous = 5) {
  # Purpose: Calculates Feature-Stratified Coverage (FSC).
  # Parameters: (as before)
  # Returns: A list containing 'min_coverage' and 'coverage_by_group' (a data frame).
  
  cat(paste("INFO: Calculating Feature-Stratified Coverage (FSC) for feature '", feature_name, "'...\n", sep=""))
  
  df_eval <- data.frame(true_label = as.character(true_labels_test), feature_val = feature_vector)
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
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
  
  if (requireNamespace("dplyr", quietly = TRUE)) {
    library(dplyr) # Load dplyr to use the pipe and its functions
    coverage_by_group_df <- tryCatch({
      suppressMessages(
        df_eval %>%
          group_by(feature_group, .drop = FALSE) %>%
          summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
          filter(count > 0)
      )
    }, error = function(e) { # Fallback if dplyr code still fails for some reason
      warning("WARN: Error during dplyr operation for FSC despite package being loaded. Falling back. Error: ", e$message)
      NULL # Signal to use base R fallback
    })
  } else {
    warning("WARN: dplyr package not installed. FSC calculation will use base R's aggregate.")
    coverage_by_group_df <- NULL # Signal to use base R fallback
  }
  
  # Base R fallback if dplyr not available or failed
  if (is.null(coverage_by_group_df)) {
    agg_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    count_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = length, drop = FALSE)
    colnames(agg_df) <- c("feature_group", "coverage"); colnames(count_df) <- c("feature_group", "count")
    coverage_by_group_df <- merge(agg_df, count_df, by="feature_group", all.x = TRUE) # Ensure all groups are kept initially
    coverage_by_group_df <- coverage_by_group_df[coverage_by_group_df$count > 0 & !is.na(coverage_by_group_df$feature_group), ]
  }
  
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: No valid groups found for FSC or all coverages are NA.\n")
    return(list(min_coverage = NA, coverage_by_group = data.frame(feature_group=factor(), coverage=numeric(), count=integer())))
  }
  
  min_fsc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  cat(paste("INFO: Minimum FSC calculated: ", round(min_fsc, 3), "\n", sep=""))
  return(list(min_coverage = min_fsc, coverage_by_group = coverage_by_group_df))
}

calculate_ssc <- function(prediction_sets_list, true_labels_test, num_bins_for_size = 3) {
  # Purpose: Calculates Set-Stratified Coverage (SSC).
  # Parameters: (as before)
  # Returns: A list containing 'min_coverage' and 'coverage_by_group' (a data frame).
  
  cat("INFO: Calculating Set-Stratified Coverage (SSC)...\n")
  
  set_sizes <- get_set_sizes(prediction_sets_list)
  df_eval <- data.frame(true_label = as.character(true_labels_test), set_size = set_sizes)
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  unique_sizes <- unique(df_eval$set_size[!is.na(df_eval$set_size)])
  if (length(unique_sizes) > num_bins_for_size && max(unique_sizes, na.rm=TRUE) > min(unique_sizes, na.rm=TRUE)) { # Added na.rm for max/min
    df_eval$size_group <- cut(df_eval$set_size, breaks = num_bins_for_size, include.lowest = TRUE, ordered_result = TRUE)
  } else {
    df_eval$size_group <- factor(df_eval$set_size)
  }
  
  if (requireNamespace("dplyr", quietly = TRUE)) {
    library(dplyr) # Load dplyr
    coverage_by_group_df <- tryCatch({
      suppressMessages(
        df_eval %>%
          group_by(size_group, .drop = FALSE) %>%
          summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
          filter(count > 0)
      )
    }, error = function(e) {
      warning("WARN: Error during dplyr operation for SSC despite package being loaded. Falling back. Error: ", e$message)
      NULL
    })
  } else {
    warning("WARN: dplyr package not installed. SSC calculation will use base R's aggregate.")
    coverage_by_group_df <- NULL
  }
  
  if (is.null(coverage_by_group_df)) {
    agg_df <- aggregate(covered ~ size_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    count_df <- aggregate(covered ~ size_group, data = df_eval, FUN = length, drop = FALSE)
    colnames(agg_df) <- c("size_group", "coverage"); colnames(count_df) <- c("size_group", "count")
    coverage_by_group_df <- merge(agg_df, count_df, by="size_group", all.x = TRUE)
    coverage_by_group_df <- coverage_by_group_df[coverage_by_group_df$count > 0 & !is.na(coverage_by_group_df$size_group), ]
  }
  
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: No valid groups found for SSC or all coverages are NA.\n")
    return(list(min_coverage = NA, coverage_by_group = data.frame(size_group=factor(), coverage=numeric(), count=integer())))
  }
  
  min_ssc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  cat(paste("INFO: Minimum SSC calculated: ", round(min_ssc, 3), "\n", sep=""))
  return(list(min_coverage = min_ssc, coverage_by_group = coverage_by_group_df))
}

plot_conditional_coverage <- function(coverage_by_group_df, group_col_name, coverage_col_name, 
                                      desired_coverage, main_title) {
  # Purpose: Generates a bar plot for conditional coverage results (FSC or SSC).
  # Parameters: (as before)
  # Returns: None (plots a ggplot).
  
  cat(paste("INFO: Plotting conditional coverage: '", main_title, "'\n", sep=""))
  
  if (nrow(coverage_by_group_df) == 0 || 
      !coverage_col_name %in% names(coverage_by_group_df) || 
      !group_col_name %in% names(coverage_by_group_df) ||
      all(is.na(coverage_by_group_df[[coverage_col_name]]))) { # Check if all coverage values are NA
    cat("WARN: Cannot plot conditional coverage due to empty, malformed, or all-NA data.\n")
    return(invisible(NULL))
  }
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("WARN: ggplot2 package not installed. Cannot generate conditional coverage plot. Printing table instead.")
    print(coverage_by_group_df)
    return(invisible(NULL))
  }
  library(ggplot2) 
  
  coverage_by_group_df[[group_col_name]] <- as.factor(coverage_by_group_df[[group_col_name]])
  
  p <- ggplot(coverage_by_group_df, aes_string(x = group_col_name, y = coverage_col_name, fill = group_col_name)) +
    geom_bar(stat = "identity", color = "black", na.rm = TRUE) +
    geom_hline(yintercept = desired_coverage, linetype = "dashed", color = "red", linewidth = 1) +
    labs(title = main_title, x = "Group", y = "Empirical Coverage") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
    ylim(0, 1)
  print(p)
}

# --- Section 2.2: Conformalized Quantile Regression---
print_interval_width <- function(lower, upper, label = "") {
  width <- upper - lower
  avg_width <- mean(width)
  cat("Ampiezza media dell'intervallo", label, "=", round(avg_width, 3), "\n")
  return(avg_width)
}
evaluate_adaptivity_plot <- function(results_df, Y_test) {
  residuals_abs <- abs(Y_test - results_df$midpoint)
  width <- results_df$upper - results_df$lower
  plot(width, residuals_abs,
       xlab = "Ampiezza intervallo predittivo",
       ylab = "Errore assoluto |Y - centro intervallo|",
       main = "Adattività degli intervalli (Quantile Regression)")
  abline(lm(residuals_abs ~ width), col = "red", lty = 2)
}
evaluate_coverage <- function(Y_true, lower, upper) {
  covered <- (Y_true >= lower) & (Y_true <= upper)
  coverage <- mean(covered)
  cat("Copertura empirica =", round(coverage, 3), "\n")
  return(coverage)
}
# --- Section 2.3: Scalar Uncertainty Estimates---
evaluate_intervals <- function(Y, lower, upper, label = "") {
  covered <- (Y >= lower) & (Y <= upper)
  coverage <- mean(covered)
  width <- upper - lower
  avg_width <- mean(width)
  
  cat(paste0("\n[", label, "] Copertura empirica = "), round(coverage, 3), "\n")
  cat(paste0("[", label, "] Ampiezza media = "), round(avg_width, 3), "\n")
  
  return(list(coverage = coverage, avg_width = avg_width, covered = covered, width = width))
}
plot_svm_intervals <- function(Y_true, lower, upper, midpoint, title) {
  if (!require("ggplot2")) install.packages("ggplot2", dependencies = TRUE)
  library(ggplot2)
  
  df <- data.frame(
    index = 1:length(Y_true),
    true_value = Y_true,
    lower = lower,
    upper = upper,
    midpoint = midpoint
  )
  
  ggplot(df, aes(x = index)) +
    geom_point(aes(y = true_value), color = "black", size = 1.8) +
    geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +
    geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "darkgreen") +
    labs(title = title,
         x = "Osservazione nel test set",
         y = "Petal.Length") +
    theme_minimal()
}
plot_adaptivity <- function(lower, upper, Y_true, label = "") {
  residuals_abs <- abs(Y_true - (lower + upper) / 2)
  width <- upper - lower
  
  plot(width, residuals_abs,
       xlab = paste("Ampiezza intervallo", label),
       ylab = "|Errore assoluto|",
       main = paste("Adattività", label))
  abline(lm(residuals_abs ~ width), col = "red", lty = 2)
}
evaluate_fsc <- function(test_data, covered_vec, feature_name = "SepalWidthCm") {
  group <- ifelse(test_data[[feature_name]] <= median(test_data[[feature_name]]), "Basso", "Alto")
  group_df <- data.frame(group = group, covered = covered_vec)
  fsc_coverage <- tapply(group_df$covered, group_df$group, mean)
  cat("\nCopertura per sottogruppi (FSC):\n")
  print(round(fsc_coverage, 3))
  return(fsc_coverage)
}
evaluate_ssc <- function(width_vec, covered_vec) {
  quantiles <- quantile(width_vec, probs = c(0.33, 0.66))
  interval_size <- cut(width_vec,
                       breaks = c(-Inf, quantiles[1], quantiles[2], Inf),
                       labels = c("Piccolo", "Medio", "Grande"))
  ssc_df <- data.frame(bin = interval_size, covered = covered_vec)
  ssc_coverage <- tapply(ssc_df$covered, ssc_df$bin, mean)
  cat("\nCopertura per ampiezza intervallo (SSC):\n")
  print(round(ssc_coverage, 3))
  return(ssc_coverage)
}
plot_fsc <- function(fsc_coverage, title = "FSC – Scalar Uncertainty") {
  library(ggplot2)
  df <- data.frame(
    gruppo = names(fsc_coverage),
    copertura = as.numeric(fsc_coverage)
  )
  
  ggplot(df, aes(x = gruppo, y = copertura, fill = gruppo)) +
    geom_col(width = 0.6) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
    ylim(0, 1.05) +
    labs(title = title, x = "Gruppo Sepal.Width", y = "Copertura empirica") +
    theme_minimal() +
    theme(legend.position = "none")
}
plot_ssc <- function(ssc_coverage, title = "SSC – Scalar Uncertainty") {
  library(ggplot2)
  df <- data.frame(
    gruppo = names(ssc_coverage),
    copertura = as.numeric(ssc_coverage)
  )
  
  ggplot(df, aes(x = gruppo, y = copertura, fill = gruppo)) +
    geom_col(width = 0.6) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
    ylim(0, 1.05) +
    labs(title = title, x = "Ampiezza intervallo", y = "Copertura empirica") +
    theme_minimal() +
    theme(legend.position = "none")
}
