# R/load_data.R
#
# Purpose:
# This script is responsible for loading and performing initial preprocessing on the Iris dataset. The Iris dataset is built into R.
#
# Functions:
#   - load_iris_data(): Loads the Iris dataset, ensures 'Species' is a factor, and shuffles the data for reproducibility.

load_iris_data <- function() {
  # Purpose: Loads the built-in Iris dataset, converts 'Species' to a factor, and shuffles the dataset.
  # Parameters: None
  # Returns: A data frame containing the shuffled Iris dataset.
  
  cat("INFO: Loading Iris dataset...\n")
  data(iris) # Load the Iris dataset from R's built-in datasets
  
  # Ensure 'Species' column is treated as a factor
  iris$Species <- as.factor(iris$Species)
  
  # Shuffle the dataset for random sampling in later stages
  set.seed(123) # Set seed for reproducible shuffling
  iris_shuffled <- iris[sample(nrow(iris)), ]
  
  cat("INFO: Iris dataset loaded, 'Species' converted to factor, and data shuffled.\n")
  return(iris_shuffled)
}


load_and_split_data <- function(file_path, target_name = "PetalLengthCm", seed = 42) {
  if (!require("dplyr")) install.packages("dplyr", dependencies = TRUE)
  library(dplyr)
  data <- read.csv(file_path)
  names(data)[which(names(data) == target_name)] <- "Petal.Length"
  data <- data %>% select(-Id, -Species)
  set.seed(seed)
  n <- nrow(data)
  train_index <- sample(1:n, size = 0.6 * n)
  remaining <- setdiff(1:n, train_index)
  calib_index <- sample(remaining, size = 0.5 * length(remaining))
  test_index <- setdiff(remaining, calib_index)
  train_data <- data[train_index, ]
  calib_data <- data[calib_index, ]
  test_data  <- data[test_index, ]
  cat("Train:", nrow(train_data), "Calibrazione:", nrow(calib_data), "Test:", nrow(test_data), "\n")
  return(list(train = train_data, calib = calib_data, test = test_data))
}