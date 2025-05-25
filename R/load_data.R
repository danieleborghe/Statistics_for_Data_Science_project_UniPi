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