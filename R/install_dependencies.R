# install_dependencies.R
#
# Purpose:
# This script checks for and installs all necessary R packages required for the project.
# It iterates through a predefined list of packages, installing any that are not already present in the R environment.

# List of required packages
required_packages <- c(
  "e1071",   # For Support Vector Machines (SVM)
  "dplyr",   # For data manipulation (used in evaluation utilities)
  "ggplot2"  # For plotting (used in evaluation utilities)
)

cat("INFO: Starting package dependency check...\n")

# Loop through the list and install if missing
for (pkg in required_packages) {
  # Check if the package is already installed
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(paste("INFO: Package '", pkg, "' not found. Attempting to install...\n", sep = ""))
    install.packages(pkg) # Install the package
    # Verify installation
    if (requireNamespace(pkg, quietly = TRUE)) {
      cat(paste("INFO: Package '", pkg, "' installed successfully.\n", sep = ""))
    } else {
      warning(paste("WARN: Failed to install package '", pkg, "'. Please install it manually.\n", sep = ""))
    }
  } else {
    cat(paste("INFO: Package '", pkg, "' is already installed.\n", sep = "")) # Optional: can be noisy
  }
}

cat("INFO: Package dependency check complete.\n")