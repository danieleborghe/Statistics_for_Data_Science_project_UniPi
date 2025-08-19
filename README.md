# A Deep Dive into Conformal Prediction: Theory, Implementation, and Analysis

This repository contains the R code, experimental results, and analysis for a research project on **Conformal Prediction (CP)**. The project provides a practical and theoretical exploration of how CP can be used to generate statistically rigorous, distribution-free prediction intervals and sets for both regression and classification tasks.

The project was developed as part of the "Statistics for Data Science" course at the **University of Pisa (UniPi)**.

---

## 📝 Table of Contents

- [The Core Concept: What is Conformal Prediction?](#-the-core-concept-what-is-conformal-prediction)
- [Project Objectives](#-project-objectives)
- [Technical Stack & Methodologies](#-technical-stack--methodologies)
- [Implemented Methods & Experiments](#-implemented-methods--experiments)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [How to Run the Experiments](#-how-to-run-the-experiments)
- [Authors](#-authors)

---

## 🎯 The Core Concept: What is Conformal Prediction?

Traditional machine learning models provide point predictions (e.g., "the price is €100" or "the class is 'spam'"), but they often lack a reliable measure of confidence. Conformal Prediction is a powerful framework that complements any underlying machine learning algorithm (like Random Forest, Neural Networks, etc.) to produce predictions with a user-defined confidence level.

Instead of a single output, a Conformal Predictor (CP) outputs:
-   **For Regression**: A **prediction interval** (e.g., "the price will be between €90 and €110").
-   **For Classification**: A **prediction set** of one or more classes (e.g., "{'spam', 'not spam'}" if the model is uncertain, or just "{'not spam'}" if it's confident).

The key guarantee of CP is **validity**: if you specify a confidence level of 95% (or significance level α=0.05), the true value will fall within the prediction interval/set 95% of the time in the long run, regardless of the data distribution.

---

## 💡 Project Objectives

This project aimed to:

1.  **Implement from Scratch** various Conformal Prediction algorithms in R.
2.  **Empirically Verify** the theoretical property of **validity** (coverage) across different models and datasets.
3.  **Analyze the Efficiency** of the predictors, measured by the size of the prediction intervals/sets.
4.  **Investigate Advanced Concepts** like **adaptiveness**, where the prediction intervals automatically adjust their size based on the difficulty of the prediction.
5.  **Compare** the performance of different non-conformity measures and underlying machine learning models.

---

## 💻 Technical Stack & Methodologies

-   **Language**: **R**
-   **Core Libraries**:
    -   **`dplyr`** & **`tidyr`**: For data manipulation and wrangling.
    -   **`ggplot2`**: For creating all the high-quality data visualizations and plots.
    -   **`randomForest`**: Used as the underlying algorithm for the non-conformity scores in classification experiments.
    -   **`quantreg`**: Used for Quantile Regression, a key component in generating adaptive prediction intervals.
    -   **`knitr`** & **`kableExtra`**: For generating formatted tables from the results.
-   **Statistical Concepts**:
    -   **Non-Conformity Measures (NCMs)**: The core of CP. We implemented various NCMs, such as Inverse Probability for classification and Absolute Error for regression.
    -   **Inductive vs. Transductive CP**: Our focus is on the more computationally efficient Inductive Conformal Prediction (ICP), which uses a calibration set.
    -   **Conditional Coverage**: We analyzed whether the validity guarantee holds across different subgroups of the data (e.g., for each class in a classification problem).

---

## 🔬 Implemented Methods & Experiments

Our research was structured around a series of experiments designed to test different facets of Conformal Prediction.

-   **Section 1: Basic Conformal Predictors**
    -   **Task**: Multi-class classification on the `iris` dataset.
    -   **Method**: Implemented a standard Inductive Conformal Predictor using a Random Forest model.
    -   **Analysis**: Verified that the empirical coverage matches the desired confidence level. Analyzed the size of prediction sets and investigated conditional coverage per class.

-   **Section 2: Advanced Regression Predictors & Adaptiveness**
    -   **Task**: Regression on the `BostonHousing` dataset.
    -   **Methods Implemented**:
        1.  **Standard CP**: A basic CP using absolute error as the NCM.
        2.  **Normalized CP**: An improved CP that normalizes the error based on the predicted value's magnitude.
        3.  **Quantile Regression CP**: A highly adaptive method where the prediction interval width is learned directly from the data using two quantile regression models.
    -   **Analysis**: Compared the validity and efficiency (average interval width) of all three methods. Conducted a deep dive into adaptiveness by showing how interval sizes correlate with prediction difficulty.

---

## 📈 Key Findings

-   **Validity is Guaranteed**: Across all experiments, our implemented Conformal Predictors successfully achieved the desired long-run coverage, confirming the theoretical guarantees of the framework.
-   **Adaptiveness is Key for Efficiency**: For regression, the standard CP produced intervals of constant width, which were inefficiently large for easy predictions. The **Quantile Regression-based CP** proved far superior, generating narrow, highly adaptive intervals that were wider only when the model's uncertainty was high.
-   **Trade-off between Coverage and Precision**: While validity is maintained, the efficiency (size of prediction sets/intervals) depends heavily on the quality of the underlying machine learning model and the chosen non-conformity measure.
-   **Conditional Coverage is Not Guaranteed**: Standard CPs guarantee average coverage over the entire data distribution, but not necessarily for specific subgroups (e.g., individual classes). Our analysis showed minor deviations in class-conditional coverage, a well-known area of ongoing research in the CP community.

---

## 📂 Repository Structure

```

.
├── R/
│   ├── conformal\_predictors.R            \# Core functions for CP implementation
│   ├── evaluation\_utils.R                \# Helper functions for evaluating results
│   ├── experimentation\_utils.R           \# Helper functions for running experiments
│   ├── comparative\_analyses/               \# Scripts to generate comparison plots/tables
│   └── experiments/                      \# Main scripts to run each experiment section
│       ├── sec1\_run\_experiment.R
│       └── ...
├── results/
│   ├── plots/                            \# Output plots from the analyses
│   └── tables/                           \# Output tables with detailed results
├── 26.ConformalPrediction.pdf           \# Main theoretical reference document
├── conformal\_prediction\_project.Rproj   \# RStudio Project file
└── README.md                            \# This file

````

---

## 🚀 How to Run the Experiments

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/danieleborghe/statistics_for_data_science_project_unipi.git](https://github.com/danieleborghe/statistics_for_data_science_project_unipi.git)
    cd statistics_for_data_science_project_unipi
    ```

2.  **Open the Project in RStudio:**
    -   The easiest way to work with this project is to open the `conformal_prediction_project.Rproj` file in RStudio. This will automatically set the correct working directory.

3.  **Install Dependencies:**
    -   Run the following command in the R console to install all the required packages:
    ```R
    install.packages(c("dplyr", "tidyr", "ggplot2", "randomForest", "quantreg", "knitr", "kableExtra"))
    ```

4.  **Run an Experiment:**
    -   Open one of the main experiment scripts located in the `R/experiments/` directory (e.g., `sec1_run_experiment.R`).
    -   Source the entire script in RStudio. The script will automatically:
        -   Load the necessary functions and data.
        -   Run the Conformal Prediction experiments.
        -   Save the resulting tables and plots to the `results/` directory.

---

## 👥 Authors

- **Daniele Borghesi**
- **Nicolas Humberto Montes De Oca Ibañez**
