# R/run_comparative_analysis.R
#
# Scopo:
# Questo script esegue analisi comparative tra i risultati dei diversi
# esperimenti di Conformal Prediction.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory per i risultati.
#   1. Analisi Comparativa #1: Classificazione
#      - Carica i dati di copertura marginale per i metodi Base, Adattivo e Bayes.
#      - Unisce e pre-elabora i dati per il grafico a barre.
#      - Calcola le coperture medie per ogni metodo.
#      - Genera e salva un istogramma comparativo.
#   2. Analisi Comparativa #2: Regressione
#      - Carica i dati di copertura marginale per Regressione Quantile e Incertezza Scalare.
#      - Unisce e pre-elabora i dati.
#      - Calcola le coperture medie.
#      - Genera e salva un istogramma comparativo.

# Setup: Sourcing e Librerie

source("R/experimentation_utils.R")
all_required_packages <- c("dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Definisci il livello di significatività alpha per il riferimento visuale.
ALPHA_CONF <- 0.1
TARGET_COVERAGE <- 1 - ALPHA_CONF
FONT_SIZE_BASE <- 18

# --- 1. Analisi Comparativa #1: Metodi di Classificazione ---

# Step 1.1: Carica i dati di copertura.
coverage_basic <- read.csv("results/tables/section1_basic/coverage_distribution_basic.csv")
coverage_adaptive <- read.csv("results/tables/section2_1_adaptive/coverage_distribution_adaptive.csv")
coverage_bayes <- read.csv("results/tables/section2_4_bayes/coverage_distribution_bayes.csv")

# Step 1.2: Aggiungi identificatori di metodo.
coverage_basic$method <- "Base (Sec. 1)"
coverage_adaptive$method <- "Adaptive (Sec. 2.1)"
coverage_bayes$method <- "Bayes (Sec. 2.4)"

# Step 1.3: Unisci e rinomina le colonne.
names(coverage_basic)[names(coverage_basic) == "empirical_coverage"] <- "coverage"
names(coverage_adaptive)[names(coverage_adaptive) == "empirical_coverage"] <- "coverage"
names(coverage_bayes)[names(coverage_bayes) == "empirical_coverage"] <- "coverage"
combined_classification_coverage <- dplyr::bind_rows(
  coverage_basic[, c("coverage", "method")],
  coverage_adaptive[, c("coverage", "method")],
  coverage_bayes[, c("coverage", "method")]
)

# Step 1.4: Calcola le frequenze per l'istogramma e le medie.
classification_freq <- combined_classification_coverage %>%
  dplyr::group_by(method, coverage) %>%
  dplyr::tally(name = "percentage") # N=100, quindi il count è già percentuale

classification_means <- combined_classification_coverage %>%
  dplyr::group_by(method) %>%
  dplyr::summarise(mean_coverage = mean(coverage, na.rm = TRUE))

# Step 1.5: Crea il grafico a barre comparativo.
plot_classification <- ggplot() +
  geom_col(
    data = classification_freq,
    aes(x = coverage, y = percentage, fill = method),
    position = "dodge"
  ) +
  # Linea della copertura target
  geom_vline(
    xintercept = TARGET_COVERAGE,
    linetype = "solid", color = "red", size = 1
  ) +
  # Linee delle coperture medie per ogni metodo
  geom_vline(
    data = classification_means,
    aes(xintercept = mean_coverage, color = method),
    linetype = "dashed", size = 1, show.legend = FALSE
  ) +
  labs(
    subtitle = "Comparison of coverage distributions from 100 runs per method.",
    x = "Empirical Coverage",
    y = "Percentage of Runs (%)",
    fill = "Method",
  ) +
  theme_light(base_size = FONT_SIZE_BASE) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, size = FONT_SIZE_BASE - 2),
    axis.title = element_text(size = FONT_SIZE_BASE + 2),
    plot.title = element_text(size = FONT_SIZE_BASE + 4, face = "bold"),
    plot.subtitle = element_text(size = FONT_SIZE_BASE),
    plot.caption = element_text(size = FONT_SIZE_BASE - 4)
  ) +
  scale_fill_brewer(palette = "Blues") +
  scale_color_brewer(palette = "Blues")


# Step 1.6: Salva il grafico.
output_filename_class <- file.path(COMPARATIVE_PLOTS_DIR, "histogram_comparison_classification_coverage.png")
ggsave(output_filename_class, plot = plot_classification, width = 12, height = 7, dpi = 300)


# --- 2. Analisi Comparativa #2: Metodi di Regressione ---

# Step 2.1: Carica i dati di copertura.
coverage_quantile <- read.csv("results/tables/section2_2_quantile_reg/coverage_width_distribution.csv")
coverage_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/coverage_width_distribution.csv")
coverage_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/coverage_width_distribution.csv") 

# Step 2.2: Aggiungi identificatori di metodo.
coverage_quantile$method <- "Quantile Reg. (Sec. 2.2)"
coverage_scalar_abs$method <- "Scalar Uncertainty (Residuals)"
coverage_scalar_stddev$method <- "Scalar Uncertainty (Std Dev)" 

# Step 2.3: Unisci i dataframe.
combined_regression_coverage <- dplyr::bind_rows(
  coverage_quantile[, c("coverage", "method")],
  coverage_scalar_abs[, c("coverage", "method")],
  coverage_scalar_stddev[, c("coverage", "method")]
)

# Step 2.4: Calcola le frequenze per l'istogramma e le medie.
regression_freq <- combined_regression_coverage %>%
  dplyr::group_by(method, coverage) %>%
  dplyr::tally(name = "percentage")

regression_means <- combined_regression_coverage %>%
  dplyr::group_by(method) %>%
  dplyr::summarise(mean_coverage = mean(coverage, na.rm = TRUE))

# Step 2.5: Crea il grafico a barre comparativo.
plot_regression <- ggplot() +
  geom_col(
    data = regression_freq,
    aes(x = coverage, y = percentage, fill = method),
    position = "dodge"
  ) +
  geom_vline(
    xintercept = TARGET_COVERAGE,
    linetype = "solid", color = "red", size = 1
  ) +
  geom_vline(
    data = regression_means,
    aes(xintercept = mean_coverage, color = method),
    linetype = "dashed", size = 1, show.legend = FALSE
  ) +
  labs(
    subtitle = "Comparison of coverage distributions from 100 runs per method.",
    x = "Empirical Coverage",
    y = "Percentage of Runs (%)",
    fill = "Method",
  ) +
  theme_light(base_size = FONT_SIZE_BASE) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, size = FONT_SIZE_BASE - 2),
    axis.title = element_text(size = FONT_SIZE_BASE + 2),
    plot.title = element_text(size = FONT_SIZE_BASE + 4, face = "bold"),
    plot.subtitle = element_text(size = FONT_SIZE_BASE),
    plot.caption = element_text(size = FONT_SIZE_BASE - 4)
  ) +
  scale_fill_brewer(palette = "Greens") +
  scale_color_brewer(palette = "Greens")

# Step 2.6: Salva il grafico.
output_filename_reg <- file.path(COMPARATIVE_PLOTS_DIR, "histogram_comparison_regression_coverage.png")
ggsave(output_filename_reg, plot = plot_regression, width = 12, height = 7, dpi = 300)