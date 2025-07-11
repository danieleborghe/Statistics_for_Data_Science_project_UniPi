# R/run_ssc_fsc_comparative_analysis.R
#
# Scopo:
# Questo script esegue analisi comparative di Copertura Stratificata per Dimensione (SSC)
# e Copertura Stratificata per Feature (FSC) dagli esperimenti di classificazionee regressione.
#
# Struttura:
#   0. Setup: Carica librerie e crea directory.
#   1. Analisi Comparativa Classificazione:
#      - Carica i dati SSC e FSC per i metodi di classificazione.
#      - Unisce i dati.
#      - Genera e salva grafici SSC e FSC affiancati per la classificazione.
#   2. Analisi Comparativa Regressione:
#      - Carica i dati SSC e FSC per i metodi di regressione.
#      - Unisce i dati.
#      - Genera e salva grafici SSC e FSC affiancati per la regressione.

# Setup: Sourcing e Librerie
source("R/experimentation_utils.R")
all_required_packages <- c("dplyr", "ggplot2", "patchwork", "scales")
check_and_load_packages(all_required_packages)

RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Definisci parametri comuni del grafico.
ALPHA_CONF <- 0.1
TARGET_COVERAGE <- 1 - ALPHA_CONF
FONT_SIZE_BASE <- 18
ANNOTATION_OFFSET <- 0.02
DODGE_WIDTH <- 0.9
BAR_WIDTH <- 0.80

# --- 1. Analisi Comparativa Classificazione ---

# --- 1.1 Caricamento e Preparazione Dati SSC Classificazione ---

ssc_basic_class <- read.csv("results/tables/section1_basic/ssc_basic_BASESEED_RUN.csv")
ssc_adaptive_class <- read.csv("results/tables/section2_1_adaptive/ssc_adaptive_BASESEED_RUN.csv")
ssc_bayes_class <- read.csv("results/tables/section2_4_bayes/ssc_bayes_BASESEED_RUN.csv")

ssc_basic_class$method <- "Base (Sec. 1)"
ssc_adaptive_class$method <- "Adaptive (Sec. 2.1)"
ssc_bayes_class$method <- "Bayes (Sec. 2.4)"

combined_ssc_classification <- dplyr::bind_rows(ssc_basic_class, ssc_adaptive_class, ssc_bayes_class)

# --- 1.2 Generazione Grafico SSC Classificazione ---
plot_ssc_classification <- ggplot(
  combined_ssc_classification,
  aes(x = factor(size_group), y = coverage, fill = method)
) +
  geom_col(position = position_dodge(width = DODGE_WIDTH), width = BAR_WIDTH) +
  geom_hline(
    yintercept = TARGET_COVERAGE,
    linetype = "dashed", color = "red", size = 0.8
  ) +
  annotate(
    "text", x = Inf, y = TARGET_COVERAGE + ANNOTATION_OFFSET,
    label = paste("Target:", TARGET_COVERAGE),
    hjust = 1.1, color = "red", fontface = "bold", size = 3
  ) +
  labs(
    title = "SSC (Classification)",
    x = "Prediction Set Size",
    y = "Empirical Coverage"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1.05)) +
  theme_light(base_size = FONT_SIZE_BASE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_brewer(palette = "Blues", direction = -1, name = NULL)

# --- 1.3 Caricamento e Preparazione Dati FSC Classificazione ---
fsc_basic_class <- read.csv("results/tables/section1_basic/fsc_by_Sepal.Length_basic_BASESEED_RUN.csv")
fsc_adaptive_class <- read.csv("results/tables/section2_1_adaptive/fsc_by_Sepal.Length_adaptive_BASESEED_RUN.csv")
fsc_bayes_class <- read.csv("results/tables/section2_4_bayes/fsc_by_Sepal.Length_bayes_BASESEED_RUN.csv")

fsc_basic_class$method <- "Base (Sec. 1)"
fsc_adaptive_class$method <- "Adaptive (Sec. 2.1)"
fsc_bayes_class$method <- "Bayes (Sec. 2.4)"

combined_fsc_classification <- dplyr::bind_rows(fsc_basic_class, fsc_adaptive_class, fsc_bayes_class)

# --- 1.4 Generazione Grafico FSC Classificazione ---
plot_fsc_classification <- ggplot(
  combined_fsc_classification,
  aes(x = feature_group, y = coverage, fill = method)
) +
  geom_col(position = position_dodge(width = DODGE_WIDTH), width = BAR_WIDTH) +
  geom_hline(
    yintercept = TARGET_COVERAGE,
    linetype = "dashed", color = "red", size = 0.8
  ) +
  annotate(
    "text", x = Inf, y = TARGET_COVERAGE + ANNOTATION_OFFSET,
    label = paste("Target:", TARGET_COVERAGE),
    hjust = 1.1, color = "red", fontface = "bold", size = 3
  ) +
  labs(
    title = "FSC (Classification)",
    x = "Sepal.Length Group",
    y = "Empirical Coverage"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1.05)) +
  theme_light(base_size = FONT_SIZE_BASE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_brewer(palette = "Blues", direction = -1, name = NULL)

# --- 1.5 Combinazione e Salvataggio Grafici Classificazione ---
combined_plots_class <- guide_area() / (plot_ssc_classification + plot_fsc_classification) +
  plot_layout(
    guides = 'collect',
    heights = c(0.1, 0.9)
  ) +
  plot_annotation(
    theme = theme(plot.title = element_text(size = FONT_SIZE_BASE * 1.5, hjust = 0.5, face = "bold"))
  ) &
  theme(legend.direction = "horizontal")

output_filename_class <- file.path(COMPARATIVE_PLOTS_DIR, "comparative_classification_ssc_fsc_EN.png")
ggsave(output_filename_class, plot = combined_plots_class, width = 14, height = 7, dpi = 300)

# --- 2. Analisi Comparativa Regressione ---

# --- 2.1 Caricamento e Preparazione Dati SSC Regressione ---
ssc_quantile_reg <- read.csv("results/tables/section2_2_quantile_reg/ssc_BASESEED_RUN.csv")
ssc_scalar_abs_reg <- read.csv("results/tables/section2_3_scalar_uncert/ssc_BASESEED_RUN.csv")
ssc_scalar_stddev_reg <- read.csv("results/tables/section2_3_stddev_uncert/ssc_BASESEED_RUN.csv")

ssc_quantile_reg$method <- "Quantile Reg. (Sec. 2.2)"
ssc_scalar_abs_reg$method <- "Scalar Uncertainty (Residuals)"
ssc_scalar_stddev_reg$method <- "Scalar Uncertainty (Std Dev)"

combined_ssc_regression <- dplyr::bind_rows(ssc_quantile_reg, ssc_scalar_abs_reg, ssc_scalar_stddev_reg)

combined_ssc_regression$size_group <- factor(
  combined_ssc_regression$size_group,
  levels = c("Stretto", "Medio", "Largo"), # Nomi originali
  labels = c("Narrow", "Medium", "Wide") 
)

# --- 2.2 Generazione Grafico SSC Regressione ---
plot_ssc_regression <- ggplot(
  combined_ssc_regression,
  aes(x = size_group, y = coverage, fill = method)
) +
  geom_col(position = position_dodge(width = DODGE_WIDTH), width = BAR_WIDTH) +
  geom_hline(
    yintercept = TARGET_COVERAGE,
    linetype = "dashed", color = "red", size = 0.8
  ) +
  annotate(
    "text", x = Inf, y = TARGET_COVERAGE + ANNOTATION_OFFSET,
    label = paste("Target:", TARGET_COVERAGE),
    hjust = 1.1, color = "red", fontface = "bold", size = 3
  ) +
  labs(
    title = "SSC (Regression)",
    x = "Interval Width Group",
    y = "Empirical Coverage"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1.05)) +
  theme_light(base_size = FONT_SIZE_BASE) +
  scale_fill_brewer(palette = "Greens", direction = -1, name = NULL)

# --- 2.3 Caricamento e Preparazione Dati FSC Regressione ---
fsc_quantile_reg <- read.csv("results/tables/section2_2_quantile_reg/fsc_by_Sepal.Length_BASESEED_RUN.csv")
fsc_scalar_abs_reg <- read.csv("results/tables/section2_3_scalar_uncert/fsc_by_Sepal.Length_BASESEED_RUN.csv")
fsc_scalar_stddev_reg <- read.csv("results/tables/section2_3_stddev_uncert/fsc_by_Sepal.Length_BASESEED_RUN.csv")

fsc_quantile_reg$method <- "Quantile Reg. (Sec. 2.2)"
fsc_scalar_abs_reg$method <- "Scalar Uncertainty (Residuals)"
fsc_scalar_stddev_reg$method <- "Scalar Uncertainty (Std Dev)"

combined_fsc_regression <- dplyr::bind_rows(fsc_quantile_reg, fsc_scalar_abs_reg, fsc_scalar_stddev_reg)

# --- 2.4 Generazione Grafico FSC Regressione ---
plot_fsc_regression <- ggplot(
  combined_fsc_regression,
  aes(x = feature_group, y = coverage, fill = method)
) +
  geom_col(position = position_dodge(width = DODGE_WIDTH), width = BAR_WIDTH) +
  geom_hline(
    yintercept = TARGET_COVERAGE,
    linetype = "dashed", color = "red", size = 0.8
  ) +
  annotate(
    "text", x = Inf, y = TARGET_COVERAGE + ANNOTATION_OFFSET,
    label = paste("Target:", TARGET_COVERAGE),
    hjust = 1.1, color = "red", fontface = "bold", size = 3
  ) +
  labs(
    title = "FSC (Regression)",
    x = "Sepal.Length Group",
    y = "Empirical Coverage"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1.05)) +
  theme_light(base_size = FONT_SIZE_BASE) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
  scale_fill_brewer(palette = "Greens", direction = -1, name = NULL)

# --- 2.5 Combinazione e Salvataggio Grafici Regressione ---
combined_plots_reg <- guide_area() / (plot_ssc_regression + plot_fsc_regression) +
  plot_layout(
    guides = 'collect',
    heights = c(0.1, 0.9)
  ) +
  plot_annotation(
    theme = theme(plot.title = element_text(size = FONT_SIZE_BASE * 1.5, hjust = 0.5, face = "bold"))
  ) &
  theme(legend.direction = "horizontal")

output_filename_reg <- file.path(COMPARATIVE_PLOTS_DIR, "comparative_regression_ssc_fsc_EN.png")
ggsave(output_filename_reg, plot = combined_plots_reg, width = 14, height = 7, dpi = 300)