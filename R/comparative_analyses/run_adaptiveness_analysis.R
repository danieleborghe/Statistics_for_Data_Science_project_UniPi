# R/run_adaptiveness_analysis.R
#
# Scopo:
# Genera grafici comparativi per analizzare l'adattività dei metodi di regressione.
# Mette in relazione i punteggi di non-conformità con l'errore di predizione assoluto (residuo),
# come suggerito nella Sezione 4.1 del paper di riferimento.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory.
#   1. Caricamento e Unione dei Dati di Adattività:
#      - Carica i file CSV generati dagli esperimenti di regressione.
#      - Unisce i dati in un unico dataframe, aggiungendo una colonna 'method'.
#   2. Generazione del Grafico Comparativo:
#      - Crea uno scatter plot di 'residuo vs. punteggio'.
#      - Usa facet_wrap per creare un pannello per ogni metodo.
#      - Aggiunge una linea di tendenza (smoothing) per visualizzare meglio la correlazione.
#      - Salva il grafico finale.

# Setup: Sourcing e Librerie
source("R/experimentation_utils.R")

all_required_packages <- c("dplyr", "ggplot2", "viridis")
check_and_load_packages(all_required_packages)

RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 1. Caricamento e Unione dei Dati di Adattività ---

# Step 1.1: Carica i dati di adattività per ogni metodo.
adapt_quantile <- read.csv("results/tables/section2_2_quantile_reg/adaptiveness_data_BASESEED_RUN.csv")
adapt_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/adaptiveness_data_BASESEED_RUN.csv")
adapt_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/adaptiveness_data_BASESEED_RUN.csv")

# Step 1.2: Aggiungi una colonna identificativa del metodo.
adapt_quantile$method <- "Quantile Reg. (Sec. 2.2)"
adapt_scalar_abs$method <- "Scalar Uncertainty (Residuals)"
adapt_scalar_stddev$method <- "Scalar Uncertainty (Std Dev)"

# Step 1.3: Unisci i dataframe.
combined_adaptiveness_data <- dplyr::bind_rows(
  adapt_quantile,
  adapt_scalar_abs,
  adapt_scalar_stddev
)

# --- 2. Generazione del Grafico Comparativo ---

# Step 2.1: Crea lo scatter plot comparativo.
adaptiveness_plot <- ggplot(
  combined_adaptiveness_data,
  aes(x = non_conformity_score, y = absolute_residual)
) +
  geom_point(aes(color = method), alpha = 0.6, size = 2) + # Punti colorati per metodo
  geom_smooth(method = "loess", color = "blue", se = FALSE, linetype = "dashed") + # Linea di tendenza
  facet_wrap(~ method, scales = "free", ncol = 3) + # Un pannello per ogni metodo
  labs(
    subtitle = "A positive correlation indicates good adaptiveness: larger errors correspond to higher scores.",
    x = "Non-Conformity Score",
    y = "Absolute Prediction Error (Residual)",
  ) +
  scale_color_viridis_d(direction = -1) +
  theme_light(base_size = 18) + 
  theme(
    legend.position = "none",
    strip.text = element_text(face = "bold", size = 16), 
    title = element_text(size = 20, face = "bold"), 
    plot.subtitle = element_text(size = 16), 
    plot.caption = element_text(size = 12), 
    axis.title = element_text(size = 16), 
    axis.text = element_text(size = 14)
  )

# Step 2.2: Salva il grafico.
output_filename <- file.path(COMPARATIVE_PLOTS_DIR, "scatter_adaptiveness_regression.png")
ggsave(output_filename, plot = adaptiveness_plot, width = 14, height = 6, dpi = 300)