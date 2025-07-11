# R/run_interval_comparison_plot.R
#
# Scopo:
# Genera una visualizzazione comparativa degli intervalli predittivi conformi
# per i diversi metodi di regressione, ordinando le osservazioni per valore reale
# per una migliore interpretabilità.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory.
#   1. Caricamento e Unione dei Dati Dettagliati degli Intervalli:
#      - Carica i file CSV 'detailed_test_intervals...' per ogni metodo.
#      - Unisce i dati in un unico dataframe, aggiungendo una colonna 'method'.
#      - **MODIFICA**: Ordina i dati per valore reale (TrueValue) e crea un nuovo indice ordinato.
#   2. Generazione del Grafico Comparativo:
#      - Crea uno scatter plot con barre di errore (error bars) per rappresentare gli intervalli.
#      - **MODIFICA**: Utilizza il nuovo indice ordinato sull'asse x.
#      - **MODIFICA**: Rimuove la linea di connessione tra i punti.
#      - Salva il grafico finale.

# --- 0. Setup: Sourcing e Librerie ---
source("R/experimentation_utils.R")

# Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2", "viridis")
check_and_load_packages(all_required_packages)

# Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 1. Caricamento e Unione dei Dati ---

# Step 1.1: Carica i dati dettagliati degli intervalli per ogni metodo.
intervals_quantile <- read.csv("results/tables/section2_2_quantile_reg/detailed_test_intervals_BASESEED_RUN.csv")
intervals_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/detailed_test_intervals_BASESEED_RUN.csv")
intervals_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/detailed_test_intervals_BASESEED_RUN.csv")

# Step 1.2: Aggiungi una colonna identificativa del metodo.
intervals_quantile$method <- "Quantile Regression"
intervals_scalar_abs$method <- "SVM Residuals"
intervals_scalar_stddev$method <- "SVM Std Dev"

# Step 1.3: Unisci i dataframe.
combined_intervals_data <- dplyr::bind_rows(
  intervals_quantile,
  intervals_scalar_abs,
  intervals_scalar_stddev
)

# Step 1.4: Ordina i metodi per una visualizzazione coerente.
combined_intervals_data$method <- factor(
  combined_intervals_data$method,
  levels = c("Quantile Regression", "SVM Residuals", "SVM Std Dev")
)

# --- CHIAVE MODIFICA: Ordinamento dei dati ---
# Ordina l'intero dataset per metodo e poi per il valore reale (TrueValue).
# Successivamente, crea un nuovo indice `OrderedSampleID` per l'asse x.
combined_intervals_data <- combined_intervals_data %>%
  dplyr::arrange(method, TrueValue) %>%
  dplyr::group_by(method) %>%
  dplyr::mutate(OrderedSampleID = row_number()) %>%
  dplyr::ungroup()

# --- 2. Generazione del Grafico Comparativo ---

# Step 2.1: Crea il grafico.
interval_comparison_plot <- ggplot(
  combined_intervals_data,
  # MODIFICA: Usa il nuovo indice ordinato sull'asse x
  aes(x = OrderedSampleID, y = TrueValue)
) +
  geom_errorbar(
    aes(ymin = LowerBound, ymax = UpperBound, color = as.factor(Covered)),
    width = 0.5,
    size = 0.8
  ) +
  geom_point(color = "black", size = 1.5) +
  # MODIFICA: Rimossa la linea `geom_line` che connetteva i punti.
  facet_wrap(~ method, ncol = 3, scales = "free_x") + # `scales = "free_x"` per non allineare gli ID tra i pannelli
  scale_color_manual(
    values = c("TRUE" = "darkgreen", "FALSE" = "red"),
    name = "Covered", # Tradotto in inglese
    labels = c("TRUE" = "Yes", "FALSE" = "No") # Tradotto in inglese
  ) +
  labs(
    subtitle = "Interval width and coverage comparison for each test set observation", # Tradotto in inglese
    x = "Test set observation (sorted by TrueValue)", # Tradotto in inglese
    y = "True Value" # Tradotto in inglese (assumendo Petal.Length è il TrueValue)
  ) +
  theme_light(base_size = 18) + # Aumentato il base_size per il font generale
  theme(
    legend.position = "top",
    strip.text = element_text(face = "bold", size = 16), # Aumentato il font dei titoli dei pannelli
    plot.title = element_text(size = 20, face = "bold"), # Aumentato il font del titolo principale
    plot.subtitle = element_text(size = 16), # Aumentato il font del sottotitolo
    axis.title = element_text(size = 16), # Aumentato il font dei titoli degli assi
    axis.text = element_text(size = 14) # Aumentato il font delle etichette degli assi
  )

# Step 2.2: Salva il grafico.
output_filename <- file.path(COMPARATIVE_PLOTS_DIR, "plot_interval_comparison_regression_sorted.png")
ggsave(output_filename, plot = interval_comparison_plot, width = 15, height = 7, dpi = 300)