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

cat("INFO: --- Avvio Script di Visualizzazione Comparativa degli Intervalli (Ordinato) ---\n")

# Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2", "viridis")
check_and_load_packages(all_required_packages)

# Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: Il grafico comparativo degli intervalli verrà salvato in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# --- 1. Caricamento e Unione dei Dati ---
cat("\nINFO: --- Caricamento e unione dei dati degli intervalli per i metodi di regressione ---\n")

tryCatch({
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
  
  # --- MODIFICA CHIAVE: Ordinamento dei dati ---
  # Ordina l'intero dataset per metodo e poi per il valore reale (TrueValue).
  # Successivamente, crea un nuovo indice `OrderedSampleID` per l'asse x.
  combined_intervals_data <- combined_intervals_data %>%
    dplyr::arrange(method, TrueValue) %>%
    dplyr::group_by(method) %>%
    dplyr::mutate(OrderedSampleID = row_number()) %>%
    dplyr::ungroup()
  
  cat("INFO: Dati caricati, uniti e ordinati per valore reale.\n")
  
  # --- 2. Generazione del Grafico Comparativo ---
  cat("INFO: Generazione del grafico comparativo degli intervalli...\n")
  
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
      name = "Coperto",
      labels = c("TRUE" = "Sì", "FALSE" = "No")
    ) +
    labs(
      title = "Confronto degli Intervalli Predittivi Conformi (Ordinati per Valore Reale)",
      subtitle = "Confronto su larghezza e copertura per ogni osservazione del test set",
      x = "Osservazione nel test set (ordinata per Petal.Length)",
      y = "Petal.Length"
    ) +
    theme_light(base_size = 14) +
    theme(
      legend.position = "top",
      strip.text = element_text(face = "bold", size = 12)
    )
  
  # Step 2.2: Salva il grafico.
  output_filename <- file.path(COMPARATIVE_PLOTS_DIR, "plot_interval_comparison_regression_sorted.png")
  ggsave(output_filename, plot = interval_comparison_plot, width = 15, height = 7, dpi = 300)
  cat(paste0("INFO: Grafico comparativo degli intervalli ordinato salvato in '", output_filename, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare la visualizzazione. Verifica che i file 'detailed_test_intervals...' esistano.\nErrore:", e$message, "\n")
})

cat("\nINFO: --- Script di Visualizzazione Comparativa Completato ---\n")