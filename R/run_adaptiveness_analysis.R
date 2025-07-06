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

# --- 0. Setup: Sourcing e Librerie ---
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Script di Analisi di Adattività (Residuo vs. Punteggio) ---\n")

# Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2", "viridis")
check_and_load_packages(all_required_packages)

# Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I grafici di adattività verranno salvati in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# --- 1. Caricamento e Unione dei Dati di Adattività ---
cat("\nINFO: --- Caricamento e unione dei dati di adattività per i metodi di regressione ---\n")

tryCatch({
  # Step 1.1: Carica i dati di adattività per ogni metodo.
  # Questi file dovrebbero essere stati generati eseguendo gli script degli esperimenti.
  adapt_quantile <- read.csv("results/tables/section2_2_quantile_reg/adaptiveness_data_BASESEED_RUN.csv")
  adapt_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/adaptiveness_data_BASESEED_RUN.csv")
  adapt_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/adaptiveness_data_BASESEED_RUN.csv")
  
  # Step 1.2: Aggiungi una colonna identificativa del metodo.
  adapt_quantile$method <- "Reg. Quantile (Sez. 2.2)"
  adapt_scalar_abs$method <- "Incertezza Scalare (Residui)"
  adapt_scalar_stddev$method <- "Incertezza Scalare (Std Dev)"
  
  # Step 1.3: Unisci i dataframe.
  combined_adaptiveness_data <- dplyr::bind_rows(
    adapt_quantile,
    adapt_scalar_abs,
    adapt_scalar_stddev
  )
  
  cat("INFO: Dati caricati e uniti con successo.\n")
  
  # --- 2. Generazione del Grafico Comparativo ---
  cat("INFO: Generazione del grafico di adattività comparativo...\n")
  
  # Step 2.1: Crea lo scatter plot comparativo.
  adaptiveness_plot <- ggplot(
    combined_adaptiveness_data,
    aes(x = non_conformity_score, y = absolute_residual)
  ) +
    geom_point(aes(color = method), alpha = 0.6, size = 2) + # Punti colorati per metodo
    geom_smooth(method = "loess", color = "blue", se = FALSE, linetype = "dashed") + # Linea di tendenza
    facet_wrap(~ method, scales = "free", ncol = 3) + # Un pannello per ogni metodo
    labs(
      title = "Analisi di Adattività: Correlazione tra Punteggio e Errore",
      subtitle = "Confronto tra i metodi di regressione sul set di calibrazione (Esecuzione BASE_SEED)",
      x = "Punteggio di Non-Conformità (Score)",
      y = "Errore di Predizione Assoluto (Residual)",
      caption = "Una correlazione positiva indica una buona adattività: errori maggiori corrispondono a punteggi più alti."
    ) +
    scale_color_viridis_d(direction = -1) +
    theme_light(base_size = 14) +
    theme(
      legend.position = "none", # La legenda non è necessaria grazie ai titoli dei pannelli
      strip.text = element_text(face = "bold") # Rende i titoli dei pannelli in grassetto
    )
  
  # Step 2.2: Salva il grafico.
  output_filename <- file.path(COMPARATIVE_PLOTS_DIR, "scatter_adaptiveness_regression.png")
  ggsave(output_filename, plot = adaptiveness_plot, width = 14, height = 6, dpi = 300)
  cat(paste0("INFO: Grafico di adattività comparativo salvato in '", output_filename, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi di adattività. Verifica che i file 'adaptiveness_data_...' esistano.\nErrore:", e$message, "\n")
})

cat("\nINFO: --- Script di Analisi di Adattività Completato ---\n")