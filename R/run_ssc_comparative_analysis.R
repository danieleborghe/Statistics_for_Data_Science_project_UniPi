# R/run_ssc_comparative_analysis.R
#
# Scopo:
# Esegue analisi comparative della Size-Stratified Coverage (SSC)
# tra i risultati degli esperimenti di classificazione e regressione.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory.
#   1. Analisi Comparativa SSC #1: Classificazione
#      - Carica i dati SSC per i metodi Base, Adattivo e Bayes.
#      - Unisce i dati. I gruppi sono basati sulla dimensione (cardinalità) dell'insieme.
#      - Genera e salva un grafico a barre comparativo.
#   2. Analisi Comparativa SSC #2: Regressione
#      - Carica i dati SSC per Regressione Quantile e Incertezza Scalare.
#      - Unisce i dati. I gruppi sono basati su quantili di larghezza ("Stretto", "Medio", "Largo").
#      - Genera e salva un grafico a barre comparativo.

# --- 0. Setup: Sourcing e Librerie ---
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Script di Analisi Comparativa SSC ---\n")

# Step 2: Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# Step 3: Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I grafici comparativi SSC verranno salvati in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# Step 4: Definisci parametri comuni per i grafici.
ALPHA_CONF <- 0.1
TARGET_COVERAGE <- 1 - ALPHA_CONF
FONT_SIZE_BASE <- 14

# --- 1. Analisi Comparativa SSC #1: Metodi di Classificazione ---
cat("\nINFO: --- Inizio Analisi Comparativa SSC #1: Classificazione ---\n")

tryCatch({
  # Step 1.1: Carica i dati SSC per ogni metodo.
  ssc_basic <- read.csv("results/tables/section1_basic/ssc_basic_BASESEED_RUN.csv")
  ssc_adaptive <- read.csv("results/tables/section2_1_adaptive/ssc_adaptive_BASESEED_RUN.csv")
  ssc_bayes <- read.csv("results/tables/section2_4_bayes/ssc_bayes_BASESEED_RUN.csv")
  
  # Step 1.2: Aggiungi una colonna identificativa del metodo.
  ssc_basic$method <- "Base (Sez. 1)"
  ssc_adaptive$method <- "Adattivo (Sez. 2.1)"
  ssc_bayes$method <- "Bayes (Sez. 2.4)"
  
  # Step 1.3: Unisci i dataframe.
  combined_ssc_classification <- dplyr::bind_rows(ssc_basic, ssc_adaptive, ssc_bayes)
  
  # Step 1.4: Crea il grafico a barre comparativo.
  plot_ssc_classification <- ggplot(
    combined_ssc_classification,
    aes(x = factor(size_group), y = coverage, fill = method)
  ) +
    geom_col(position = "dodge") +
    geom_hline(
      yintercept = TARGET_COVERAGE,
      linetype = "dashed", color = "red", size = 1
    ) +
    annotate(
      "text", x = Inf, y = TARGET_COVERAGE + 0.02,
      label = paste("Target:", TARGET_COVERAGE),
      hjust = 1.1, color = "red", fontface = "bold"
    ) +
    labs(
      title = "Analisi SSC Comparativa (Classificazione)",
      subtitle = "Confronto della copertura stratificata per dimensione dell'insieme di predizione",
      x = "Dimensione dell'Insieme di Predizione",
      y = "Copertura Empirica per Gruppo",
      fill = "Metodo"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom") +
    scale_fill_viridis_d(direction = -1)
  
  # Step 1.5: Salva il grafico.
  output_filename_ssc_class <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_ssc_classification.png")
  ggsave(output_filename_ssc_class, plot = plot_ssc_classification, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico SSC di classificazione comparativo salvato in '", output_filename_ssc_class, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi SSC di classificazione. Errore:", e$message, "\n")
})

# --- 2. Analisi Comparativa SSC #2: Metodi di Regressione ---
cat("\nINFO: --- Inizio Analisi Comparativa SSC #2: Regressione ---\n")

tryCatch({
  # Step 2.1: Carica i dati SSC.
  ssc_quantile <- read.csv("results/tables/section2_2_quantile_reg/ssc_BASESEED_RUN.csv")
  ssc_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/ssc_BASESEED_RUN.csv")
  ssc_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/ssc_BASESEED_RUN.csv")
  
  # Step 2.2: Aggiungi identificatori di metodo.
  ssc_quantile$method <- "Reg. Quantile (Sez. 2.2)"
  ssc_scalar_abs$method <- "Incertezza Scalare (Residui)"
  ssc_scalar_stddev$method <- "Incertezza Scalare (Std Dev)"
  
  # Step 2.3: Unisci i dataframe.
  combined_ssc_regression <- dplyr::bind_rows(ssc_quantile, ssc_scalar_abs, ssc_scalar_stddev) # <-- AGGIUNGI
  
  # Step 2.4: Ordina i gruppi di larghezza in modo logico.
  combined_ssc_regression$size_group <- factor(
    combined_ssc_regression$size_group,
    levels = c("Stretto", "Medio", "Largo")
  )
  
  # Step 2.5: Crea il grafico a barre comparativo.
  plot_ssc_regression <- ggplot(
    combined_ssc_regression,
    aes(x = size_group, y = coverage, fill = method)
  ) +
    geom_col(position = "dodge") +
    geom_hline(
      yintercept = TARGET_COVERAGE,
      linetype = "dashed", color = "red", size = 1
    ) +
    annotate(
      "text", x = Inf, y = TARGET_COVERAGE + 0.02,
      label = paste("Target:", TARGET_COVERAGE),
      hjust = 1.1, color = "red", fontface = "bold"
    ) +
    labs(
      title = "Analisi SSC Comparativa (Regressione)",
      subtitle = "Confronto della copertura stratificata per larghezza dell'intervallo",
      x = "Gruppo di Larghezza dell'Intervallo",
      y = "Copertura Empirica per Gruppo",
      fill = "Metodo"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom") +
    scale_fill_brewer(palette = "Set1")
  
  # Step 2.6: Salva il grafico.
  output_filename_ssc_reg <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_ssc_regression.png")
  ggsave(output_filename_ssc_reg, plot = plot_ssc_regression, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico SSC di regressione comparativo salvato in '", output_filename_ssc_reg, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi SSC di regressione. Errore:", e$message, "\n")
})

cat("\nINFO: --- Script di Analisi Comparativa SSC Completato ---\n")