# R/run_fsc_comparative_analysis.R
#
# Scopo:
# Esegue analisi comparative della Feature-Stratified Coverage (FSC)
# tra i risultati degli esperimenti di classificazione e regressione.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory.
#   1. Analisi Comparativa FSC #1: Classificazione
#      - Carica i dati FSC per i metodi Base, Adattivo e Bayes.
#      - Unisce i dati, assumendo che la feature sia la stessa (Sepal.Length).
#      - Genera e salva un grafico a barre comparativo.
#   2. Analisi Comparativa FSC #2: Regressione
#      - Carica i dati FSC per Regressione Quantile e Incertezza Scalare.
#      - Unisce i dati.
#      - Genera e salva un grafico a barre comparativo.

# --- 0. Setup: Sourcing e Librerie ---
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Script di Analisi Comparativa FSC ---\n")

# Step 2: Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# Step 3: Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I grafici comparativi FSC verranno salvati in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# Step 4: Definisci parametri comuni per i grafici.
ALPHA_CONF <- 0.1
TARGET_COVERAGE <- 1 - ALPHA_CONF
FONT_SIZE_BASE <- 14

# --- 1. Analisi Comparativa FSC #1: Metodi di Classificazione ---
cat("\nINFO: --- Inizio Analisi Comparativa FSC #1: Classificazione ---\n")

tryCatch({
  # Step 1.1: Carica i dati FSC per ogni metodo.
  # NOTA: Si assume che la feature usata sia 'Sepal.Length' per tutti, come da richiesta.
  fsc_basic <- read.csv("results/tables/section1_basic/fsc_by_Sepal.Length_basic_BASESEED_RUN.csv")
  # Per il metodo adattivo, carichiamo il file ma lo tratteremo come se fosse basato su Sepal.Length.
  fsc_adaptive <- read.csv("results/tables/section2_1_adaptive/fsc_by_Sepal.Length_adaptive_BASESEED_RUN.csv")
  fsc_bayes <- read.csv("results/tables/section2_4_bayes/fsc_by_Sepal.Length_bayes_BASESEED_RUN.csv")
  
  # Step 1.2: Aggiungi una colonna identificativa del metodo.
  fsc_basic$method <- "Base (Sez. 1)"
  fsc_adaptive$method <- "Adattivo (Sez. 2.1)"
  fsc_bayes$method <- "Bayes (Sez. 2.4)"
  
  # Step 1.3: Unisci i dataframe in uno solo.
  combined_fsc_classification <- dplyr::bind_rows(
    fsc_basic, fsc_adaptive, fsc_bayes
  )
  
  # Step 1.4: Crea il grafico a barre comparativo.
  plot_fsc_classification <- ggplot(
    combined_fsc_classification,
    aes(x = feature_group, y = coverage, fill = method)
  ) +
    geom_col(position = "dodge") +
    geom_hline(
      yintercept = TARGET_COVERAGE,
      linetype = "dashed", color = "red", size = 1
    ) +
    # Aggiunge un'annotazione per la linea target.
    annotate(
      "text", x = Inf, y = TARGET_COVERAGE + 0.02,
      label = paste("Target:", TARGET_COVERAGE),
      hjust = 1.1, color = "red", fontface = "bold"
    ) +
    labs(
      title = "Analisi FSC Comparativa (Classificazione)",
      subtitle = "Confronto della copertura stratificata per gruppi di 'Sepal.Length'",
      x = "Gruppo di Feature (Sepal.Length)",
      y = "Copertura Empirica per Gruppo",
      fill = "Metodo"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 25, vjust = 0.6)
    ) +
    scale_fill_viridis_d(direction = -1)
  
  # Step 1.5: Salva il grafico.
  output_filename_fsc_class <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_fsc_classification.png")
  ggsave(output_filename_fsc_class, plot = plot_fsc_classification, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico FSC di classificazione comparativo salvato in '", output_filename_fsc_class, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi FSC di classificazione. Errore:", e$message, "\n")
})

# --- 2. Analisi Comparativa FSC #2: Metodi di Regressione ---
cat("\nINFO: --- Inizio Analisi Comparativa FSC #2: Regressione ---\n")

tryCatch({
  # Step 2.1: Carica i dati FSC.
  fsc_quantile <- read.csv("results/tables/section2_2_quantile_reg/fsc_by_Sepal.Length_BASESEED_RUN.csv")
  fsc_scalar <- read.csv("results/tables/section2_3_scalar_uncert/fsc_by_Sepal.Length_BASESEED_RUN.csv")
  
  # Step 2.2: Aggiungi identificatori di metodo.
  fsc_quantile$method <- "Reg. Quantile (Sez. 2.2)"
  fsc_scalar$method <- "Incertezza Scalare (Sez. 2.3)"
  
  # Step 2.3: Unisci i dataframe.
  combined_fsc_regression <- dplyr::bind_rows(fsc_quantile, fsc_scalar)
  
  # Step 2.4: Crea il grafico a barre comparativo.
  plot_fsc_regression <- ggplot(
    combined_fsc_regression,
    aes(x = feature_group, y = coverage, fill = method)
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
      title = "Analisi FSC Comparativa (Regressione)",
      subtitle = "Confronto della copertura stratificata per gruppi di 'Sepal.Length'",
      x = "Gruppo di Feature (Sepal.Length)",
      y = "Copertura Empirica per Gruppo",
      fill = "Metodo"
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 25, vjust = 0.6)
    ) +
    scale_fill_brewer(palette = "Set1")
  
  # Step 2.5: Salva il grafico.
  output_filename_fsc_reg <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_fsc_regression.png")
  ggsave(output_filename_fsc_reg, plot = plot_fsc_regression, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico FSC di regressione comparativo salvato in '", output_filename_fsc_reg, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi FSC di regressione. Errore:", e$message, "\n")
})

cat("\nINFO: --- Script di Analisi Comparativa FSC Completato ---\n")