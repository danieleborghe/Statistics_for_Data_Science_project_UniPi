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

# --- 0. Setup: Sourcing e Librerie ---
# Step 1: Carica le funzioni di utilità.
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Script di Analisi Comparativa ---\n")

# Step 2: Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# Step 3: Definisci le directory per i risultati comparativi.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")

# Step 4: Crea la directory dei risultati.
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I grafici comparativi verranno salvati in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# Step 5: Definisci il livello di significatività alpha per il riferimento visuale.
ALPHA_CONF <- 0.1
TARGET_COVERAGE <- 1 - ALPHA_CONF
FONT_SIZE_BASE <- 14 # Dimensione base del font per i grafici

# --- 1. Analisi Comparativa #1: Metodi di Classificazione ---
cat("\nINFO: --- Inizio Analisi Comparativa #1: Metodi di Classificazione ---\n")

tryCatch({
  # Step 1.1: Carica i dati di copertura.
  coverage_basic <- read.csv("results/tables/section1_basic/coverage_distribution_basic.csv")
  coverage_adaptive <- read.csv("results/tables/section2_1_adaptive/coverage_distribution_adaptive.csv")
  coverage_bayes <- read.csv("results/tables/section2_4_bayes/coverage_distribution_bayes.csv")
  
  # Step 1.2: Aggiungi identificatori di metodo.
  coverage_basic$method <- "Base (Sez. 1)"
  coverage_adaptive$method <- "Adattivo (Sez. 2.1)"
  coverage_bayes$method <- "Bayes (Sez. 2.4)"
  
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
    dplyr::tally(name = "percentage") # Poiché N=100, la conta è già la percentuale.
  
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
      title = "Analisi Comparativa: Copertura Empirica (Classificazione)",
      subtitle = "Confronto tra le distribuzioni di copertura da 100 esecuzioni per metodo.",
      x = "Copertura Empirica",
      y = "Percentuale di Esecuzioni (%)",
      fill = "Metodo",
      caption = "Linea rossa solida: Copertura Target. Linee tratteggiate: Media del metodo."
    ) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_viridis_d(direction = -1) +
    scale_color_viridis_d(direction = -1) # Usa la stessa palette per le linee medie
  
  # Step 1.6: Salva il grafico.
  output_filename_class <- file.path(COMPARATIVE_PLOTS_DIR, "histogram_comparison_classification_coverage.png")
  ggsave(output_filename_class, plot = plot_classification, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico di classificazione comparativo salvato in '", output_filename_class, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi di classificazione. Errore:", e$message, "\n")
})


# --- 2. Analisi Comparativa #2: Metodi di Regressione ---
cat("\nINFO: --- Inizio Analisi Comparativa #2: Metodi di Regressione ---\n")

tryCatch({
  # Step 2.1: Carica i dati di copertura.
  coverage_quantile <- read.csv("results/tables/section2_2_quantile_reg/coverage_width_distribution.csv")
  coverage_scalar <- read.csv("results/tables/section2_3_scalar_uncert/coverage_width_distribution.csv")
  
  # Step 2.2: Aggiungi identificatori di metodo.
  coverage_quantile$method <- "Reg. Quantile (Sez. 2.2)"
  coverage_scalar$method <- "Incertezza Scalare (Sez. 2.3)"
  
  # Step 2.3: Unisci i dataframe.
  combined_regression_coverage <- dplyr::bind_rows(
    coverage_quantile[, c("coverage", "method")],
    coverage_scalar[, c("coverage", "method")]
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
      title = "Analisi Comparativa: Copertura Empirica (Regressione)",
      subtitle = "Confronto tra le distribuzioni di copertura da 100 esecuzioni per metodo.",
      x = "Copertura Empirica",
      y = "Percentuale di Esecuzioni (%)",
      fill = "Metodo",
      caption = "Linea rossa solida: Copertura Target. Linee tratteggiate: Media del metodo."
    ) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom", axis.text.x = element_text(angle = 45, hjust = 1)) +
    scale_fill_brewer(palette = "Set1") +
    scale_color_brewer(palette = "Set1")
  
  # Step 2.6: Salva il grafico.
  output_filename_reg <- file.path(COMPARATIVE_PLOTS_DIR, "histogram_comparison_regression_coverage.png")
  ggsave(output_filename_reg, plot = plot_regression, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico di regressione comparativo salvato in '", output_filename_reg, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi di regressione. Errore:", e$message, "\n")
})

cat("\nINFO: --- Script di Analisi Comparativa Completato ---\n")