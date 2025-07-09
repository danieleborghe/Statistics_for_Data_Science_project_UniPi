# R/run_size_width_comparative_analysis.R
#
# Scopo:
# Esegue analisi comparative sulla dimensione degli insiemi di predizione (classificazione)
# e sulla larghezza degli intervalli (regressione) tra i vari esperimenti.
#
# Struttura:
#   0. Setup: Caricamento librerie e creazione directory.
#   1. Analisi Comparativa #1: Dimensione Insiemi (Classificazione)
#      - Carica i dati grezzi delle dimensioni degli insiemi.
#      - Calcola le medie e le frequenze.
#      - Genera e salva un grafico a barre comparativo.
#   2. Analisi Comparativa #2: Larghezza Intervalli (Regressione)
#      - Carica i dati grezzi delle larghezze degli intervalli.
#      - Calcola le medie.
#      - Genera e salva un grafico di densità comparativo.

# --- 0. Setup: Sourcing e Librerie ---
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Script di Analisi Comparativa su Dimensioni/Larghezze ---\n")

# Step 2: Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2", "scales")
check_and_load_packages(all_required_packages)

# Step 3: Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots", "comparative_analysis")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I grafici comparativi verranno salvati in '", COMPARATIVE_PLOTS_DIR, "'\n"))

# Step 4: Definisci parametri comuni per i grafici.
FONT_SIZE_BASE <- 14

# --- 1. Analisi Comparativa #1: Dimensione Insiemi di Predizione (Classificazione) ---
cat("\nINFO: --- Inizio Analisi #1: Dimensione Insiemi (Classificazione) ---\n")

tryCatch({
  # Step 1.1: Carica i dati grezzi delle dimensioni degli insiemi.
  size_basic <- read.csv("results/tables/section1_basic/set_sizes_raw_basic_BASESEED_RUN.csv")
  size_adaptive <- read.csv("results/tables/section2_1_adaptive/set_sizes_raw_adaptive_BASESEED_RUN.csv")
  size_bayes <- read.csv("results/tables/section2_4_bayes/set_sizes_raw_bayes_BASESEED_RUN.csv")
  
  # Step 1.2: Aggiungi una colonna identificativa del metodo.
  size_basic$method <- "Base (Sez. 1)"
  size_adaptive$method <- "Adattivo (Sez. 2.1)"
  size_bayes$method <- "Bayes (Sez. 2.4)"
  
  # Step 1.3: Unisci i dataframe.
  combined_sizes <- dplyr::bind_rows(size_basic, size_adaptive, size_bayes)
  
  # Step 1.4: Calcola la frequenza (in percentuale) per ogni dimensione e metodo.
  size_freq <- combined_sizes %>%
    dplyr::group_by(method, set_size) %>%
    dplyr::tally() %>%
    dplyr::group_by(method) %>%
    dplyr::mutate(percentage = n / sum(n))
  
  # Calcola le medie
  size_means <- combined_sizes %>%
    dplyr::group_by(method) %>%
    dplyr::summarise(mean_size = mean(set_size, na.rm = TRUE))
  
  # Step 1.5: Crea il grafico a barre comparativo.
  plot_set_size <- ggplot() +
    geom_col(
      data = size_freq,
      aes(x = factor(set_size), y = percentage, fill = method),
      position = "dodge"
    ) +
    geom_vline(
      data = size_means,
      aes(xintercept = mean_size, color = method),
      linetype = "dashed", size = 1, show.legend = FALSE
    ) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "Analisi Comparativa: Distribuzione Dimensioni Insiemi (Classificazione)",
      subtitle = "Confronto basato sulla singola esecuzione con BASE_SEED di ogni esperimento.",
      x = "Dimensione dell'Insieme di Predizione",
      y = "Frequenza Relativa (%)",
      fill = "Metodo",
      caption = "Le linee tratteggiate indicano la dimensione media per ciascun metodo."
    ) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom") +
    scale_fill_viridis_d(direction = -1) +
    scale_color_viridis_d(direction = -1)
  
  # Step 1.6: Salva il grafico.
  output_filename_size <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_set_size_classification.png")
  ggsave(output_filename_size, plot = plot_set_size, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico dimensioni insiemi (classificazione) salvato in '", output_filename_size, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi delle dimensioni insiemi. Errore:", e$message, "\n")
})


# --- 2. Analisi Comparativa #2: Larghezza Intervalli (Regressione) ---
cat("\nINFO: --- Inizio Analisi #2: Larghezza Intervalli (Regressione) ---\n")

tryCatch({
  # Step 2.1: Carica i dati grezzi delle larghezze.
  width_quantile <- read.csv("results/tables/section2_2_quantile_reg/widths_raw_BASESEED_RUN.csv")
  width_scalar_abs <- read.csv("results/tables/section2_3_scalar_uncert/widths_raw_BASESEED_RUN.csv")
  width_scalar_stddev <- read.csv("results/tables/section2_3_stddev_uncert/widths_raw_BASESEED_RUN.csv")
  
  # Step 2.2: Aggiungi identificatori di metodo.
  width_quantile$method <- "Reg. Quantile (Sez. 2.2)"
  width_scalar_abs$method <- "Incertezza Scalare (Residui)"
  width_scalar_stddev$method <- "Incertezza Scalare (Std Dev)"
  
  # Step 2.3: Unisci i dataframe.
  combined_widths <- dplyr::bind_rows(width_quantile, width_scalar_abs, width_scalar_stddev) # <-- AGGIUNGI
  
  # Step 2.4: Calcola le medie
  width_means <- combined_widths %>%
    dplyr::group_by(method) %>%
    dplyr::summarise(mean_width = mean(IntervalWidth, na.rm = TRUE))
  
  # Step 2.5: Crea il grafico di densità comparativo.
  plot_interval_width <- ggplot(combined_widths, aes(x = IntervalWidth, fill = method)) +
    geom_density(alpha = 0.7) +
    geom_vline(
      data = width_means,
      aes(xintercept = mean_width, color = method),
      linetype = "dashed", size = 1.2, show.legend = FALSE
    ) +
    labs(
      title = "Analisi Comparativa: Distribuzione Larghezza Intervalli (Regressione)",
      subtitle = "Confronto basato sulla singola esecuzione con BASE_SEED di ogni esperimento.",
      x = "Larghezza dell'Intervallo di Predizione",
      y = "Densità",
      fill = "Metodo",
      caption = "Le linee tratteggiate indicano la larghezza media per ciascun metodo."
    ) +
    theme_light(base_size = FONT_SIZE_BASE) +
    theme(legend.position = "bottom") +
    scale_fill_brewer(palette = "Set1") +
    scale_color_brewer(palette = "Set1")
  
  # Step 2.6: Salva il grafico.
  output_filename_width <- file.path(COMPARATIVE_PLOTS_DIR, "density_comparison_interval_width_regression.png")
  ggsave(output_filename_width, plot = plot_interval_width, width = 12, height = 8, dpi = 300)
  cat(paste0("INFO: Grafico larghezza intervalli (regressione) salvato in '", output_filename_width, "'\n"))
  
}, error = function(e) {
  cat("ERRORE: Impossibile completare l'analisi delle larghezze intervalli. Errore:", e$message, "\n")
})

cat("\nINFO: --- Script di Analisi Comparativa Completato ---\n")