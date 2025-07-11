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

# Step 2: Definisci e carica i pacchetti R richiesti.
all_required_packages <- c("dplyr", "ggplot2", "scales")
check_and_load_packages(all_required_packages)

# Step 3: Definisci le directory per i risultati.
RESULTS_DIR <- "results"
COMPARATIVE_PLOTS_DIR <- file.path(RESULTS_DIR, "plots")
dir.create(COMPARATIVE_PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# Step 4: Definisci parametri comuni per i grafici.
FONT_SIZE_BASE <- 18 # Increased base font size for plots
DODGE_WIDTH <- 0.9 # Overall width allocated for a group of bars (between categories)
BAR_WIDTH <- 0.7 # Individual bar width (smaller than DODGE_WIDTH for space)

# --- 1. Analisi Comparativa #1: Dimensione Insiemi di Predizione (Classificazione) ---

# Step 1.1: Carica i dati grezzi delle dimensioni degli insiemi.
size_basic <- read.csv("results/tables/section1_basic/set_sizes_raw_basic_BASESEED_RUN.csv")
size_adaptive <- read.csv("results/tables/section2_1_adaptive/set_sizes_raw_adaptive_BASESEED_RUN.csv")
size_bayes <- read.csv("results/tables/section2_4_bayes/set_sizes_raw_bayes_BASESEED_RUN.csv")

# Step 1.2: Aggiungi una colonna identificativa del metodo.
size_basic$method <- "Base (Sec. 1)"
size_adaptive$method <- "Adaptive (Sec. 2.1)"
size_bayes$method <- "Bayes (Sec. 2.4)"

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
    position = position_dodge(width = DODGE_WIDTH), # Use DODGE_WIDTH
    width = BAR_WIDTH # Use BAR_WIDTH for individual bars
  ) +
  geom_vline(
    data = size_means,
    aes(xintercept = mean_size, color = method),
    linetype = "dashed", size = 1, show.legend = FALSE
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(
    title = "Comparative Analysis: Prediction Set Size Distribution (Classification)",
    subtitle = "Comparison based on a single BASE_SEED run for each experiment.",
    x = "Prediction Set Size",
    y = "Relative Frequency (%)",
    fill = "Method",
    caption = "Dashed lines indicate the average size for each method."
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
  scale_fill_brewer(palette = "Blues") + # Changed to Blues palette
  scale_color_brewer(palette = "Blues") # Changed to Blues palette

# Step 1.6: Salva il grafico.
output_filename_size <- file.path(COMPARATIVE_PLOTS_DIR, "bar_comparison_set_size_classification.png")
ggsave(output_filename_size, plot = plot_set_size, width = 12, height = 8, dpi = 300)