# R/sec2_1_run_experiment.R
#
# Scopo:
# Questo script esegue l'esperimento dei Set di Predizione Adattivi (Sezione 2.1 dell'articolo).
# Include esecuzioni multiple per la generazione dell'istogramma di copertura marginale
# e una singola esecuzione dettagliata di valutazione utilizzando BASE_SEED.
#
# Struttura:
#   0. Setup: Creazione Directory Risultati, Caricamento Librerie
#   1. Impostazioni Esperimento: Definizione di parametri come ALPHA, N_RUNS, seed, proporzioni dei dati.
#   2. Esecuzioni Multiple per la Distribuzione della Copertura Marginale:
#      2.1 Caricamento Completo del Dataset (una volta)
#      2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo
#      2.3 Avvio Ciclo N_RUNS
#          - Iterazione X: Setup (Seed, Logging)
#          - Iterazione X: Divisione Dati
#          - Iterazione X: Addestramento Modello
#          - Iterazione X: Calibrazione (Punteggi Adattivi)
#          - Iterazione X: Predizione (Set Adattivi)
#          - Iterazione X: Memorizzazione Copertura Empirica
#   3. Analisi e Salvataggio della Distribuzione della Copertura Marginale da N_RUNS:
#      3.1 Calcolo e Stampa delle Statistiche Riepilogative (Media, DS)
#      3.2 Salvataggio dei Valori Grezzi di Copertura su CSV
#      3.3 Tracciamento e Salvataggio dell'Istogramma delle Coperture
#   4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED):
#      4.1 Setup per la Singola Esecuzione Dettagliata (Imposta BASE_SEED)
#      4.2 Divisione Dati per la Singola Esecuzione
#      4.3 Addestramento Modello per la Singola Esecuzione
#      4.4 Calibrazione per la Singola Esecuzione (Punteggi Adattivi)
#      4.5 Predizione per la Singola Esecuzione (Set Adattivi)
#      4.6 Salvataggio CSV Dettagliato delle Predizioni per la Singola Esecuzione
#      4.7 Valutazione e Salvataggio di Tutte le Metriche per la Singola Esecuzione:
#          4.7.1 Copertura Marginale a Singola Esecuzione
#          4.7.2 Dimensioni Insiemi a Singola Esecuzione (Riepilogo, Grezzi, Istogramma)
#          4.7.3 FSC a Singola Esecuzione (Tabella, Grafico)
#          4.7.4 SSC a Singola Esecuzione (Tabella, Grafico)

# Step 0: Sourcing degli script R condivisi.
# Questo carica le funzioni di utilità e predizione necessarie per l'esperimento.
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")

cat("INFO: --- Avvio Esperimento: Set di Predizione Adattivi (Sezione 2.1) con Esecuzioni Multiple per la Copertura ---\n")

# --- 0. Setup: Caricamento Librerie ---
# Step 1: Definisci tutti i pacchetti R richiesti per questo script.
all_required_packages <- c("e1071", "dplyr", "ggplot2")
# Step 2: Controlla, installa (se necessario) e carica i pacchetti definiti.
check_and_load_packages(all_required_packages)

# --- 0. Setup: Creazione Directory Risultati ---
# Step 1: Definisci i nomi delle directory per salvare i risultati.
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section2_1_adaptive"
PLOTS_DIR <- file.path(RESULTS_DIR, "plots", METHOD_NAME_SUFFIX)
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)

# Step 2: Crea le directory dei risultati.
# `showWarnings = FALSE` evita avvisi se le directory esistono già.
# `recursive = TRUE` crea le sottodirectory necessarie.
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
cat(paste0("INFO: I risultati verranno salvati in '", RESULTS_DIR, "/ (sottocartelle: ", METHOD_NAME_SUFFIX, ")'\n"))

# --- 1. Impostazioni Esperimento ---
cat("INFO: Definizione impostazioni esperimento per i Set di Predizione Adattivi...\n")
# Step 1: Definisci il seed base per la riproducibilità.
# È un seed diverso per questo specifico esperimento.
BASE_SEED <- 43
# Step 2: Definisci il livello di significatività (alpha) per la predizione conforme.
ALPHA_CONF <- 0.1
# Step 3: Definisci la proporzione dei dati per l'addestramento del modello SVM sottostante.
PROP_TRAIN <- 0.7
# Step 4: Definisci la proporzione dei dati per il set di calibrazione conforme.
PROP_CALIB <- 0.1
# Step 5: Definisci il numero di esecuzioni per generare la distribuzione della copertura marginale.
N_RUNS <- 100
cat(paste0("INFO: Impostazioni: ALPHA_CONF=", ALPHA_CONF, ", N_RUNS_FOR_COVERAGE_HISTOGRAM=", N_RUNS, "\n"))
cat(paste0("INFO: PROP_TRAIN=", PROP_TRAIN, ", PROP_CALIB=", PROP_CALIB, "\n"))
cat(paste0("INFO: BASE_SEED per l'analisi dettagliata a singola esecuzione: ", BASE_SEED, "\n"))

# --- 2. Esecuzioni Multiple per la Distribuzione della Copertura Marginale ---
# Step 1: Inizializza un vettore numerico per memorizzare le coperture empiriche di ogni esecuzione.
all_empirical_coverages_adaptive <- numeric(N_RUNS)
cat(paste0("INFO: Avvio ", N_RUNS, " esecuzioni per raccogliere dati di copertura marginale (Metodo Adattivo)...\n"))

# --- 2.1 Caricamento Completo del Dataset (una volta) ---
# Step 1: Carica il dataset Iris completo, preparato per la classificazione.
iris_data_full <- load_iris_for_classification()
n_total <- nrow(iris_data_full) # Ottieni il numero totale di campioni nel dataset.

# --- 2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo ---
# Step 1: Calcola il numero di campioni per i set di addestramento, calibrazione e test.
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
# Assicura che tutti i dati siano utilizzati per il set di test.
n_test_loop <- n_total - n_train_loop - n_calib_loop

# --- 2.3 Avvio Ciclo N_RUNS per la Copertura Marginale ---
for (run_iter in 1:N_RUNS) {
  # ------ Iterazione X: Setup (Seed, Logging) ------
  # Step 1: Imposta un seed unico per ogni iterazione per garantire la riproducibilità di ogni divisione.
  current_run_seed <- BASE_SEED + run_iter
  set.seed(current_run_seed)
  
  # Step 2: Stampa un messaggio di progresso per le esecuzioni.
  if (run_iter == 1 || run_iter %% (N_RUNS / 10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: --- Esecuzione ", run_iter, "/", N_RUNS, " (Seed: ", current_run_seed, ") (Adattivo) ---\n"))
  }
  
  # ------ Iterazione X: Divisione Dati ------
  # Step 1: Rimescola gli indici del dataset per creare una nuova divisione per l'esecuzione corrente.
  shuffled_indices_for_run <- sample(n_total)
  current_data_for_run <- iris_data_full[shuffled_indices_for_run, ]
  # Step 2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
  train_df_iter <- current_data_for_run[1:n_train_loop, ]
  calib_df_iter <- current_data_for_run[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
  test_df_iter <- current_data_for_run[(n_train_loop + n_calib_loop + 1):n_total, ]
  
  # ------ Iterazione X: Addestramento Modello ------
  # Step 1: Definisci la formula SVM (Species come variabile target, '.' per tutte le altre feature).
  svm_formula <- Species ~ .
  # Step 2: Addestra il modello SVM sui dati di addestramento dell'iterazione corrente.
  svm_model_iter <- train_svm_model(svm_formula, train_df_iter)
  
  # ------ Iterazione X: Calibrazione (Punteggi Adattivi) ------
  # Step 1: Predici le probabilità di classe per il set di calibrazione.
  calib_probs_iter <- predict_svm_probabilities(svm_model_iter, calib_df_iter)
  # Step 2: Estrai le vere etichette dal set di calibrazione.
  calib_true_labels_iter <- calib_df_iter$Species
  # Step 3: Calcola i punteggi di non-conformità adattivi per il set di calibrazione.
  non_conf_scores_iter <- get_non_conformity_scores_adaptive(calib_probs_iter, calib_true_labels_iter)
  # Step 4: Calcola il valore $q_{hat}$ basato sui punteggi di non-conformità del set di calibrazione.
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iterazione X: Predizione (Set Adattivi) ------
  # Step 1: Predici le probabilità di classe per il set di test.
  test_probs_iter <- predict_svm_probabilities(svm_model_iter, test_df_iter)
  # Step 2: Estrai le vere etichette dal set di test.
  test_true_labels_iter <- test_df_iter$Species
  # Step 3: Crea gli insiemi di predizione adattivi per il set di test.
  prediction_sets_iter <- create_prediction_sets_adaptive(test_probs_iter, q_hat_iter)
  
  # ------ Iterazione X: Memorizzazione Copertura Empirica ------
  # Step 1: Calcola la copertura empirica per l'esecuzione corrente.
  current_coverage <- calculate_empirical_coverage(prediction_sets_iter, test_true_labels_iter)
  # Step 2: Memorizza la copertura calcolata.
  all_empirical_coverages_adaptive[run_iter] <- current_coverage
  
  # Step 3: Stampa un messaggio di progresso con la copertura dell'esecuzione corrente.
  if (run_iter == 1 || run_iter %% (N_RUNS / 10) == 0 || run_iter == N_RUNS) {
    cat(paste0("INFO: Copertura Esecuzione ", run_iter, " (Adattivo): ", round(current_coverage, 3), "\n"))
  }
}
cat("INFO: Tutte le ", N_RUNS, " esecuzioni per la copertura marginale completate (Metodo Adattivo).\n")

# --- 3. Analisi e Salvataggio della Distribuzione della Copertura Marginale da N_RUNS ---
cat("\nINFO: --- Analisi Distribuzione Copertura Marginale (Metodo Adattivo) ---\n")
# --- 3.1 Calcolo e Stampa delle Statistiche Riepilogative ---
# Step 1: Calcola la media della copertura empirica su tutte le esecuzioni.
mean_empirical_coverage_adaptive <- mean(all_empirical_coverages_adaptive, na.rm = TRUE)
# Step 2: Calcola la deviazione standard della copertura empirica.
sd_empirical_coverage_adaptive <- sd(all_empirical_coverages_adaptive, na.rm = TRUE)
cat(paste0("RISULTATO: Copertura Marginale Empirica Media (Adattivo) su ", N_RUNS, " esecuzioni: ", round(mean_empirical_coverage_adaptive, 3), "\n"))
cat(paste0("RISULTATO: Dev. Std. della Copertura Marginale Empirica (Adattivo): ", round(sd_empirical_coverage_adaptive, 3), "\n"))
cat(paste0("RISULTATO: Copertura Target era >= ", 1 - ALPHA_CONF, "\n"))

# --- 3.2 Salvataggio dei Valori Grezzi di Copertura su CSV ---
# Step 1: Crea un data frame con i risultati della copertura per ogni esecuzione.
coverage_distribution_df_adaptive <- data.frame(run_iteration = 1:N_RUNS, empirical_coverage = all_empirical_coverages_adaptive)
# Step 2: Definisci il percorso del file CSV.
coverage_dist_filename <- file.path(TABLES_DIR, "coverage_distribution_adaptive.csv")
# Step 3: Salva il data frame in un file CSV.
write.csv(coverage_distribution_df_adaptive, coverage_dist_filename, row.names = FALSE)
cat(paste0("INFO: Distribuzione di ", N_RUNS, " coperture empiriche (Adattivo) salvata in '", coverage_dist_filename, "'\n"))

# --- 3.3 Tracciamento e Salvataggio dell'Istogramma delle Coperture ---
# Step 1: Definisci il percorso del file PNG per l'istogramma.
coverage_hist_filename <- file.path(PLOTS_DIR, "histogram_marginal_coverage_adaptive.png")
# Step 2: Avvia il dispositivo grafico PNG.
png(coverage_hist_filename, width = 800, height = 600)
# Step 3: Traccia l'istogramma delle coperture.
plot_coverage_histogram(all_empirical_coverages_adaptive,
                        alpha_conf = ALPHA_CONF,
                        n_runs = N_RUNS,
                        method_name = "Set di Predizione Adattivi"
)
# Step 4: Chiudi il dispositivo grafico, salvando l'immagine.
dev.off()
cat(paste0("INFO: Istogramma della copertura marginale (Adattivo) salvato in '", coverage_hist_filename, "'\n"))

# --- 4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED per la riproducibilità) ---
cat("\nINFO: --- Valutazione Dettagliata per una Singola Esecuzione Riproducibile (Metodo Adattivo - utilizzando BASE_SEED) ---\n")
# --- 4.1 Setup per Singola Esecuzione Dettagliata (Imposta BASE_SEED) ---
# Step 1: Imposta il seed base per garantire la riproducibilità di questa singola esecuzione.
set.seed(BASE_SEED)
cat(paste0("INFO: Esecuzione singola divisione dati con BASE_SEED = ", BASE_SEED, " per analisi dettagliata (Adattivo)...\n"))

# --- 4.2 Divisione Dati per Singola Esecuzione ---
# Step 1: Rimescola gli indici del dataset basandosi sul BASE_SEED.
single_run_shuffled_indices <- sample(n_total)
single_run_data <- iris_data_full[single_run_shuffled_indices, ]
# Step 2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
single_run_train_df <- single_run_data[1:n_train_loop, ]
single_run_calib_df <- single_run_data[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
single_run_test_df <- single_run_data[(n_train_loop + n_calib_loop + 1):n_total, ]
cat(paste0("INFO: Dimensioni dataset singola esecuzione (Adattivo): Addestramento=", nrow(single_run_train_df),
           ", Calibrazione=", nrow(single_run_calib_df), ", Test=", nrow(single_run_test_df), "\n"
))

# --- 4.3 Addestramento Modello per Singola Esecuzione ---
# Step 1: Addestra il modello SVM sul set di addestramento della singola esecuzione.
# La formula SVM è stata definita in precedenza.
single_run_svm_model <- train_svm_model(svm_formula, single_run_train_df)

# --- 4.4 Calibrazione per Singola Esecuzione (Punteggi Adattivi) ---
# Step 1: Predici le probabilità di classe per il set di calibrazione.
single_run_calib_probs <- predict_svm_probabilities(single_run_svm_model, single_run_calib_df)
# Step 2: Estrai le vere etichette dal set di calibrazione.
single_run_calib_true_labels <- single_run_calib_df$Species
# Step 3: Calcola i punteggi di non-conformità adattivi.
single_run_non_conf_scores <- get_non_conformity_scores_adaptive(single_run_calib_probs, single_run_calib_true_labels)
# Step 4: Calcola $q_{hat}$ per la singola esecuzione.
single_run_q_hat_adaptive <- calculate_q_hat(single_run_non_conf_scores, ALPHA_CONF, n_calib = nrow(single_run_calib_df))
cat(paste0("INFO: Calibrazione singola esecuzione completata. q_hat (adattivo) = ", round(single_run_q_hat_adaptive, 4), "\n"))

# --- 4.5 Predizione per Singola Esecuzione (Set Adattivi) ---
# Step 1: Predici le probabilità di classe per il set di test.
single_run_test_probs <- predict_svm_probabilities(single_run_svm_model, single_run_test_df)
# Step 2: Estrai le vere etichette dal set di test.
single_run_test_true_labels <- single_run_test_df$Species
# Step 3: Crea gli insiemi di predizione adattivi per il set di test.
single_run_prediction_sets <- create_prediction_sets_adaptive(single_run_test_probs, single_run_q_hat_adaptive)

# --- 4.6 Salvataggio CSV Dettagliato delle Predizioni per Singola Esecuzione ---
# Step 1: Salva le predizioni di test dettagliate in un file CSV.
save_detailed_test_predictions(
  test_true_labels = single_run_test_true_labels,
  prediction_sets_list = single_run_prediction_sets,
  test_probs_matrix = single_run_test_probs,
  output_directory = TABLES_DIR,
  base_filename = "detailed_test_predictions_adaptive_BASESEED_RUN.csv"
)

# --- 4.7 Valutazione e Salvataggio di Tutte le Metriche per Singola Esecuzione ---
cat("\nINFO: --- Risultati Valutazione per Singola Esecuzione BASE_SEED (Metodo Adattivo) ---\n")
# ---- 4.7.1 Copertura Marginale a Singola Esecuzione ----
# Step 1: Calcola la copertura empirica per la singola esecuzione.
single_run_empirical_cov <- calculate_empirical_coverage(single_run_prediction_sets, single_run_test_true_labels)
cat(paste0("RISULTATO (Esecuzione BASE_SEED): Copertura Marginale Empirica (Adattivo): ", round(single_run_empirical_cov, 3),
           " (Target >= ", 1 - ALPHA_CONF, ")\n"
))
# Step 2: Crea un data frame riassuntivo della copertura.
single_run_coverage_summary <- data.frame(
  method = "Adaptive_Section2.1_BASESEED_Run", alpha = ALPHA_CONF,
  target_coverage = 1 - ALPHA_CONF, empirical_coverage = single_run_empirical_cov,
  q_hat = single_run_q_hat_adaptive
)
# Step 3: Salva il riassunto della copertura in un file CSV.
write.csv(single_run_coverage_summary, file.path(TABLES_DIR, "coverage_summary_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Riepilogo copertura marginale per Esecuzione BASE_SEED (Adattivo) salvato in '", TABLES_DIR, "/coverage_summary_adaptive_BASESEED_RUN.csv'\n"))

# ---- 4.7.2 Dimensioni Insiemi a Singola Esecuzione (Riepilogo, Grezzi, Istogramma) ----
# Step 1: Ottieni le dimensioni degli insiemi di predizione.
single_run_set_sizes <- get_set_sizes(single_run_prediction_sets)
cat("RISULTATO (Esecuzione BASE_SEED): Statistiche Dimensioni Insieme (Adattivo):\n")
print(summary(single_run_set_sizes))
# Step 2: Calcola la dimensione media degli insiemi.
single_run_avg_set_size <- mean(single_run_set_sizes, na.rm = TRUE)
cat(paste0("RISULTATO (Esecuzione BASE_SEED): Dimensione media insieme (Adattivo): ", round(single_run_avg_set_size, 3), "\n"))
# Step 3: Crea un data frame riassuntivo delle dimensioni degli insiemi.
single_run_set_size_summary_df <- data.frame(
  Min = min(single_run_set_sizes, na.rm = TRUE), Q1 = quantile(single_run_set_sizes, 0.25, na.rm = TRUE, names = FALSE),
  Median = median(single_run_set_sizes, na.rm = TRUE), Mean = single_run_avg_set_size,
  Q3 = quantile(single_run_set_sizes, 0.75, na.rm = TRUE, names = FALSE), Max = max(single_run_set_sizes, na.rm = TRUE)
)
# Step 4: Salva il riassunto e i valori grezzi delle dimensioni degli insiemi in file CSV.
write.csv(single_run_set_size_summary_df, file.path(TABLES_DIR, "set_size_summary_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
write.csv(data.frame(set_size = single_run_set_sizes), file.path(TABLES_DIR, "set_sizes_raw_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
cat(paste0("INFO: Riepilogo e dimensioni grezze degli insiemi per Esecuzione BASE_SEED (Adattivo) salvati in '", TABLES_DIR, "/'\n"))

# Step 5: Definisci il percorso del file PNG per l'istogramma delle dimensioni degli insiemi.
single_run_plot_filename_set_size <- file.path(PLOTS_DIR, "histogram_set_sizes_adaptive_BASESEED_RUN.png")
# Step 6: Avvia il dispositivo grafico PNG.
png(single_run_plot_filename_set_size, width = 800, height = 600)
# Step 7: Traccia l'istogramma delle dimensioni degli insiemi.
plot_set_size_histogram(single_run_set_sizes, main_title = "Dimensioni Insiemi (Conforme Adattivo - Iris - Esecuzione BASE_SEED)")
# Step 8: Chiudi il dispositivo grafico, salvando l'immagine.
dev.off()
cat(paste0("INFO: Istogramma dimensioni insiemi per Esecuzione BASE_SEED (Adattivo) salvato in '", single_run_plot_filename_set_size, "'\n"))

# ---- 4.7.3 FSC a Singola Esecuzione (Tabella, Grafico) ----
# Step 1: Definisci il nome della feature per l'analisi FSC (Feature-conditional Coverage).
single_run_fsc_feature_name <- "Petal.Width"
# Step 2: Calcola i risultati FSC.
single_run_fsc_results <- calculate_fsc(single_run_prediction_sets, single_run_test_true_labels,
                                        single_run_test_df[[single_run_fsc_feature_name]], feature_name = single_run_fsc_feature_name,
                                        num_bins_for_continuous = 4
)
cat(paste0("RISULTATO (Esecuzione BASE_SEED): FSC (Adattivo - per ", single_run_fsc_feature_name, "):\n"))
if (!is.na(single_run_fsc_results$min_coverage)) {
  # Step 3: Stampa la copertura minima FSC e la copertura per gruppo.
  cat(paste0("  Copertura FSC minima: ", round(single_run_fsc_results$min_coverage, 3), "\n"))
  print(single_run_fsc_results$coverage_by_group)
  # Step 4: Salva la tabella dei risultati FSC in un file CSV.
  write.csv(single_run_fsc_results$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_", single_run_fsc_feature_name, "_adaptive_BASESEED_RUN.csv")), row.names = FALSE)
  cat(paste0("INFO: Tabella risultati FSC per Esecuzione BASE_SEED (Adattivo) salvata in '", TABLES_DIR, "/fsc_by_", single_run_fsc_feature_name, "_adaptive_BASESEED_RUN.csv'\n"))
  
  # Step 5: Definisci il percorso del file PNG per il grafico FSC.
  single_run_plot_filename_fsc <- file.path(PLOTS_DIR, paste0("plot_fsc_", single_run_fsc_feature_name, "_adaptive_BASESEED_RUN.png"))
  # Step 6: Se ggplot2 è disponibile e ci sono dati, traccia e salva il grafico FSC.
  if (requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_fsc_results$coverage_by_group) > 0) {
    png(single_run_plot_filename_fsc, width = 800, height = 600)
    plot_conditional_coverage(single_run_fsc_results$coverage_by_group, "feature_group", "coverage",
                              1 - ALPHA_CONF, paste0("FSC (Adattivo - ", single_run_fsc_feature_name, " - Iris - Esecuzione BASE_SEED)")
    )
    dev.off()
    cat(paste0("INFO: Grafico FSC per Esecuzione BASE_SEED (Adattivo) salvato in '", single_run_plot_filename_fsc, "'\n"))
  } else {
    cat("INFO: ggplot2 non disponibile o nessun dato per il grafico FSC.\n")
  }
} else {
  cat("  FSC (Esecuzione BASE_SEED - Adattivo): NA o nessun gruppo.\n")
}

# ---- 4.7.4 SSC a Singola Esecuzione (Tabella, Grafico) ----
# Step 1: Calcola il numero di bin per l'analisi SSC (Set-size conditional Coverage).
single_run_ssc_bins <- max(1, min(length(unique(single_run_set_sizes)), length(levels(iris_data_full$Species))))
# Step 2: Calcola i risultati SSC.
single_run_ssc_results <- calculate_ssc(single_run_prediction_sets, single_run_test_true_labels, num_bins_for_size = single_run_ssc_bins)
cat("RISULTATO (Esecuzione BASE_SEED): SSC (Adattivo):\n")
if (!is.na(single_run_ssc_results$min_coverage)) {
  # Step 3: Stampa la copertura minima SSC e la copertura per gruppo.
  cat(paste0("  Copertura SSC minima: ", round(single_run_ssc_results$min_coverage, 3), "\n"))
  print(single_run_ssc_results$coverage_by_group)
  # Step 4: Salva la tabella dei risultati SSC in un file CSV.
  write.csv(single_run_ssc_results$coverage_by_group, file.path(TABLES_DIR, "ssc_adaptive_BASESEED_RUN.csv"), row.names = FALSE)
  cat(paste0("INFO: Tabella risultati SSC per Esecuzione BASE_SEED (Adattivo) salvata in '", TABLES_DIR, "/ssc_adaptive_BASESEED_RUN.csv'\n"))
  
  # Step 5: Definisci il percorso del file PNG per il grafico SSC.
  single_run_plot_filename_ssc <- file.path(PLOTS_DIR, "plot_ssc_adaptive_BASESEED_RUN.png")
  # Step 6: Se ggplot2 è disponibile e ci sono dati, traccia e salva il grafico SSC.
  if (requireNamespace("ggplot2", quietly = TRUE) && nrow(single_run_ssc_results$coverage_by_group) > 0) {
    png(single_run_plot_filename_ssc, width = 800, height = 600)
    plot_conditional_coverage(single_run_ssc_results$coverage_by_group, "size_group", "coverage",
                              1 - ALPHA_CONF, "SSC (Adattivo - Iris - Esecuzione BASE_SEED)"
    )
    dev.off()
    cat(paste0("INFO: Grafico SSC per Esecuzione BASE_SEED (Adattivo) salvato in '", single_run_plot_filename_ssc, "'\n"))
  } else {
    cat("INFO: ggplot2 non disponibile o nessun dato per il grafico SSC.\n")
  }
} else {
  cat("  SSC (Esecuzione BASE_SEED - Adattivo): NA o nessun gruppo.\n")
}

cat("INFO: --- Fine Esperimento Set di Predizione Adattivi ---\n")
