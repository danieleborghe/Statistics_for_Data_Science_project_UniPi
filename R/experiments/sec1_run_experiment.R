# R/sec1_run_experiment.R
#
# Scopo:
# Questo script esegue l'esperimento di Predizione Conforme di Base (Sezione 1 dell'articolo).
# Include esecuzioni multiple per generare una distribuzione dei valori di copertura marginale
# Valutazioni dettagliate (dimensioni degli insiemi, FSC, SSC, CSV delle predizioni dettagliate)
# vengono eseguite e salvate basandosi su una singola esecuzione riproducibile utilizzando BASE_SEED.
#
# Struttura:
#   0. Setup: Creazione Directory Risultati, Caricamento Librerie
#   1. Impostazioni Esperimento: Definizione di parametri come ALPHA, N_RUNS, seed.
#   2. Esecuzioni Multiple per la Distribuzione della Copertura Marginale:
#      2.1 Caricamento Completo del Dataset (una volta)
#      2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo
#      2.3 Avvio Ciclo N_RUNS
#          - Iterazione X: Divisione Dati
#          - Iterazione X: Addestramento Modello
#          - Iterazione X: Calibrazione
#          - Iterazione X: Predizione
#          - Iterazione X: Memorizzazione Copertura
#   3. Analisi e Salvataggio della Distribuzione della Copertura Marginale da N_RUNS:
#      3.1 Calcolo e Stampa delle Statistiche Riepilogative
#      3.2 Salvataggio dei Valori Grezzi di Copertura
#   4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED):
#      4.1 Setup per la Singola Esecuzione Dettagliata
#      4.2 Divisione Dati per la Singola Esecuzione
#      4.3 Addestramento Modello per la Singola Esecuzione
#      4.4 Calibrazione per la Singola Esecuzione
#      4.5 Predizione per la Singola Esecuzione
#      4.6 Valutazione e Salvataggio delle Metriche per la Singola Esecuzione (Copertura, Dimensioni Insiemi, FSC, SSC)

# Sourcing degli script R condivisi.
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")

# Caricamento Librerie
all_required_packages <- c("e1071", "dplyr", "ggplot2")
check_and_load_packages(all_required_packages)

# Definisci i nomi delle directory per salvare i risultati.
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section1_basic"
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

# --- 1. Impostazioni Esperimento ---

# Step 1: Definisci il seed base per la riproducibilità.
# Questo seed sarà usato per l'analisi dettagliata a singola esecuzione e come base per i seed delle esecuzioni multiple.
BASE_SEED <- 42
# Step 2: Definisci il livello di significatività (alpha) per la predizione conforme.
# Il target di copertura sarà $1 - \text{ALPHA_CONF}$.
ALPHA_CONF <- 0.1
# Step 3: Definisci la proporzione dei dati
PROP_TRAIN <- 0.7
PROP_CALIB <- 0.1
# Step 4: Definisci il numero di esecuzioni per generare la distribuzione della copertura marginale.
N_RUNS <- 100

# --- 2. Esecuzioni Multiple per la Distribuzione della Copertura Marginale ---

# Inizializza un vettore numerico per memorizzare le coperture empiriche di ogni esecuzione.
all_empirical_coverages_basic <- numeric(N_RUNS)

# --- 2.1 Caricamento Completo del Dataset (una volta) ---

# Carica il dataset Iris completo, preparato per la classificazione.
iris_data_full <- load_iris_for_classification()
n_total <- nrow(iris_data_full)

# --- 2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo ---

# Calcola il numero di campioni per i set di addestramento, calibrazione e test.
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop

# --- 2.3 Avvio Ciclo N_RUNS per la Copertura Marginale ---

for (run_iter in 1:N_RUNS) {
  # Step 1: Imposta un seed unico per ogni iterazione per garantire la riproducibilità di ogni divisione.
  current_run_seed <- BASE_SEED + run_iter
  set.seed(current_run_seed)
  
  # ------ Iterazione: Divisione Dati ------
  
  # Step 2.1: Rimescola gli indici del dataset per creare una nuova divisione per l'esecuzione corrente.
  shuffled_indices_for_run <- sample(n_total)
  current_data_for_run <- iris_data_full[shuffled_indices_for_run, ]
  
  # Step 2.2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
  train_df_iter <- current_data_for_run[1:n_train_loop, ]
  calib_df_iter <- current_data_for_run[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
  test_df_iter <- current_data_for_run[(n_train_loop + n_calib_loop + 1):n_total, ]
  
  # ------ Iterazione: Addestramento Modello ------
  
  # Step 2.3: Definisci la formula SVM (Species come variabile target, '.' per tutte le altre feature).
  svm_formula <- Species ~ .
  # Step 2.4: Addestra il modello SVM sui dati di addestramento dell'iterazione corrente.
  svm_model_iter <- train_svm_classification_model(svm_formula, train_df_iter)
  
  # ------ Iterazione: Calibrazione ------
  
  # Step 2.5: Predici le probabilità di classe per il set di calibrazione.
  calib_probs_iter <- predict_svm_probabilities(svm_model_iter, calib_df_iter)
  # Step 2.6: Estrai le vere etichette dal set di calibrazione.
  calib_true_labels_iter <- calib_df_iter$Species
  # Step 2.7: Calcola i punteggi di non-conformità per il set di calibrazione (metodo base).
  non_conf_scores_iter <- get_non_conformity_scores_basic(calib_probs_iter, calib_true_labels_iter)
  # Step 2.8: Calcola il valore $q_{hat}$ basato sui punteggi di non-conformità del set di calibrazione.
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iterazione: Predizione sul Set di Test ------
  
  # Step 2.9: Predici le probabilità di classe per il set di test.
  test_probs_iter <- predict_svm_probabilities(svm_model_iter, test_df_iter)
  # Step 2.10: Estrai le vere etichette dal set di test.
  test_true_labels_iter <- test_df_iter$Species
  # Step 2.11: Crea gli insiemi di predizione per il set di test utilizzando il $q_{hat}$ calcolato.
  prediction_sets_iter <- create_prediction_sets_basic(test_probs_iter, q_hat_iter)
  
  # ------ Iterazione: Memorizzazione Copertura Empirica ------
  
  # Step 2.12: Calcola la copertura empirica per l'esecuzione corrente.
  current_coverage <- calculate_empirical_coverage(prediction_sets_iter, test_true_labels_iter)
  # Step 2.13: Memorizza la copertura calcolata.
  all_empirical_coverages_basic[run_iter] <- current_coverage
}

# --- 3. Analisi e Salvataggio della Distribuzione della Copertura Marginale da N_RUNS ---

# --- 3.1 Calcolo e Stampa delle Statistiche Riepilogative ---

# Step 1: Calcola la media della copertura empirica su tutte le esecuzioni.
mean_empirical_coverage <- mean(all_empirical_coverages_basic, na.rm = TRUE)
# Step 2: Calcola la deviazione standard della copertura empirica.
sd_empirical_coverage <- sd(all_empirical_coverages_basic, na.rm = TRUE)

# --- 3.2 Salvataggio dei Valori Grezzi di Copertura ---

# Step 1: Crea un data frame con i risultati della copertura per ogni esecuzione.
coverage_distribution_df_basic <- data.frame(run_iteration = 1:N_RUNS, empirical_coverage = all_empirical_coverages_basic)
# Step 2: Definisci il percorso del file CSV.
coverage_dist_filename <- file.path(TABLES_DIR, "coverage_distribution_basic.csv")
# Step 3: Salva il data frame in un file CSV.
write.csv(coverage_distribution_df_basic, coverage_dist_filename, row.names = FALSE)

# --- 4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED per la riproducibilità) ---

# --- 4.1 Setup per Singola Esecuzione Dettagliata ---

# Imposta il seed base per garantire la riproducibilità di questa singola esecuzione.
set.seed(BASE_SEED)

# --- 4.2 Divisione Dati per Singola Esecuzione ---

# Step 1: Rimescola gli indici del dataset basandosi sul BASE_SEED.
single_run_shuffled_indices <- sample(n_total)
single_run_data <- iris_data_full[single_run_shuffled_indices, ]

# Step 2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
# Vengono utilizzate le stesse proporzioni definite in precedenza.
single_run_train_df <- single_run_data[1:n_train_loop, ]
single_run_calib_df <- single_run_data[(n_train_loop + 1):(n_train_loop + n_calib_loop), ]
single_run_test_df <- single_run_data[(n_train_loop + n_calib_loop + 1):n_total, ]

# --- 4.3 Addestramento Modello per Singola Esecuzione ---

# Addestra il modello SVM sul set di addestramento della singola esecuzione.
single_run_svm_model <- train_svm_classification_model(svm_formula, single_run_train_df)

# --- 4.4 Calibrazione per Singola Esecuzione ---

# Step 1: Predici le probabilità di classe per il set di calibrazione.
single_run_calib_probs <- predict_svm_probabilities(single_run_svm_model, single_run_calib_df)
# Step 2: Estrai le vere etichette dal set di calibrazione.
single_run_calib_true_labels <- single_run_calib_df$Species
# Step 3: Calcola i punteggi di non-conformità (metodo base).
single_run_non_conf_scores <- get_non_conformity_scores_basic(single_run_calib_probs, single_run_calib_true_labels)
# Step 4: Calcola $q_{hat}$ per la singola esecuzione.
single_run_q_hat_basic <- calculate_q_hat(single_run_non_conf_scores, ALPHA_CONF, n_calib = nrow(single_run_calib_df))

# --- 4.5 Predizione per Singola Esecuzione ---

# Step 1: Predici le probabilità di classe per il set di test.
single_run_test_probs <- predict_svm_probabilities(single_run_svm_model, single_run_test_df)
# Step 2: Estrai le vere etichette dal set di test.
single_run_test_true_labels <- single_run_test_df$Species
# Step 3: Crea gli insiemi di predizione per il set di test.
single_run_prediction_sets <- create_prediction_sets_basic(single_run_test_probs, single_run_q_hat_basic)

# --- 4.6 Valutazione e Salvataggio delle Metriche per Singola Esecuzione (Copertura, Dimensioni Insiemi, FSC, SSC) ---

# ---- 4.6.1 Dimensioni Insiemi a Singola Esecuzione ----

# Step 1: Ottieni le dimensioni degli insiemi di predizione.
single_run_set_sizes <- get_set_sizes(single_run_prediction_sets)
# Step 2: Calcola la dimensione media degli insiemi.
single_run_avg_set_size <- mean(single_run_set_sizes, na.rm = TRUE)
# Step 3: Crea un data frame riassuntivo delle dimensioni degli insiemi.
single_run_set_size_summary_df <- data.frame(
  Min = min(single_run_set_sizes, na.rm = TRUE), Q1 = quantile(single_run_set_sizes, 0.25, na.rm = TRUE, names = FALSE),
  Median = median(single_run_set_sizes, na.rm = TRUE), Mean = single_run_avg_set_size,
  Q3 = quantile(single_run_set_sizes, 0.75, na.rm = TRUE, names = FALSE), Max = max(single_run_set_sizes, na.rm = TRUE)
)
# Step 4: Salva il riassunto e i valori grezzi delle dimensioni degli insiemi in file CSV.
write.csv(single_run_set_size_summary_df, file.path(TABLES_DIR, "set_size_summary_basic_BASESEED_RUN.csv"), row.names = FALSE)
write.csv(data.frame(set_size = single_run_set_sizes), file.path(TABLES_DIR, "set_sizes_raw_basic_BASESEED_RUN.csv"), row.names = FALSE)

# ---- 4.6.2 FSC a Singola Esecuzione ----

# Step 1: Definisci il nome della feature per l'analisi FSC (Feature-conditional Coverage).
single_run_fsc_feature_name <- "Sepal.Length"
# Step 2: Calcola i risultati FSC.
single_run_fsc_results <- calculate_fsc(single_run_prediction_sets, single_run_test_true_labels,
                                        single_run_test_df[[single_run_fsc_feature_name]], feature_name = single_run_fsc_feature_name,
                                        num_bins_for_continuous = 4
)
# Step 3: Salva la tabella dei risultati FSC in un file CSV.
write.csv(single_run_fsc_results$coverage_by_group, file.path(TABLES_DIR, paste0("fsc_by_", single_run_fsc_feature_name, "_basic_BASESEED_RUN.csv")), row.names = FALSE)

# ---- 4.6.3 SSC a Singola Esecuzione ----

# Step 1: Calcola il numero di bin per l'analisi SSC (Set-size conditional Coverage).
single_run_ssc_bins <- max(1, min(length(unique(single_run_set_sizes)), length(levels(iris_data_full$Species))))
# Step 2: Calcola i risultati SSC.
single_run_ssc_results <- calculate_ssc(single_run_prediction_sets, single_run_test_true_labels, num_bins_for_size = single_run_ssc_bins)
# Step 3: Salva la tabella dei risultati SSC in un file CSV.
write.csv(single_run_ssc_results$coverage_by_group, file.path(TABLES_DIR, "ssc_basic_BASESEED_RUN.csv"), row.names = FALSE)
