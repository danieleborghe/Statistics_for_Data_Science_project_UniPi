# R/sec2_3_scalar_run_experiment.R
#
# Scopo:
# Questo script esegue l'esperimento di Conformalizing Scalar Uncertainty Estimates (Stime di Incertezza Scalare Conforme)
# (Sezione 2.3 dell'articolo). Implementa il metodo in cui l'incertezza è modellata predendo la magnitudo dei
# residui assoluti di un modello di predizione primario. Lo script segue la struttura standard del progetto:
# esecuzioni multiple per la distribuzione della copertura e una singola valutazione dettagliata.
#
# Struttura:
#   0. Setup: Creazione Directory Risultati, Caricamento Librerie
#   1. Impostazioni Esperimento: Definizione di parametri come ALPHA, N_RUNS, seed.
#   2. Esecuzioni Multiple per la Distribuzione della Copertura e della Larghezza:
#      2.1 Caricamento Completo del Dataset (una volta)
#      2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo
#      2.3 Avvio Ciclo N_RUNS
#          - Iterazione X: Divisione Dati
#          - Iterazione X: Addestramento Modello (Primario + Incertezza)
#          - Iterazione X: Calibrazione
#          - Iterazione X: Predizione
#          - Iterazione X: Memorizzazione Metriche (Copertura e Larghezza)
#   3. Analisi e Salvataggio della Distribuzione della Copertura e della Larghezza da N_RUNS:
#      3.1 Calcolo e Stampa delle Statistiche Riepilogative
#      3.2 Salvataggio dei Valori Grezzi della Distribuzione
#   4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED):
#      4.1 Setup per la Singola Esecuzione Dettagliata
#      4.2 Divisione Dati per la Singola Esecuzione
#      4.3 Addestramento Modello per la Singola Esecuzione
#      4.4 Calibrazione per la Singola Esecuzione
#      4.5 Predizione per la Singola Esecuzione
#      4.6 Salvataggio CSV Dettagliato degli Intervalli di Test
#      4.7 Valutazione e Salvataggio di Tutte le Metriche per la Singola Esecuzione:

# --- 0. Setup: Sourcing e Librerie ---
# Step 1: Carica le funzioni di utilità e predizione necessarie per l'esperimento.
source("R/conformal_predictors.R")
source("R/evaluation_utils.R")
source("R/experimentation_utils.R")


# --- 0. Setup: Caricamento Librerie ---
# Step 1: Definisci tutti i pacchetti R richiesti per questo script.
all_required_packages <- c("e1071", "dplyr", "ggplot2", "quantreg")
# Step 2: Controlla, installa (se necessario) e carica i pacchetti definiti.
check_and_load_packages(all_required_packages)

# --- 0. Setup: Creazione Directory Risultati ---
# Step 1: Definisci i nomi delle directory per salvare i risultati.
RESULTS_DIR <- "results"
METHOD_NAME_SUFFIX <- "section2_3_scalar_uncert"
TABLES_DIR <- file.path(RESULTS_DIR, "tables", METHOD_NAME_SUFFIX)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)


# --- 1. Impostazioni Esperimento ---

# Step 1: Definisci il seed base per la riproducibilità.
BASE_SEED <- 42
# Step 2: Definisci il livello di significatività (alpha) per la predizione conforme.
ALPHA_CONF <- 0.1
# Step 3: Definisci la proporzione dei dati per l'addestramento.
PROP_TRAIN <- 0.7
# Step 4: Definisci la proporzione dei dati per la calibrazione.
PROP_CALIB <- 0.1
# Step 5: Definisci il numero di esecuzioni per generare la distribuzione.
N_RUNS <- 100
# Step 6: Definisci la variabile target per la regressione.
TARGET_VARIABLE <- "Petal.Length"
# Step 7: Definisci la formula di regressione.
REG_FORMULA <- as.formula(paste(TARGET_VARIABLE, "~ ."))


# --- 2. Esecuzioni Multiple per la Distribuzione della Copertura e della Larghezza ---
# Step 1: Inizializza vettori numerici per memorizzare le coperture empiriche e le larghezze medie di ogni esecuzione.
all_empirical_coverages_su <- numeric(N_RUNS)
all_avg_widths_su <- numeric(N_RUNS)


# --- 2.1 Caricamento Completo del Dataset (una volta) ---
iris_data_full <- load_iris_for_regression()
n_total <- nrow(iris_data_full) # Ottieni il numero totale di campioni nel dataset.

# --- 2.2 Definizione Parametri di Divisione Dati Specifici per il Ciclo ---
n_train_loop <- floor(PROP_TRAIN * n_total)
n_calib_loop <- floor(PROP_CALIB * n_total)
n_test_loop <- n_total - n_train_loop - n_calib_loop

# --- 2.3 Avvio Ciclo N_RUNS ---
for (run_iter in 1:N_RUNS) {
  # Imposta un seed unico per ogni iterazione per garantire la riproducibilità.
  current_run_seed <- BASE_SEED + run_iter
  set.seed(current_run_seed)
  
  # ------ Iterazione X: Divisione Dati ------
  
  # Step 1: Rimescola gli indici del dataset per creare una nuova divisione per l'esecuzione corrente.
  shuffled_indices <- sample(n_total)
  # Step 2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
  train_df_iter <- iris_data_full[shuffled_indices[1:n_train_loop], ]
  calib_df_iter <- iris_data_full[shuffled_indices[(n_train_loop + 1):(n_train_loop + n_calib_loop)], ]
  test_df_iter <- iris_data_full[shuffled_indices[(n_train_loop + n_calib_loop + 1):n_total], ]
  
  # ------ Iterazione X: Addestramento Modello (Primario + Incertezza) ------
  
  # Addestra il modello di predizione primario e il modello di incertezza sui dati di addestramento.
  models_iter <- train_primary_and_uncertainty_models(REG_FORMULA, train_df_iter, TARGET_VARIABLE)
  
  # ------ Iterazione X: Calibrazione ------
  
  # Step 1: Calcola i punteggi di non-conformità scalari sui dati di calibrazione.
  non_conf_scores_iter <- get_non_conformity_scores_scalar(models_iter, calib_df_iter, TARGET_VARIABLE)
  # Step 2: Calcola il valore $q_{hat}$ basato sui punteggi di non-conformità.
  q_hat_iter <- calculate_q_hat(non_conf_scores_iter, ALPHA_CONF, n_calib = nrow(calib_df_iter))
  
  # ------ Iterazione X: Predizione ------
  
  # Crea gli intervalli di predizione per il set di test utilizzando i modelli e il $q_{hat}$.
  prediction_intervals_iter <- create_prediction_intervals_scalar(models_iter, test_df_iter, q_hat_iter)
  
  # ------ Iterazione X: Memorizzazione Metriche ------
  
  # Step 1: Estrai i veri valori della variabile target dal set di test.
  test_true_values_iter <- test_df_iter[[TARGET_VARIABLE]]
  # Step 2: Calcola la copertura empirica per l'iterazione corrente.
  all_empirical_coverages_su[run_iter] <- mean((test_true_values_iter >= prediction_intervals_iter$lower_bound) & (test_true_values_iter <= prediction_intervals_iter$upper_bound))
  # Step 3: Calcola la larghezza media degli intervalli per l'iterazione corrente.
  all_avg_widths_su[run_iter] <- mean(prediction_intervals_iter$upper_bound - prediction_intervals_iter$lower_bound, na.rm = TRUE)
}

# --- 3. Analisi e Salvataggio della Distribuzione della Copertura e della Larghezza da N_RUNS ---

# --- 3.1 Calcolo e Stampa delle Statistiche Riepilogative ---

# Step 1: Calcola la media e la deviazione standard della copertura empirica.
mean_coverage <- mean(all_empirical_coverages_su, na.rm = TRUE)
sd_coverage <- sd(all_empirical_coverages_su, na.rm = TRUE)
# Step 2: Calcola la larghezza media degli intervalli.
mean_width <- mean(all_avg_widths_su, na.rm = TRUE)


# --- 3.2 Salvataggio dei Valori Grezzi della Distribuzione ---

# Crea un data frame contenente i risultati di copertura e larghezza per ogni esecuzione.
write.csv(
  data.frame(run = 1:N_RUNS, coverage = all_empirical_coverages_su, avg_width = all_avg_widths_su),
  file.path(TABLES_DIR, "coverage_width_distribution.csv"), row.names = FALSE
)

# --- 4. Valutazione Dettagliata a Singola Esecuzione (utilizzando BASE_SEED) ---

# --- 4.1 Setup per Singola Esecuzione Dettagliata ---

# Imposta il seed base per garantire la riproducibilità di questa singola esecuzione.
set.seed(BASE_SEED)

# --- 4.2 Divisione Dati per Singola Esecuzione ---

# Step 1: Rimescola gli indici del dataset basandosi sul BASE_SEED.
shuffled_indices_single <- sample(n_total)
# Step 2: Dividi il dataset rimescolato in set di addestramento, calibrazione e test.
train_df <- iris_data_full[shuffled_indices_single[1:n_train_loop], ]
calib_df <- iris_data_full[shuffled_indices_single[(n_train_loop + 1):(n_train_loop + n_calib_loop)], ]
test_df <- iris_data_full[shuffled_indices_single[(n_train_loop + n_calib_loop + 1):n_total], ]

# --- 4.3 Addestramento Modello per Singola Esecuzione ---

# Addestra il modello primario e il modello di incertezza sui dati di addestramento della singola esecuzione.
models <- train_primary_and_uncertainty_models(REG_FORMULA, train_df, TARGET_VARIABLE)

# --- 4.4 Calibrazione per Singola Esecuzione ---

# Step 1: Calcola i punteggi di non-conformità scalari sui dati di calibrazione.
non_conf_scores <- get_non_conformity_scores_scalar(models, calib_df, TARGET_VARIABLE)
# Step 2: Calcola $q_{hat}$ per la singola esecuzione.
q_hat <- calculate_q_hat(non_conf_scores, ALPHA_CONF, n_calib = nrow(calib_df))

# --- 4.5 Analisi adattività ---

# Step 1: Calcola i residui assoluti sul set di calibrazione.
calib_preds <- predict(models$f_model, newdata = calib_df)
calib_residuals <- abs(calib_df[[TARGET_VARIABLE]] - calib_preds)

# Step 2: Crea un data frame che lega i punteggi ai residui.
adaptiveness_data <- data.frame(
  non_conformity_score = non_conf_scores,
  absolute_residual = calib_residuals
)

# Step 3: Salva il data frame in un file CSV.
adaptiveness_filename <- file.path(TABLES_DIR, "adaptiveness_data_BASESEED_RUN.csv")
write.csv(adaptiveness_data, adaptiveness_filename, row.names = FALSE)

# --- 4.6 Predizione per Singola Esecuzione ---

# Crea gli intervalli di predizione per il set di test della singola esecuzione.
prediction_intervals <- create_prediction_intervals_scalar(models, test_df, q_hat)

# --- 4.7 Salvataggio CSV Dettagliato degli Intervalli di Test ---

# Step 1: Crea un data frame con i risultati dettagliati degli intervalli.
intervals_df <- data.frame(
  SampleID = 1:nrow(test_df),
  TrueValue = test_df[[TARGET_VARIABLE]],
  LowerBound = prediction_intervals$lower_bound,
  UpperBound = prediction_intervals$upper_bound
)
# Step 2: Calcola la larghezza di ogni intervallo.
intervals_df$IntervalWidth <- intervals_df$UpperBound - intervals_df$LowerBound
# Step 3: Determina se il valore reale è coperto dall'intervallo.
intervals_df$Covered <- (intervals_df$TrueValue >= intervals_df$LowerBound) & (intervals_df$TrueValue <= intervals_df$UpperBound)
# Step 4: Salva il data frame dettagliato in un file CSV.
write.csv(intervals_df, file.path(TABLES_DIR, "detailed_test_intervals_BASESEED_RUN.csv"), row.names = FALSE)


# --- 4.8 Valutazione e Salvataggio di Tutte le Metriche per Singola Esecuzione ---

# ---- 4.8.1 Riepilogo Copertura e Larghezza Marginale a Singola Esecuzione ----

# Step 1: Calcola la copertura empirica media per la singola esecuzione.
single_run_coverage <- mean(intervals_df$Covered, na.rm = TRUE)
# Step 2: Calcola la larghezza media degli intervalli per la singola esecuzione.
single_run_avg_width <- mean(intervals_df$IntervalWidth, na.rm = TRUE)
# Step 3: Crea un data frame riassuntivo delle metriche chiave.
summary_df <- data.frame(
  method = METHOD_NAME_SUFFIX, alpha = ALPHA_CONF, target_coverage = 1 - ALPHA_CONF,
  empirical_coverage = single_run_coverage, avg_width = single_run_avg_width, q_hat = q_hat
)
# Step 4: Salva il riepilogo in un file CSV.
summary_filename <- file.path(TABLES_DIR, "summary_BASESEED_RUN.csv")
write.csv(summary_df, summary_filename, row.names = FALSE)

# ---- 4.8.2 Larghezze degli Intervalli a Singola Esecuzione (Riepilogo, Grezze) ----

# Step 1: Crea un data frame riassuntivo delle statistiche di larghezza.
width_summary_df <- data.frame(
  Min = min(intervals_df$IntervalWidth, na.rm = TRUE),
  Q1 = quantile(intervals_df$IntervalWidth, 0.25, na.rm = TRUE, names = FALSE),
  Median = median(intervals_df$IntervalWidth, na.rm = TRUE),
  Mean = single_run_avg_width,
  Q3 = quantile(intervals_df$IntervalWidth, 0.75, na.rm = TRUE, names = FALSE),
  Max = max(intervals_df$IntervalWidth, na.rm = TRUE)
)
# Step 2: Salva il riepilogo delle larghezze in un file CSV.
width_summary_filename <- file.path(TABLES_DIR, "width_summary_BASESEED_RUN.csv")
write.csv(width_summary_df, width_summary_filename, row.names = FALSE)
# Step 3: Salva i valori grezzi delle larghezze in un file CSV.
width_raw_filename <- file.path(TABLES_DIR, "widths_raw_BASESEED_RUN.csv")
write.csv(data.frame(IntervalWidth = intervals_df$IntervalWidth), width_raw_filename, row.names = FALSE)

# ---- 4.8.3 FSC a Singola Esecuzione (Feature-Stratified Coverage) ----

# Step 1: Definisci il nome della feature per l'analisi FSC.
fsc_feature_name <- "Sepal.Length"
# Step 2: Crea gruppi basati sulla feature per la stratificazione.
feature_groups <- cut(test_df[[fsc_feature_name]], breaks = 4, include.lowest = TRUE, ordered_result = TRUE)
# Step 3: Combina i gruppi delle feature con lo stato di copertura.
fsc_df <- data.frame(feature_group = feature_groups, covered = intervals_df$Covered)

# Step 4: Calcola la copertura per ciascun gruppo di feature.
fsc_results_df <- fsc_df %>%
  dplyr::group_by(feature_group, .drop = FALSE) %>%
  dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
  dplyr::filter(count > 0)

# Step 5: Salva la tabella dei risultati FSC in un file CSV.
fsc_table_filename <- file.path(TABLES_DIR, paste0("fsc_by_", fsc_feature_name, "_BASESEED_RUN.csv"))
write.csv(fsc_results_df, fsc_table_filename, row.names = FALSE)

# ---- 4.8.4 SSC a Singola Esecuzione (Set-Stratified Coverage) ----

# Step 1: Crea gruppi basati sulla larghezza dell'intervallo per la stratificazione.
width_groups <- cut(intervals_df$IntervalWidth,
                    breaks = quantile(intervals_df$IntervalWidth, probs = c(0, 0.33, 0.66, 1), na.rm = TRUE),
                    labels = c("Stretto", "Medio", "Largo"),
                    include.lowest = TRUE,
                    ordered_result = TRUE
)

# Step 2: Combina i gruppi di larghezza con lo stato di copertura.
ssc_df <- data.frame(size_group = width_groups, covered = intervals_df$Covered)

# Step 3: Calcola la copertura per ciascun gruppo di larghezza.
ssc_results_df <- ssc_df %>%
  dplyr::group_by(size_group, .drop = FALSE) %>%
  dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
  dplyr::filter(count > 0)

# Step 4: Salva la tabella dei risultati SSC in un file CSV.
ssc_table_filename <- file.path(TABLES_DIR, "ssc_BASESEED_RUN.csv")
write.csv(ssc_results_df, ssc_table_filename, row.names = FALSE)
