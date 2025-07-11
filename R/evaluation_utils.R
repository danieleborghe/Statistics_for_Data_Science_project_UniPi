# R/evaluation_utils.R
#
# Scopo:
# Questo script fornisce funzioni per la valutazione degli insiemi di predizione conforme,
# concentrandosi su metriche descritte nella Sezione 4 dell'articolo di riferimento, come
# la copertura empirica, l'analisi delle dimensioni degli insiemi, la Copertura Stratificata per Feature (FSC),
# e la Copertura Stratificata per Dimensione dell'Insieme (SSC).
#
# Funzioni:
#   - prepare_evaluation_dataframe(...): Prepara il data frame di base per la valutazione della copertura.
#   - discretize_variable_into_groups(...): Discretizza una variabile (numerica o categorica) in gruppi.
#   - calculate_coverage_and_count_by_group(...): Calcola copertura e conteggio per ciascun gruppo definito.
#   - calculate_empirical_coverage(...): Calcola la copertura marginale.
#   - get_set_sizes(...): Estrae le dimensioni degli insiemi di predizione.
#   - calculate_fsc(...): Calcola la Copertura Stratificata per Feature.
#   - calculate_ssc(...): Calcola la Copertura Stratificata per Dimensione dell'Insieme.

# --- Funzioni Helper per calculate_fsc e calculate_ssc per la classificazione---

prepare_evaluation_dataframe <- function(prediction_sets_list, true_labels_test) {
  # Scopo: Prepara il data frame di base per la valutazione della copertura,
  #        includendo le vere etichette e lo stato di copertura per ogni campione.
  #
  # Parametri:
  #   - prediction_sets_list: Una lista dove ogni elemento è un vettore di caratteri
  #                           che rappresenta un insieme di predizione per un singolo campione.
  #   - true_labels_test: Un vettore delle vere etichette di classe per i campioni di test.
  #
  # Ritorna: Un data frame con colonne 'true_label' e 'covered' (TRUE/FALSE).
  
  # Step 1: Inizializza il data frame con le etichette vere.
  df_eval <- data.frame(true_label = as.character(true_labels_test))
  
  # Step 2: Calcola lo stato 'coperto' per ogni campione.
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  # Step 3: Restituisce il data frame preparato.
  return(df_eval)
}

discretize_variable_into_groups <- function(variable_vector, num_bins_for_continuous, var_name = "variable") {
  # Scopo: Discretizza una variabile (numerica o categorica) in gruppi.
  #        Per le variabili numeriche continue, crea dei bin. Per altre, le converte in fattori.
  #
  # Parametri:
  #   - variable_vector: Un vettore di valori della variabile da discretizzare.
  #   - num_bins_for_continuous: Il numero di bin da usare se la variabile è numerica e continua.
  #   - var_name: Nome della variabile (per messaggi di debug o futuri miglioramenti).
  #
  # Ritorna: Un vettore fattore che rappresenta i gruppi discretizzati.
  
  # Step 1: Inizializza il vettore dei gruppi.
  group_vector <- NULL
  
  # Step 2: Controlla il tipo della variabile e discretizza di conseguenza.
  if (is.numeric(variable_vector)) {
    unique_numeric_vals <- unique(variable_vector[!is.na(variable_vector)])
    if (length(unique_numeric_vals) > num_bins_for_continuous && max(unique_numeric_vals, na.rm = TRUE) > min(unique_numeric_vals, na.rm = TRUE)) {
      # Discretizza la variabile numerica in intervalli (bin).
      group_vector <- cut(variable_vector, breaks = num_bins_for_continuous, include.lowest = TRUE, ordered_result = TRUE)
    } else {
      # Se pochi valori unici, tratta come fattore discreto.
      group_vector <- factor(variable_vector)
    }
  } else {
    # Se non numerica, forza a fattore.
    group_vector <- as.factor(variable_vector)
  }
  
  # Step 3: Restituisce il vettore dei gruppi.
  return(group_vector)
}

calculate_coverage_and_count_by_group <- function(df_eval_with_groups, group_col_name) {
  # Scopo: Calcola la copertura empirica e il conteggio dei campioni per ciascun gruppo definito.
  #
  # Parametri:
  #   - df_eval_with_groups: Un data frame che include le colonne 'covered' e la colonna del gruppo.
  #   - group_col_name: Il nome della colonna che contiene i gruppi (e.g., "feature_group", "size_group").
  #
  # Ritorna: Un data frame con colonne per il gruppo, la copertura e il conteggio.
  
  # Step 1: Assicurati che la colonna del gruppo sia un fattore per raggruppamento coerente.
  df_eval_with_groups[[group_col_name]] <- as.factor(df_eval_with_groups[[group_col_name]])
  
  # Step 2: Utilizza dplyr per il calcolo.
  coverage_by_group_df <- df_eval_with_groups %>%
    dplyr::group_by(!!rlang::sym(group_col_name), .drop = FALSE) %>%
    dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
    dplyr::filter(count > 0)
  
  # Step 3: Restituisce il data frame con copertura e conteggio per gruppo.
  return(coverage_by_group_df)
}

# --- Sezione 4.1: Controlli di Base ---
calculate_empirical_coverage <- function(prediction_sets_list, true_labels_test) {
  # Scopo: Calcola la copertura marginale empirica degli insiemi di predizione.
  #
  # Parametri:
  #   - prediction_sets_list: Una lista dove ogni elemento è un vettore di caratteri
  #                           che rappresenta un insieme di predizione per un singolo campione.
  #   - true_labels_test: Un vettore delle vere etichette di classe per i campioni di test.
  #
  # Ritorna: Un singolo valore numerico che rappresenta la copertura empirica (da 0 a 1).
  
  # Step 1: Inizializza un contatore per gli insiemi coperti.
  covered_count <- 0
  
  # Step 2: Itera su ogni campione nel set di test.
  for (i in 1:length(prediction_sets_list)) {
    # Step 2.1: Controlla se l'etichetta vera del campione è contenuta nel suo insieme di predizione.
    if (as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]) {
      # Step 2.2: Se coperto, incrementa il contatore.
      covered_count <- covered_count + 1
    }
  }
  # Step 3: Calcola la copertura empirica come la proporzione di campioni coperti.
  empirical_coverage <- covered_count / length(true_labels_test)
  
  # Step 4: Restituisce il valore della copertura empirica.
  return(empirical_coverage)
}

get_set_sizes <- function(prediction_sets_list) {
  # Scopo: Calcola la dimensione (numero di elementi) di ciascun insieme di predizione.
  #
  # Parametri:
  #   - prediction_sets_list: Una lista di insiemi di predizione.
  #
  # Ritorna: Un vettore numerico contenente la dimensione di ciascun insieme di predizione.
  
  # Step 1: Applica la funzione `length` a ciascun elemento della lista per ottenere le dimensioni.
  set_sizes <- sapply(prediction_sets_list, length)
  
  # Step 2: Restituisce il vettore delle dimensioni degli insiemi.
  return(set_sizes)
}

# --- Sezione 4.2: Valutazione dell'Adattività per la classificazione ---
calculate_fsc <- function(prediction_sets_list, true_labels_test, feature_vector,
                          feature_name = "Feature", num_bins_for_continuous = 5) {
  # Scopo: Calcola la Copertura Stratificata per Feature (FSC).
  #        Questa metrica valuta se la copertura è mantenuta in sottogruppi
  #        della popolazione definiti da una specifica feature.
  #
  # Parametri:
  #   - prediction_sets_list: Una lista di insiemi di predizione per i campioni di test.
  #   - true_labels_test: Un vettore delle vere etichette di classe per i campioni di test.
  #   - feature_vector: Un vettore dei valori della feature da stratificare.
  #   - feature_name: Nome della feature (per messaggi e titoli).
  #   - num_bins_for_continuous: Numero di bin da usare per features numeriche continue.
  #
  # Ritorna: Una lista contenente `min_coverage` (copertura minima tra i gruppi)
  #          e `coverage_by_group` (un data frame con la copertura per ogni gruppo).
  
  # Step 1: Prepara il data frame di valutazione di base.
  df_eval <- prepare_evaluation_dataframe(prediction_sets_list, true_labels_test)
  
  # Step 2: Discretizza la feature in gruppi e aggiungila al data frame.
  df_eval$feature_group <- discretize_variable_into_groups(feature_vector, num_bins_for_continuous, var_name = feature_name)
  
  # Step 3: Calcola la copertura e il conteggio per ciascun gruppo di feature.
  coverage_by_group_df <- calculate_coverage_and_count_by_group(df_eval, "feature_group")
  
  # Step 4: Gestisci il caso in cui non ci siano gruppi validi o tutte le coperture siano NA.
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    return(list(min_coverage = NA, coverage_by_group = data.frame(feature_group = factor(), coverage = numeric(), count = integer())))
  }
  
  # Step 5: Calcola la copertura minima FSC tra tutti i gruppi.
  min_fsc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  
  # Step 6: Restituisce la copertura minima e il data frame della copertura per gruppo.
  return(list(min_coverage = min_fsc, coverage_by_group = coverage_by_group_df))
}

calculate_ssc <- function(prediction_sets_list, true_labels_test, num_bins_for_size = 3) {
  # Scopo: Calcola la Copertura Stratificata per Dimensione dell'Insieme (SSC).
  #        Questa metrica valuta se la copertura è mantenuta in sottogruppi
  #        della popolazione definiti dalla dimensione degli insiemi di predizione.
  #
  # Parametri:
  #   - prediction_sets_list: Una lista di insiemi di predizione per i campioni di test.
  #   - true_labels_test: Un vettore delle vere etichette di classe per i campioni di test.
  #   - num_bins_for_size: Numero di bin da usare per discretizzare le dimensioni degli insiemi.
  #
  # Ritorna: Una lista contenente `min_coverage` (copertura minima tra i gruppi di dimensioni)
  #          e `coverage_by_group` (un data frame con la copertura per ogni gruppo di dimensioni).
  
  # Step 1: Prepara il data frame di valutazione di base.
  df_eval <- prepare_evaluation_dataframe(prediction_sets_list, true_labels_test)
  
  # Step 2: Ottieni le dimensioni di ciascun insieme di predizione e aggiungile al data frame.
  df_eval$set_size <- get_set_sizes(prediction_sets_list)
  
  # Step 3: Discretizza le dimensioni degli insiemi in gruppi e aggiungile al data frame.
  df_eval$size_group <- discretize_variable_into_groups(df_eval$set_size, num_bins_for_size, var_name = "set_size")
  
  # Step 4: Calcola la copertura e il conteggio per ciascun gruppo di dimensione.
  coverage_by_group_df <- calculate_coverage_and_count_by_group(df_eval, "size_group")
  
  # Step 5: Gestisci il caso in cui non ci siano gruppi validi o tutte le coperture siano NA.
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    return(list(min_coverage = NA, coverage_by_group = data.frame(size_group = factor(), coverage = numeric(), count = integer())))
  }
  
  # Step 6: Calcola la copertura minima SSC tra tutti i gruppi.
  min_ssc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  
  # Step 7: Restituisce la copertura minima e il data frame della copertura per gruppo.
  return(list(min_coverage = min_ssc, coverage_by_group = coverage_by_group_df))
}