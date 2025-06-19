# R/evaluation_utils.R
#
# Scopo:
# Questo script fornisce funzioni per la valutazione degli insiemi di predizione conforme,
# concentrandosi su metriche descritte nella Sezione 4 dell'articolo di riferimento, come
# la copertura empirica, l'analisi delle dimensioni degli insiemi, la Copertura Stratificata per Feature (FSC),
# e la Copertura Stratificata per Dimensione dell'Insieme (SSC). Include anche una funzione per tracciare
# la distribuzione delle coperture marginali da esecuzioni multiple.
#
# Funzioni:
#   - calculate_empirical_coverage(...): Calcola la copertura marginale.
#   - get_set_sizes(...): Estrae le dimensioni degli insiemi di predizione.
#   - plot_set_size_histogram(...): Traccia l'istogramma delle dimensioni degli insiemi.
#   - calculate_fsc(...): Calcola la Copertura Stratificata per Feature.
#   - calculate_ssc(...): Calcola la Copertura Stratificata per Dimensione dell'Insieme.
#   - plot_conditional_coverage(...): Traccia i risultati della copertura condizionale.
#   - plot_coverage_histogram(...): Traccia l'istogramma della distribuzione della copertura marginale.

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

plot_coverage_histogram <- function(all_empirical_coverages, alpha_conf, n_runs, method_name = "") {
  # Scopo: Genera e visualizza un istogramma della distribuzione dei valori di copertura
  #          marginale empirica ottenuti da esecuzioni multiple dell'esperimento.
  #
  # Parametri:
  #   - all_empirical_coverages: Un vettore numerico di valori di copertura empirica.
  #   - alpha_conf: Il livello di significatività utilizzato (es. 0.1).
  #   - n_runs: Il numero totale di esecuzioni (R) che hanno generato le coperture.
  #   - method_name: Una stringa che identifica il metodo conforme (per il titolo del grafico).
  #
  # Ritorna: Nessuno (traccia un istogramma).
  
  # Step 1: Stampa un messaggio informativo sull'inizio del tracciamento.
  # Si è ridotto l'uso di `cat` a solo messaggi essenziali come questo.
  cat(paste0("INFO: Tracciamento istogramma di ", n_runs, " valori di copertura marginale per '", method_name, "'...\n"))
  
  # Step 2: Calcola la copertura target desiderata e la copertura media osservata.
  target_coverage <- 1 - alpha_conf
  mean_observed_coverage <- mean(all_empirical_coverages, na.rm = TRUE)
  
  # Step 3: Crea il titolo per l'istogramma.
  hist_title <- paste("Distribuzione della Copertura Marginale\n(", method_name, ", N=", n_runs, " esecuzioni)", sep = "")
  
  # Step 4: Genera l'istogramma.
  hist(all_empirical_coverages,
       main = hist_title,
       xlab = "Copertura Empirica",
       ylab = "Frequenza",
       col = "skyblue",
       border = "black",
       las = 1, # Orienta le etichette dell'asse y orizzontalmente
       breaks = "Scott" # O specifica un numero di bin, es. 20
  )
  # Step 5: Aggiungi una linea verticale per la copertura target desiderata (rossa, tratteggiata).
  abline(v = target_coverage, col = "red", lwd = 2, lty = 2)
  # Step 6: Aggiungi una linea verticale per la copertura media osservata (verde scuro, punteggiata).
  abline(v = mean_observed_coverage, col = "darkgreen", lwd = 2, lty = 3)
  # Step 7: Aggiungi una leggenda per le linee aggiunte.
  legend("topright",
         legend = c(paste("Copertura Target (", round(target_coverage, 2), ")", sep = ""),
                    paste("Media Osservata (", round(mean_observed_coverage, 3), ")", sep = "")
         ),
         col = c("red", "darkgreen"),
         lty = c(2, 3), lwd = 2, cex = 0.8,
         box.lty = 0
  ) # Nessun bordo attorno alla leggenda per un aspetto più pulito
  
  # Step 8: Stampa un messaggio informativo sul completamento del tracciamento.
  cat(paste0("INFO: Istogramma per '", method_name, "' generato.\n"))
}

plot_set_size_histogram <- function(set_sizes, main_title = "Distribuzione Dimensioni Insiemi di Predizione") {
  # Scopo: Genera e visualizza un istogramma delle dimensioni degli insiemi di predizione.
  #
  # Parametri:
  #   - set_sizes: Un vettore numerico delle dimensioni degli insiemi di predizione.
  #   - main_title: Il titolo per il grafico dell'istogramma.
  #
  # Ritorna: Nessuno (traccia un istogramma).
  
  # Step 1: Stampa un messaggio informativo sull'inizio del tracciamento.
  cat(paste("INFO: Tracciamento istogramma dimensioni insieme: '", main_title, "'\n", sep = ""))
  
  # Step 2: Genera l'istogramma delle dimensioni degli insiemi.
  hist(set_sizes,
       main = main_title,
       xlab = "Dimensione Insieme",
       ylab = "Frequenza",
       breaks = seq(min(set_sizes, na.rm = TRUE) - 0.5, max(set_sizes, na.rm = TRUE) + 0.5, by = 1), # Assicura bin centrati sui numeri interi
       col = "lightblue",
       border = "black"
  )
  # Step 3: Stampa la dimensione media osservata degli insiemi.
  cat(paste("INFO: Dimensione media insieme osservata: ", round(mean(set_sizes, na.rm = TRUE), 3), "\n", sep = ""))
}


# --- Sezione 4.2: Valutazione dell'Adattività ---
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
  
  # Step 1: Stampa un messaggio informativo sull'inizio del calcolo.
  cat(paste("INFO: Calcolo della Copertura Stratificata per Feature (FSC) per la feature '", feature_name, "'...\n", sep = ""))
  
  # Step 2: Crea un data frame temporaneo per la valutazione.
  # Include l'etichetta vera, il valore della feature e uno stato 'coperto'.
  df_eval <- data.frame(true_label = as.character(true_labels_test), feature_val = feature_vector)
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  # Step 3: Raggruppa i dati in base alla feature.
  # Se la feature è numerica e ha molti valori unici, viene discretizzata in bin.
  # Altrimenti, viene trattata come fattore.
  if (is.numeric(df_eval$feature_val)) {
    unique_numeric_vals <- unique(df_eval$feature_val[!is.na(df_eval$feature_val)])
    if (length(unique_numeric_vals) > num_bins_for_continuous) {
      # Discretizza la feature numerica in `num_bins_for_continuous` intervalli.
      df_eval$feature_group <- cut(df_eval$feature_val, breaks = num_bins_for_continuous, include.lowest = TRUE, ordered_result = TRUE)
    } else {
      # Se pochi valori unici, tratta come fattore discreto.
      df_eval$feature_group <- factor(df_eval$feature_val)
    }
  } else {
    # Se non numerica, forza a fattore.
    df_eval$feature_group <- as.factor(df_eval$feature_val)
  }
  
  # Step 4: Calcola la copertura e il conteggio per ciascun gruppo di feature.
  # Si preferisce `dplyr` per la sua sintassi leggibile, con un fallback a R base.
  coverage_by_group_df <- NULL
  if (requireNamespace("dplyr", quietly = TRUE)) {
    # Carica dplyr per assicurarsi che le funzioni siano disponibili.
    library(dplyr)
    coverage_by_group_df <- tryCatch(
      {
        # SuppressMessages per evitare output verboso da dplyr.
        suppressMessages(
          df_eval %>%
            dplyr::group_by(feature_group, .drop = FALSE) %>% # .drop=FALSE mantiene i gruppi anche se vuoti (utile per completezza)
            dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
            dplyr::filter(count > 0) # Rimuovi i gruppi senza campioni
        )
      },
      error = function(e) {
        warning("WARN: Errore durante l'operazione dplyr per FSC nonostante il pacchetto sia stato caricato. Ripiego. Errore: ", e$message)
        NULL # Segnala di usare il fallback di R base
      }
    )
  } else {
    warning("WARN: Il pacchetto dplyr non è installato. Il calcolo FSC utilizzerà aggregate di R base.")
  }
  
  # Fallback a R base se dplyr non disponibile o ha fallito.
  if (is.null(coverage_by_group_df)) {
    # Calcola la media (copertura) per gruppo.
    agg_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    # Calcola il conteggio dei campioni per gruppo.
    count_df <- aggregate(covered ~ feature_group, data = df_eval, FUN = length, drop = FALSE)
    # Rinomina le colonne e unisce i data frame.
    colnames(agg_df) <- c("feature_group", "coverage")
    colnames(count_df) <- c("feature_group", "count")
    coverage_by_group_df <- merge(agg_df, count_df, by = "feature_group", all.x = TRUE) # Assicura che tutti i gruppi siano mantenuti inizialmente
    # Filtra i gruppi senza campioni validi.
    coverage_by_group_df <- coverage_by_group_df[coverage_by_group_df$count > 0 & !is.na(coverage_by_group_df$feature_group), ]
  }
  
  # Step 5: Gestisci il caso in cui non ci siano gruppi validi o tutte le coperture siano NA.
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: Nessun gruppo valido trovato per FSC o tutte le coperture sono NA.\n")
    # Restituisce un data frame vuoto con le colonne corrette per evitare errori successivi.
    return(list(min_coverage = NA, coverage_by_group = data.frame(feature_group = factor(), coverage = numeric(), count = integer())))
  }
  
  # Step 6: Calcola la copertura minima FSC tra tutti i gruppi.
  min_fsc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  cat(paste("INFO: FSC minimo calcolato: ", round(min_fsc, 3), "\n", sep = ""))
  
  # Step 7: Restituisce la copertura minima e il data frame della copertura per gruppo.
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
  
  # Step 1: Stampa un messaggio informativo sull'inizio del calcolo.
  cat("INFO: Calcolo della Copertura Stratificata per Dimensione dell'Insieme (SSC)...\n")
  
  # Step 2: Ottieni le dimensioni di ciascun insieme di predizione.
  set_sizes <- get_set_sizes(prediction_sets_list)
  # Step 3: Crea un data frame temporaneo per la valutazione.
  df_eval <- data.frame(true_label = as.character(true_labels_test), set_size = set_sizes)
  # Step 4: Determina se ogni campione è coperto dal suo insieme di predizione.
  df_eval$covered <- sapply(1:length(prediction_sets_list), function(i) {
    as.character(true_labels_test[i]) %in% prediction_sets_list[[i]]
  })
  
  # Step 5: Raggruppa i dati in base alla dimensione dell'insieme.
  # Se ci sono abbastanza dimensioni uniche, le discretizza in bin.
  # Altrimenti, le tratta come fattori discreti.
  unique_sizes <- unique(df_eval$set_size[!is.na(df_eval$set_size)])
  if (length(unique_sizes) > num_bins_for_size && max(unique_sizes, na.rm = TRUE) > min(unique_sizes, na.rm = TRUE)) {
    # Discretizza le dimensioni degli insiemi in `num_bins_for_size` intervalli.
    df_eval$size_group <- cut(df_eval$set_size, breaks = num_bins_for_size, include.lowest = TRUE, ordered_result = TRUE)
  } else {
    # Se poche dimensioni uniche, tratta come fattore discreto.
    df_eval$size_group <- factor(df_eval$set_size)
  }
  
  # Step 6: Calcola la copertura e il conteggio per ciascun gruppo di dimensione.
  # Si preferisce `dplyr` per la sua sintassi leggibile, con un fallback a R base.
  coverage_by_group_df <- NULL
  if (requireNamespace("dplyr", quietly = TRUE)) {
    # Carica dplyr per assicurarsi che le funzioni siano disponibili.
    library(dplyr)
    coverage_by_group_df <- tryCatch(
      {
        # SuppressMessages per evitare output verboso da dplyr.
        suppressMessages(
          df_eval %>%
            dplyr::group_by(size_group, .drop = FALSE) %>%
            dplyr::summarise(coverage = mean(covered, na.rm = TRUE), count = n(), .groups = 'drop') %>%
            dplyr::filter(count > 0) # Rimuovi i gruppi senza campioni
        )
      },
      error = function(e) {
        warning("WARN: Errore durante l'operazione dplyr per SSC nonostante il pacchetto sia stato caricato. Ripiego. Errore: ", e$message)
        NULL
      }
    )
  } else {
    warning("WARN: Il pacchetto dplyr non è installato. Il calcolo SSC utilizzerà aggregate di R base.")
  }
  
  # Fallback a R base se dplyr non disponibile o ha fallito.
  if (is.null(coverage_by_group_df)) {
    # Calcola la media (copertura) per gruppo.
    agg_df <- aggregate(covered ~ size_group, data = df_eval, FUN = function(x) mean(x, na.rm = TRUE), drop = FALSE)
    # Calcola il conteggio dei campioni per gruppo.
    count_df <- aggregate(covered ~ size_group, data = df_eval, FUN = length, drop = FALSE)
    # Rinomina le colonne e unisce i data frame.
    colnames(agg_df) <- c("size_group", "coverage")
    colnames(count_df) <- c("size_group", "count")
    coverage_by_group_df <- merge(agg_df, count_df, by = "size_group", all.x = TRUE)
    # Filtra i gruppi senza campioni validi.
    coverage_by_group_df <- coverage_by_group_df[coverage_by_group_df$count > 0 & !is.na(coverage_by_group_df$size_group), ]
  }
  
  # Step 7: Gestisci il caso in cui non ci siano gruppi validi o tutte le coperture siano NA.
  if (nrow(coverage_by_group_df) == 0 || all(is.na(coverage_by_group_df$coverage))) {
    cat("WARN: Nessun gruppo valido trovato per SSC o tutte le coperture sono NA.\n")
    # Restituisce un data frame vuoto con le colonne corrette per evitare errori successivi.
    return(list(min_coverage = NA, coverage_by_group = data.frame(size_group = factor(), coverage = numeric(), count = integer())))
  }
  
  # Step 8: Calcola la copertura minima SSC tra tutti i gruppi.
  min_ssc <- min(coverage_by_group_df$coverage, na.rm = TRUE)
  cat(paste("INFO: SSC minimo calcolato: ", round(min_ssc, 3), "\n", sep = ""))
  
  # Step 9: Restituisce la copertura minima e il data frame della copertura per gruppo.
  return(list(min_coverage = min_ssc, coverage_by_group = coverage_by_group_df))
}

plot_conditional_coverage <- function(coverage_by_group_df, group_col_name, coverage_col_name,
                                      desired_coverage, main_title) {
  # Scopo: Genera un grafico a barre per i risultati della copertura condizionale (FSC o SSC).
  #
  # Parametri:
  #   - coverage_by_group_df: Un data frame contenente la colonna del gruppo, la colonna di copertura
  #                           e opzionalmente la colonna del conteggio dei campioni.
  #   - group_col_name: Il nome della colonna che contiene i nomi dei gruppi (es. "feature_group", "size_group").
  #   - coverage_col_name: Il nome della colonna che contiene i valori di copertura (es. "coverage").
  #   - desired_coverage: Il livello di copertura target desiderato (per la linea di riferimento).
  #   - main_title: Il titolo principale del grafico.
  #
  # Ritorna: Nessuno (traccia un grafico ggplot).
  
  # Step 1: Stampa un messaggio informativo sull'inizio del tracciamento.
  cat(paste("INFO: Tracciamento copertura condizionale: '", main_title, "'\n", sep = ""))
  
  # Step 2: Effettua controlli di validità sui dati del data frame.
  if (nrow(coverage_by_group_df) == 0 ||
      !coverage_col_name %in% names(coverage_by_group_df) ||
      !group_col_name %in% names(coverage_by_group_df) ||
      all(is.na(coverage_by_group_df[[coverage_col_name]]))) {
    cat("WARN: Impossibile tracciare la copertura condizionale a causa di dati vuoti, malformati o tutti NA.\n")
    return(invisible(NULL)) # Restituisce invisibilmente NULL se i dati non sono validi per il tracciamento.
  }
  
  # Step 3: Verifica la presenza del pacchetto ggplot2 e lo carica.
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    warning("WARN: Il pacchetto ggplot2 non è installato. Impossibile generare il grafico di copertura condizionale. Stampa tabella invece.")
    print(coverage_by_group_df)
    return(invisible(NULL))
  }
  library(ggplot2)
  
  # Step 4: Assicura che la colonna del gruppo sia un fattore per un corretto raggruppamento nel grafico.
  coverage_by_group_df[[group_col_name]] <- as.factor(coverage_by_group_df[[group_col_name]])
  
  # Step 5: Crea il grafico a barre utilizzando ggplot2.
  p <- ggplot(coverage_by_group_df, aes_string(x = group_col_name, y = coverage_col_name, fill = group_col_name)) +
    geom_bar(stat = "identity", color = "black", na.rm = TRUE) + # `stat="identity"` usa i valori y così come sono
    geom_hline(yintercept = desired_coverage, linetype = "dashed", color = "red", linewidth = 1) + # Linea di riferimento per la copertura desiderata
    labs(title = main_title, x = "Gruppo", y = "Copertura Empirica") + # Etichette e titolo
    theme_minimal() + # Tema minimalista per un aspetto pulito
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") + # Rotazione etichette asse x e nascondi leggenda
    ylim(0, 1) # Imposta i limiti dell'asse y da 0 a 1 per la copertura
  
  # Step 6: Stampa il grafico.
  print(p)
}