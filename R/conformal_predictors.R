# R/conformal_predictors.R
#
# Scopo:
# Questo script implementa la logica fondamentale per diverse procedure di Conformal Prediction (Predizione Conforme), come descritto nell'articolo.
# Include funzioni per il calcolo dei punteggi di non-conformità e per la creazione di insiemi/intervalli di predizione.
#
# Funzioni implementate:
#   - calculate_q_hat(...): Calcola il quantile $q_{hat}$ (condiviso da tutti i metodi).
#   - get_non_conformity_scores_basic(...): Calcola i punteggi di non-conformità di base per la classificazione.
#   - create_prediction_sets_basic(...): Crea gli insiemi di predizione di base per la classificazione.
#   - get_non_conformity_scores_adaptive(...): Calcola i punteggi di non-conformità adattivi per la classificazione.
#   - create_prediction_sets_adaptive(...): Crea gli insiemi di predizione adattivi per la classificazione.
#   - get_non_conformity_scores_bayes(...): Calcola i punteggi di non-conformità di Bayes per la classificazione.
#   - create_prediction_sets_bayes(...): Crea gli insiemi di predizione di Bayes per la classificazione.
#   - train_quantile_models(...): Addestra modelli di regressione quantile.
#   - get_non_conformity_scores_quantile(...): Calcola i punteggi di non-conformità per la regressione quantile.
#   - create_prediction_intervals_quantile(...): Crea gli intervalli di predizione per la regressione quantile.
#   - train_primary_and_uncertainty_models(...): Addestra i modelli primari e di incertezza per la regressione.
#   - get_non_conformity_scores_scalar(...): Calcola i punteggi di non-conformità scalari per la regressione.
#   - create_prediction_intervals_scalar(...): Crea gli intervalli di predizione scalari per la regressione.

# --- Funzione di Utilità (Condivisa) ---

calculate_q_hat <- function(non_conformity_scores, alpha, n_calib) {
  # Scopo: Calcola il valore $q_{hat}$, ovvero il quantile empirico $(1-\alpha)$ dei punteggi di non-conformità,
  #        aggiustato per la dimensione finita del campione.
  #
  # Parametri:
  #   - non_conformity_scores: Un vettore numerico di punteggi ottenuti dal set di calibrazione.
  #   - alpha: Il livello di significatività desiderato (es. 0.1 per una copertura del 90%).
  #   - n_calib: Il numero di campioni nel set di calibrazione.
  #
  # Ritorna: Il valore $q_{hat}$ calcolato.
  
  # Step 1: Calcola il livello di probabilità per la funzione quantile di R.
  # Questo è basato sulla formula dell'articolo per aggiustare il quantile.
  prob_level <- ceiling((n_calib + 1) * (1 - alpha)) / n_calib
  
  # Step 2: Assicura che il livello di probabilità sia nel range [0, 1].
  # Previene errori nel caso di valori estremi di alpha o n_calib.
  prob_level <- min(1, max(0, prob_level))
  
  # Step 3: Calcola $q_{hat}$ utilizzando il tipo 1 per l'inverso della ECDF (Funzione di Distribuzione Cumulativa Empirica).
  # `type = 1` garantisce un comportamento specifico del quantile che è appropriato per la Predizione Conforme.
  q_hat <- quantile(non_conformity_scores, probs = prob_level, type = 1, names = FALSE)
  
  # Step 4: Restituisce il valore $q_{hat}$.
  return(q_hat)
}


# --- Sezione 1: Predizione Conforme Base (Classificazione) ---

get_non_conformity_scores_basic <- function(predicted_probs_matrix, true_labels) {
  # Scopo: Calcola i punteggi di non-conformità di base per la classificazione.
  #        Il punteggio è $1 - P(Y=vero\_label|X)$.
  #
  # Parametri:
  #   - predicted_probs_matrix: Una matrice di probabilità predette (righe=campioni, colonne=classi).
  #   - true_labels: Un vettore delle vere etichette di classe per i campioni.
  #
  # Ritorna: Un vettore numerico di punteggi di non-conformità.
  
  # Step 1: Calcola i punteggi di non-conformità.
  # Per ogni campione, seleziona la probabilità predetta della sua vera classe e la sottrai da 1.
  # `cbind(1:nrow(predicted_probs_matrix), match(true_labels, colnames(predicted_probs_matrix)))` crea una matrice
  # di indici per selezionare la probabilità corretta per ogni riga.
  scores <- 1 - predicted_probs_matrix[cbind(1:nrow(predicted_probs_matrix), match(true_labels, colnames(predicted_probs_matrix)))]
  
  # Step 2: Restituisce il vettore dei punteggi.
  return(scores)
}

create_prediction_sets_basic <- function(predicted_probs_matrix_new_data, q_hat) {
  # Scopo: Crea insiemi di predizione di base basati sulla regola $P(Y=y|X) \ge 1 - q_{hat}$.
  #
  # Parametri:
  #   - predicted_probs_matrix_new_data: Matrice di probabilità per i nuovi dati.
  #   - q_hat: La soglia $q_{hat}$ calibrata.
  #
  # Ritorna: Una lista di vettori di caratteri, dove ogni vettore è un insieme di predizione.
  
  # Step 1: Calcola la soglia di probabilità.
  # Questa soglia determina quali classi saranno incluse nell'insieme di predizione.
  threshold <- 1 - q_hat
  
  # Step 2: Applica la soglia a ciascuna riga (campione) della matrice di probabilità.
  # Per ogni riga, seleziona i nomi delle colonne (classi) dove la probabilità predetta è maggiore o uguale alla soglia.
  prediction_sets <- apply(predicted_probs_matrix_new_data, 1, function(row) {
    names(row)[row >= threshold]
  })
  
  # Step 3: Restituisce la lista degli insiemi di predizione.
  return(prediction_sets)
}


# --- Sezione 2.1: Insiemi di Predizione Adattivi (Classificazione) ---

get_non_conformity_scores_adaptive <- function(predicted_probs_matrix, true_labels) {
  # Scopo: Calcola i punteggi di non-conformità adattivi.
  #        Il punteggio è la somma delle probabilità delle classi, ordinate per probabilità,
  #        fino a includere la vera classe.
  #
  # Parametri:
  #   - predicted_probs_matrix: Matrice di probabilità predette.
  #   - true_labels: Vettore delle vere etichette di classe.
  #
  # Ritorna: Un vettore numerico di punteggi di non-conformità adattivi.
  
  # Step 1: Inizializza un vettore numerico per memorizzare i punteggi.
  scores <- numeric(length(true_labels))
  
  # Step 2: Itera su ogni campione nel set di calibrazione.
  for (i in 1:length(true_labels)) {
    # Step 2.1: Estrai le probabilità predette per il campione corrente.
    probs_i <- predicted_probs_matrix[i, ]
    
    # Step 2.2: Ottieni il nome della vera classe per il campione corrente.
    true_class_name_i <- as.character(true_labels[i])
    
    # Step 2.3: Ordina gli indici delle probabilità in ordine decrescente.
    sorted_indices <- order(probs_i, decreasing = TRUE)
    
    # Step 2.4: Trova il rango (posizione nell'ordinamento) della vera classe.
    rank_true_class <- which(names(probs_i)[sorted_indices] == true_class_name_i)
    
    # Step 2.5: Calcola il punteggio come la somma cumulativa delle probabilità
    #           delle classi fino e inclusa la vera classe, in ordine decrescente.
    scores[i] <- sum(sort(probs_i, decreasing = TRUE)[1:rank_true_class])
  }
  
  # Step 3: Restituisce il vettore dei punteggi.
  return(scores)
}

create_prediction_sets_adaptive <- function(predicted_probs_matrix_new_data, q_hat) {
  # Scopo: Crea insiemi di predizione adattivi. Le classi vengono aggiunte all'insieme in ordine
  #        di probabilità decrescente fino a quando la loro somma cumulativa di probabilità
  #        non raggiunge o supera $q_{hat}$.
  #
  # Parametri:
  #   - predicted_probs_matrix_new_data: Matrice di probabilità per i nuovi dati.
  #   - q_hat: La soglia $q_{hat}$ calibrata dai punteggi adattivi.
  #
  # Ritorna: Una lista di vettori di caratteri, che rappresentano gli insiemi di predizione adattivi.
  
  # Step 1: Inizializza una lista per memorizzare gli insiemi di predizione.
  prediction_sets <- vector("list", nrow(predicted_probs_matrix_new_data))
  
  # Step 2: Itera su ogni campione nei nuovi dati.
  for (i in 1:nrow(predicted_probs_matrix_new_data)) {
    # Step 2.1: Estrai le probabilità predette per il campione corrente.
    probs_i <- predicted_probs_matrix_new_data[i, ]
    
    # Step 2.2: Ordina le probabilità in ordine decrescente.
    sorted_probs <- sort(probs_i, decreasing = TRUE)
    
    # Step 2.3: Calcola la somma cumulativa delle probabilità ordinate.
    cumulative_probs <- cumsum(sorted_probs)
    
    # Step 2.4: Trova il numero di classi da includere nell'insieme.
    # Questo è il punto in cui la somma cumulativa supera o eguaglia $q_{hat}$ per la prima volta.
    num_classes_to_include <- which.max(cumulative_probs >= q_hat)
    
    # Step 2.5: Aggiungi i nomi delle classi corrispondenti all'insieme di predizione.
    prediction_sets[[i]] <- names(sorted_probs)[1:num_classes_to_include]
  }
  
  # Step 3: Restituisce la lista degli insiemi di predizione.
  return(prediction_sets)
}


# --- Sezione 2.4: Conformalizing Bayes (Classificazione) ---

get_non_conformity_scores_bayes <- function(predicted_probs_matrix, true_labels) {
  # Scopo: Calcola i punteggi di non-conformità per Conformalizing Bayes.
  #        Il punteggio è il negativo della probabilità predetta dal modello per la vera classe.
  #
  # Parametri:
  #   - predicted_probs_matrix: Una matrice di probabilità predette.
  #   - true_labels: Un vettore delle vere etichette di classe.
  #
  # Ritorna: Un vettore numerico di punteggi di non-conformità.
  
  # Step 1: Calcola i punteggi di non-conformità.
  # Prende la probabilità della vera classe per ogni campione e la rende negativa.
  scores <- -predicted_probs_matrix[cbind(1:nrow(predicted_probs_matrix), match(true_labels, colnames(predicted_probs_matrix)))]
  
  # Step 2: Restituisce il vettore dei punteggi.
  return(scores)
}

create_prediction_sets_bayes <- function(predicted_probs_matrix_new_data, q_hat_bayes) {
  # Scopo: Crea insiemi di predizione per Conformalizing Bayes basati sulla regola $P(y|X) > -q_{hat}$.
  #
  # Parametri:
  #   - predicted_probs_matrix_new_data: Matrice di probabilità per i nuovi dati.
  #   - q_hat_bayes: La soglia $q_{hat}$ calibrata (tipicamente negativa).
  #
  # Ritorna: Una lista di vettori di caratteri, dove ogni vettore è un insieme di predizione.
  
  # Step 1: Calcola la soglia.
  # Poiché $q_{hat\_bayes}$ è negativo, `-q_hat_bayes` sarà positivo.
  threshold <- -q_hat_bayes
  
  # Step 2: Applica la soglia a ciascuna riga (campione) della matrice di probabilità.
  # Seleziona le classi dove la probabilità predetta è strettamente maggiore della soglia.
  prediction_sets <- apply(predicted_probs_matrix_new_data, 1, function(row) {
    names(row)[row > threshold]
  })
  
  # Step 3: Restituisce la lista degli insiemi di predizione.
  return(prediction_sets)
}


# --- Sezione 2.2: Conformalized Quantile Regression (Regressione) ---

train_quantile_models <- function(formula, training_data, alpha) {
  # Scopo: Addestra due modelli di regressione quantile per i limiti inferiore e superiore.
  #
  # Parametri:
  #   - formula: La formula di regressione (es. `Y ~ .`).
  #   - training_data: Il data frame per l'addestramento.
  #   - alpha: Il livello di significatività (es. 0.1).
  #
  # Ritorna: Una lista contenente i modelli addestrati `lower_model` e `upper_model`.
  
  # Step 1: Verifica la presenza del pacchetto 'quantreg'.
  # Se non è installato, interrompe l'esecuzione e informa l'utente.
  if (!requireNamespace("quantreg", quietly = TRUE)) stop("Il pacchetto 'quantreg' è richiesto per questa funzione.")
  
  # Step 2: Definisce i quantili (tau) per i modelli di regressione quantile.
  # tau_lower per il limite inferiore e tau_upper per il limite superiore.
  tau_lower <- alpha / 2
  tau_upper <- 1 - (alpha / 2)
  
  # Step 3: Addestra il modello per il limite inferiore.
  # Utilizza la funzione `rq` del pacchetto `quantreg`.
  model_lower <- quantreg::rq(formula, data = training_data, tau = tau_lower)
  
  # Step 4: Addestra il modello per il limite superiore.
  model_upper <- quantreg::rq(formula, data = training_data, tau = tau_upper)
  
  # Step 5: Restituisce una lista contenente entrambi i modelli addestrati.
  return(list(lower_model = model_lower, upper_model = model_upper))
}

get_non_conformity_scores_quantile <- function(models, calib_data, target_variable_name) {
  # Scopo: Calcola i punteggi di non-conformità per la regressione quantile.
  #        Il punteggio è $\max(lower\_bound(X) - Y, Y - upper\_bound(X))$.
  #
  # Parametri:
  #   - models: Una lista contenente i modelli addestrati `lower_model` e `upper_model`.
  #   - calib_data: Il data frame di calibrazione.
  #   - target_variable_name: Nome stringa della colonna della variabile target.
  #
  # Ritorna: Un vettore numerico di punteggi di non-conformità.
  
  # Step 1: Estrai i veri valori della variabile target dal set di calibrazione.
  y_true <- calib_data[[target_variable_name]]
  
  # Step 2: Effettua le predizioni dei limiti inferiore e superiore sui dati di calibrazione.
  q_hat_lower <- predict(models$lower_model, newdata = calib_data)
  q_hat_upper <- predict(models$upper_model, newdata = calib_data)
  
  # Step 3: Calcola i punteggi di non-conformità.
  # Il punteggio è la distanza massima tra il valore reale e i limiti predetti.
  scores <- pmax(q_hat_lower - y_true, y_true - q_hat_upper)
  
  # Step 4: Restituisce il vettore dei punteggi.
  return(scores)
}

create_prediction_intervals_quantile <- function(models, new_data, q_hat) {
  # Scopo: Crea intervalli di predizione conformi per i nuovi dati utilizzando la regressione quantile.
  #        L'intervallo è $[lower\_bound(X) - q_{hat}, upper\_bound(X) + q_{hat}]$.
  #
  # Parametri:
  #   - models: Una lista con i modelli addestrati `lower_model` e `upper_model`.
  #   - new_data: I dati per i quali creare gli intervalli.
  #   - q_hat: Il quantile $q_{hat}$ calibrato dal passo di calibrazione.
  #
  # Ritorna: Un data frame con le colonne `lower_bound` e `upper_bound`.
  
  # Step 1: Effettua le predizioni dei limiti inferiore e superiore sui nuovi dati.
  q_hat_lower <- predict(models$lower_model, newdata = new_data)
  q_hat_upper <- predict(models$upper_model, newdata = new_data)
  
  # Step 2: Calcola i limiti finali dell'intervallo di predizione.
  # Si aggiunge e si sottrae $q_{hat}$ ai limiti predetti.
  final_lower <- q_hat_lower - q_hat
  final_upper <- q_hat_upper + q_hat
  
  # Step 3: Restituisce un data frame con i limiti inferiore e superiore.
  return(data.frame(lower_bound = final_lower, upper_bound = final_upper))
}


# --- Sezione 2.3: Conformalizing Scalar Uncertainty (Regressione) ---

train_primary_and_uncertainty_models <- function(formula, training_data, target_variable_name) {
  # Scopo: Addestra il modello di predizione primario ($f_{hat}$) e il modello di incertezza ($u_{hat}$).
  #        Il modello di incertezza apprende a predire i residui assoluti del modello primario.
  #
  # Parametri:
  #   - formula: La formula di regressione per il modello primario.
  #   - training_data: Il data frame per l'addestramento.
  #   - target_variable_name: Nome stringa della variabile target.
  #
  # Ritorna: Una lista contenente i modelli addestrati `f_model` (primario) e `u_model` (incertezza).
  
  # Step 1: Verifica la presenza del pacchetto 'e1071'.
  # Se non è installato, interrompe l'esecuzione e informa l'utente.
  if (!requireNamespace("e1071", quietly = TRUE)) stop("Il pacchetto 'e1071' è richiesto per questa funzione.")
  
  # Step 2: Addestra il modello primario ($f_{hat}$).
  # Qui viene utilizzato un modello SVM per la regressione.
  f_model <- e1071::svm(formula, data = training_data)
  
  # Step 3: Calcola i residui assoluti sui dati di addestramento.
  # Questi residui saranno la variabile target per il modello di incertezza.
  train_preds <- predict(f_model, newdata = training_data)
  train_abs_residuals <- abs(training_data[[target_variable_name]] - train_preds)
  
  # Step 4: Prepara il data frame per l'addestramento del modello di incertezza.
  # Aggiunge i residui assoluti come nuova colonna.
  u_train_data <- training_data
  u_train_data$abs_residual <- train_abs_residuals
  
  # Step 5: Costruisci la formula per il modello di incertezza.
  # La variabile target è `abs_residual`, e si esclude la variabile originale target.
  u_formula_str <- paste("abs_residual ~ . -", target_variable_name)
  
  # Step 6: Addestra il modello di incertezza ($u_{hat}$).
  # Anche qui viene utilizzato un modello SVM.
  u_model <- e1071::svm(as.formula(u_formula_str), data = u_train_data)
  
  # Step 7: Restituisce una lista contenente il modello primario e il modello di incertezza.
  return(list(f_model = f_model, u_model = u_model))
}

get_non_conformity_scores_scalar <- function(models, calib_data, target_variable_name) {
  # Scopo: Calcola i punteggi di non-conformità per il metodo di incertezza scalare.
  #        Il punteggio è $|y - f_{hat}(x)| / u_{hat}(x)$.
  #
  # Parametri:
  #   - models: Una lista contenente i modelli addestrati `f_model` e `u_model`.
  #   - calib_data: Il data frame di calibrazione.
  #   - target_variable_name: Nome stringa della variabile target.
  #
  # Ritorna: Un vettore numerico di punteggi di non-conformità.
  
  # Step 1: Ottieni le predizioni del modello primario sui dati di calibrazione.
  calib_f_preds <- predict(models$f_model, newdata = calib_data)
  
  # Step 2: Ottieni le predizioni del modello di incertezza sui dati di calibrazione.
  calib_u_preds <- predict(models$u_model, newdata = calib_data)
  
  # Step 3: Calcola i residui assoluti tra i valori reali e le predizioni del modello primario.
  calib_abs_residuals <- abs(calib_data[[target_variable_name]] - calib_f_preds)
  
  # Step 4: Calcola i punteggi di non-conformità.
  # Il punteggio è il residuo assoluto diviso per la predizione di incertezza.
  scores <- calib_abs_residuals / calib_u_preds
  
  # Step 5: Restituisce il vettore dei punteggi.
  return(scores)
}

create_prediction_intervals_scalar <- function(models, new_data, q_hat) {
  # Scopo: Crea intervalli di predizione conformi utilizzando il metodo di incertezza scalare.
  #        L'intervallo è $[f_{hat}(x) - q_{hat} * u_{hat}(x), f_{hat}(x) + q_{hat} * u_{hat}(x)]$.
  #
  # Parametri:
  #   - models: Una lista contenente i modelli addestrati `f_model` e `u_model`.
  #   - new_data: I dati per i quali creare gli intervalli.
  #   - q_hat: Il quantile $q_{hat}$ calibrato dal passo di calibrazione.
  #
  # Ritorna: Un data frame con le colonne `lower_bound` e `upper_bound`.
  
  # Step 1: Ottieni le predizioni del modello primario sui nuovi dati.
  test_f_preds <- predict(models$f_model, newdata = new_data)
  
  # Step 2: Ottieni le predizioni del modello di incertezza sui nuovi dati.
  test_u_preds <- predict(models$u_model, newdata = new_data)
  
  # Step 3: Calcola il limite inferiore dell'intervallo di predizione.
  lower_bound <- test_f_preds - q_hat * test_u_preds
  
  # Step 4: Calcola il limite superiore dell'intervallo di predizione.
  upper_bound <- test_f_preds + q_hat * test_u_preds
  
  # Step 5: Restituisce un data frame con i limiti inferiore e superiore.
  return(data.frame(lower_bound = lower_bound, upper_bound = upper_bound))
}
