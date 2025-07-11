# R/experimentation_utils.R
#
# Scopo:
# Questo script contiene funzioni di utilità che possono essere riutilizzate
# in diverse parti del progetto R_Conformal_Iris_SVM.
#
# Funzioni:
#   - check_and_load_packages(...): Controlla, installa e carica i pacchetti richiesti.
#   - load_iris_for_classification(...): Carica e prepara il dataset Iris per la classificazione.
#   - load_iris_for_regression(...): Carica e prepara il dataset Iris per la regressione.
#   - train_svm_classification_model(...): Addestra un modello SVM per la classificazione.
#   - predict_svm_probabilities(...): Predice le probabilità di classe usando un modello SVM.
#   - train_svm_regression_model(...): Addestra un modello SVM per la regressione.
#   - predict_svm_regression_values(...): Predice valori numerici usando un modello SVM di regressione.
#   - train_quantile_models(...): Addestra due modelli di regressione quantile per i limiti inferiore e superiore.
#   - train_primary_and_uncertainty_models(...): Addestra un modello primario e un modello di incertezza (per i residui assoluti).
#   - train_mean_and_stddev_models(...): Addestra un modello per la media e un modello per la varianza dei residui.

check_and_load_packages <- function(required_packages) {
  # Scopo: Controlla se una lista di pacchetti è installata, installa quelli mancanti
  #        e poi li carica nella sessione corrente.
  #
  # Parametri:
  #   - required_packages: Un vettore di caratteri con i nomi dei pacchetti.
  #
  # Ritorna: Nessuno. Stampa messaggi sulla console e carica i pacchetti.
  
  # Step 1: Itera su ciascun pacchetto richiesto.
  for (pkg in required_packages) {
    # Step 1.1: Controlla se il pacchetto è già installato.
    if (!requireNamespace(pkg, quietly = TRUE)) {
      # Step 2.2: Se il pacchetto non è trovato, tenta di installarlo.
      install.packages(pkg, dependencies = TRUE)
    }
    
    # Step 1.2: Carica il pacchetto
    library(pkg, character.only = TRUE)
  }
}

load_iris_for_classification <- function() {
  # Scopo: Carica il dataset Iris incorporato in R, si assicura che 'Species' sia un fattore,
  #        e rimescola il dataset.
  #
  # Parametri: Nessuno
  #
  # Ritorna: Un data frame contenente il dataset Iris rimescolato, pronto per la classificazione.
  
  # Step 1: Carica il dataset Iris.
  data(iris) # Carica il dataset Iris dai dataset predefiniti di R
  
  # Step 2: Assicura che la colonna 'Species' sia trattata come un fattore (cruciale per i modelli di classificazione)
  iris$Species <- as.factor(iris$Species)
  
  # Step 3: Rimescola il dataset per garantire un campionamento casuale nelle fasi successive.
  # Imposta un seed per la riproducibilità del rimescolamento.
  set.seed(123)
  iris_shuffled <- iris[sample(nrow(iris)), ]
  
  # Step 4: Restituisce il dataset Iris rimescolato e preparato.
  return(iris_shuffled)
}

load_iris_for_regression <- function() {
  # Scopo: Carica il dataset Iris e lo prepara per un task di regressione
  #        rimuovendo la colonna 'Species' e rimescolando le righe.
  #
  # Parametri: Nessuno.
  #
  # Ritorna: Un data frame rimescolato del dataset Iris adatto per la regressione.
  
  # Step 1: Carica il dataset Iris.
  data(iris)
  
  # Step 2: Rimuove la colonna 'Species', che non è necessaria per la regressione.
  iris_regression <- iris[, -which(names(iris) == "Species")]
  
  # Step 3: Rimescola il dataset per garantire casualità.
  set.seed(123)
  shuffled_iris <- iris_regression[sample(nrow(iris_regression)), ]
  
  # Step 4: Restituisce il dataset Iris rimescolato e preparato per la regressione.
  return(shuffled_iris)
}

train_svm_classification_model <- function(formula, training_data) {
  # Scopo: Addestra un modello SVM per la classificazione utilizzando la formula specificata
  #        e i dati di addestramento, con l'abilitazione della stima della probabilità.
  #
  # Parametri:
  #   - formula: Una formula R che specifica il modello
  #   - training_data: Un data frame contenente i dati di addestramento.
  #
  # Ritorna: Un oggetto modello SVM addestrato dal pacchetto 'e1071'.
  
  # Step 1: Estrai il nome della variabile target dalla formula.
  target_var_name <- all.vars(formula)[1]
  
  # Step 2: Addestra il modello SVM.
  # `kernel = "radial"`
  # `probability = TRUE` ci fornisce le probabilità
  # `scale = TRUE` scala i dati prima dell'addestramento.
  model <- e1071::svm(formula, data = training_data, kernel = "radial", probability = TRUE, scale = TRUE)
  
  # Step 3: Conclude l'addestramento del modello SVM.
  return(model)
}

predict_svm_probabilities <- function(svm_model, newdata) {
  # Scopo: Predice le probabilità di classe per nuovi dati utilizzando un modello SVM di classificazione addestrato.
  #
  # Parametri:
  #   - svm_model: Un oggetto modello SVM addestrato dalla funzione `train_svm_classification_model()`.
  #   - newdata: Un data frame per il quale predire le probabilità.
  #
  # Ritorna: Una matrice di probabilità di classe, dove le righe sono i campioni e le colonne sono le classi.
  
  # Step 1: Effettua le predizioni utilizzando il modello SVM.
  predictions_object <- predict(svm_model, newdata, probability = TRUE)
  
  # Step 2: Estrai l'attributo delle probabilità dall'oggetto di predizione.
  probabilities <- attr(predictions_object, "probabilities")
  
  # Step 3: Restituisce la matrice delle probabilità.
  return(probabilities)
}

train_svm_regression_model <- function(formula, training_data) {
  # Scopo: Addestra un modello SVM per la regressione utilizzando la formula specificata
  #        e i dati di addestramento.
  #
  # Parametri:
  #   - formula: Una formula R che specifica il modello
  #   - training_data: Un data frame contenente i dati di addestramento.
  #
  # Ritorna: Un oggetto modello SVM addestrato dal pacchetto 'e1071' per la regressione.
  
  # Step 1: Estrai il nome della variabile target dalla formula.
  target_var_name <- all.vars(formula)[1]
  
  # Step 2: Addestra il modello SVM per la regressione.
  # Utilizza il kernel radiale e scala i dati.
  model <- e1071::svm(formula, data = training_data, type = "eps-regression", kernel = "radial", scale = TRUE)
  
  # Step 4: Restituisce il modello addestrato.
  return(model)
}

predict_svm_regression_values <- function(svm_regression_model, newdata) {
  # Scopo: Predice i valori numerici per nuovi dati utilizzando un modello SVM di regressione addestrato.
  #
  # Parametri:
  #   - svm_regression_model: Un oggetto modello SVM addestrato per la regressione.
  #   - newdata: Un data frame per il quale predire i valori.
  #
  # Ritorna: Un vettore numerico di valori predetti.
  
  # Step 1: Effettua le predizioni utilizzando il modello SVM di regressione.
  predictions <- predict(svm_regression_model, newdata = newdata)
  
  # Step 2: Restituisce il vettore delle predizioni.
  return(predictions)
}

train_quantile_models <- function(formula, training_data, alpha) {
  # Scopo: Addestra due modelli di regressione quantile per i limiti inferiore e superiore.
  #
  # Parametri:
  #   - formula: La formula di regressione
  #   - training_data: Il data frame per l'addestramento.
  #   - alpha: Il livello di significatività
  #
  # Ritorna: Una lista contenente i modelli addestrati `lower_model` e `upper_model`.
  
  # Step 1: Definisce i quantili (tau) per i modelli di regressione quantile.
  # tau_lower per il limite inferiore e tau_upper per il limite superiore.
  tau_lower <- alpha / 2
  tau_upper <- 1 - (alpha / 2)
  
  # Step 2: Addestra il modello per il limite inferiore.
  # Utilizza la funzione `rq` del pacchetto `quantreg`.
  model_lower <- quantreg::rq(formula, data = training_data, tau = tau_lower)
  
  # Step 3: Addestra il modello per il limite superiore.
  model_upper <- quantreg::rq(formula, data = training_data, tau = tau_upper)
  
  # Step 5: Restituisce una lista contenente entrambi i modelli addestrati.
  return(list(lower_model = model_lower, upper_model = model_upper))
}

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
  
  # Step 1: Addestra il modello primario ($f_{hat}$) usando la nuova funzione dedicata.
  f_model <- train_svm_regression_model(formula, training_data)
  
  # Step 2: Calcola le predizioni del modello primario sui dati di addestramento
  #         e i residui assoluti.
  train_preds <- predict_svm_regression_values(f_model, newdata = training_data)
  train_abs_residuals <- abs(training_data[[target_variable_name]] - train_preds)
  
  # Step 3: Prepara il data frame per l'addestramento del modello di incertezza.
  u_train_data <- training_data
  u_train_data$abs_residual <- train_abs_residuals
  
  # Step 4: Costruisci la formula per il modello di incertezza.
  u_formula_str <- paste("abs_residual ~ . -", target_variable_name)
  
  # Step 5: Addestra il modello di incertezza ($u_{hat}$) usando la nuova funzione dedicata.
  u_model <- train_svm_regression_model(as.formula(u_formula_str), training_data = u_train_data)
  
  # Step 6: Restituisce una lista contenente il modello primario e il modello di incertezza.
  return(list(f_model = f_model, u_model = u_model))
}

train_mean_and_stddev_models <- function(formula, training_data, target_variable_name) {
  # Scopo: Addestra il modello primario per la media e un modello per la varianza dell'errore
  #        utilizzando SVM per la regressione.
  #
  # Parametri:
  #   - formula: La formula di regressione per il modello primario (media).
  #   - training_data: Il data frame per l'addestramento.
  #   - target_variable_name: Nome stringa della variabile target originale.
  #
  # Ritorna: Una lista contenente i modelli addestrati `mean_model` (per la media)
  #          e `variance_model` (per la varianza dei residui).
  
  # Step 1: Addestra il modello primario per la media (`mean_model`).
  # Questo modello apprende a predire i valori medi della variabile target.
  mean_model <- train_svm_regression_model(formula, training_data)
  
  # Step 2: Calcola le predizioni del modello per la media sui dati di addestramento.
  train_preds <- predict_svm_regression_values(mean_model, newdata = training_data)
  
  # Step 3: Calcola i residui al quadrato.
  # Questi saranno la variabile target per il modello di varianza,
  train_sq_residuals <- (training_data[[target_variable_name]] - train_preds)^2
  
  # Step 4: Prepara il data frame per l'addestramento del modello di varianza.
  # Aggiunge i residui al quadrato come nuova colonna.
  variance_train_data <- training_data
  variance_train_data$sq_residual <- train_sq_residuals
  
  # Step 5: Costruisci la formula per il modello di varianza.
  # La variabile target è `sq_residual`, e si esclude la variabile originale target.
  variance_formula_str <- paste("sq_residual ~ . -", target_variable_name)
  
  # Step 6: Addestra il modello per la varianza dell'errore (`variance_model`).
  # Questo modello apprende a predire la varianza basandosi sui regressori.
  variance_model <- train_svm_regression_model(as.formula(variance_formula_str), training_data = variance_train_data)
  
  # Step 7: Restituisce una lista contenente il modello per la media e quello per la varianza.
  return(list(mean_model = mean_model, variance_model = variance_model))
}