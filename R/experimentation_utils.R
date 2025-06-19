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
#   - train_svm_model(...): Addestra un modello SVM per la classificazione.
#   - predict_svm_probabilities(...): Predice le probabilità di classe da un modello SVM addestrato.
#   - save_detailed_test_predictions(...): Salva un CSV dettagliato delle predizioni di test.
#   - save_plot_to_png(...)
#   - save_table_to_csv(...)
#   - print_summary_statistics(...)

check_and_load_packages <- function(required_packages) {
  # Scopo: Controlla se una lista di pacchetti è installata, installa quelli mancanti
  #        e poi li carica nella sessione corrente.
  #
  # Parametri:
  #   - required_packages: Un vettore di caratteri con i nomi dei pacchetti.
  #
  # Ritorna: Nessuno. Stampa messaggi sulla console e carica i pacchetti.
  
  # Step 1: Inizia il processo di controllo e caricamento dei pacchetti.
  cat("INFO: Avvio controllo e caricamento dipendenze pacchetti...\n")
  
  # Step 2: Itera su ciascun pacchetto richiesto.
  for (pkg in required_packages) {
    # Step 2.1: Controlla se il pacchetto è già installato.
    if (!requireNamespace(pkg, quietly = TRUE)) {
      # Step 2.2: Se il pacchetto non è trovato, tenta di installarlo.
      cat(paste("INFO: Pacchetto '", pkg, "' non trovato. Tentativo di installazione...\n", sep = ""))
      install.packages(pkg, dependencies = TRUE)
    }
    
    # Step 2.3: Carica il pacchetto, gestendo eventuali errori.
    tryCatch({
      library(pkg, character.only = TRUE)
      # Non è più necessario stampare il successo del caricamento qui,
      # il messaggio finale di completamento sarà sufficiente.
    }, error = function(e) {
      warning(paste("WARN: Fallito il caricamento del pacchetto '", pkg, "'. Per favore, verifica l'installazione.\nErrore: ", e$message, sep = ""))
    })
  }
  # Step 3: Conclude il processo di controllo e caricamento dei pacchetti.
  cat("INFO: Controllo e caricamento dipendenze pacchetti completato.\n")
}


load_iris_for_classification <- function() {
  # Scopo: Carica il dataset Iris incorporato in R, si assicura che 'Species' sia un fattore,
  #        e rimescola il dataset.
  #
  # Parametri: Nessuno
  #
  # Ritorna: Un data frame contenente il dataset Iris rimescolato, pronto per la classificazione.
  
  # Step 1: Carica il dataset Iris.
  cat("INFO: Caricamento dataset Iris per la classificazione...\n")
  data(iris) # Carica il dataset Iris dai dataset predefiniti di R
  
  # Step 2: Assicura che la colonna 'Species' sia trattata come un fattore.
  # Questo è cruciale per i modelli di classificazione.
  iris$Species <- as.factor(iris$Species)
  
  # Step 3: Rimescola il dataset per garantire un campionamento casuale nelle fasi successive.
  # Imposta un seed per la riproducibilità del rimescolamento.
  set.seed(123)
  iris_shuffled <- iris[sample(nrow(iris)), ]
  
  # Step 4: Restituisce il dataset Iris rimescolato e preparato.
  cat("INFO: Dataset Iris caricato, 'Species' convertito a fattore, e dati rimescolati.\n")
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
  cat("INFO: Caricamento dataset Iris per la regressione...\n")
  data(iris)
  
  # Step 2: Rimuove la colonna 'Species', che non è necessaria per la regressione.
  iris_regression <- iris[, -which(names(iris) == "Species")]
  
  # Step 3: Rimescola il dataset per garantire casualità.
  set.seed(123)
  shuffled_iris <- iris_regression[sample(nrow(iris_regression)), ]
  
  # Step 4: Restituisce il dataset Iris rimescolato e preparato per la regressione.
  cat("INFO: Dataset Iris caricato e rimescolato per il task di regressione.\n")
  return(shuffled_iris)
}


train_svm_model <- function(formula, training_data) {
  # Scopo: Addestra un modello SVM per la classificazione utilizzando la formula specificata
  #        e i dati di addestramento, con l'abilitazione della stima della probabilità.
  #
  # Parametri:
  #   - formula: Una formula R che specifica il modello (es. `Species ~ .`).
  #   - training_data: Un data frame contenente i dati di addestramento.
  #
  # Ritorna: Un oggetto modello SVM addestrato dal pacchetto 'e1071'.
  
  # Step 1: Inizia l'addestramento del modello SVM.
  cat("INFO: Addestramento modello SVM per la classificazione...\n")
  
  # Step 2: Estrai il nome della variabile target dalla formula.
  target_var_name <- all.vars(formula)[1]
  
  # Step 3: Verifica che la variabile target sia un fattore.
  # Questo è un requisito per la classificazione SVM.
  if (!is.factor(training_data[[target_var_name]])) {
    stop(paste("ERRORE: La variabile target '", target_var_name, "' deve essere un fattore per la classificazione SVM.", sep = ""))
  }
  
  # Step 4: Addestra il modello SVM.
  # `kernel = "radial"` è un kernel comune per SVM.
  # `probability = TRUE` è essenziale per ottenere le probabilità di classe.
  # `scale = TRUE` scala i dati prima dell'addestramento.
  model <- e1071::svm(formula, data = training_data, kernel = "radial", probability = TRUE, scale = TRUE)
  
  # Step 5: Conclude l'addestramento del modello SVM.
  cat("INFO: Addestramento modello SVM completato.\n")
  return(model)
}

predict_svm_probabilities <- function(svm_model, newdata) {
  # Scopo: Predice le probabilità di classe per nuovi dati utilizzando un modello SVM di classificazione addestrato.
  #
  # Parametri:
  #   - svm_model: Un oggetto modello SVM addestrato dalla funzione `train_svm_model()`.
  #   - newdata: Un data frame per il quale predire le probabilità.
  #
  # Ritorna: Una matrice di probabilità di classe, dove le righe sono i campioni e le colonne sono le classi.
  
  # Step 1: Effettua le predizioni utilizzando il modello SVM.
  # Assicurati che `probability = TRUE` sia passato per ottenere l'attributo delle probabilità.
  predictions_object <- predict(svm_model, newdata, probability = TRUE)
  
  # Step 2: Estrai l'attributo delle probabilità dall'oggetto di predizione.
  probabilities <- attr(predictions_object, "probabilities")
  
  # Step 3: Verifica che le probabilità siano state recuperate correttamente.
  # Se `is.null(probabilities)` è vero, significa che il modello non è stato addestrato con `probability=TRUE`.
  if (is.null(probabilities)) {
    stop("ERRORE: Impossibile recuperare le probabilità dalla predizione SVM. Il modello è stato addestrato con probability=TRUE?")
  }
  # Step 4: Restituisce la matrice delle probabilità.
  return(probabilities)
}


save_detailed_test_predictions <- function(test_true_labels,
                                           prediction_sets_list,
                                           test_probs_matrix,
                                           output_directory,
                                           base_filename) {
  # Scopo: Crea e salva un file CSV con informazioni dettagliate per ogni campione di test.
  #
  # Parametri:
  #   - test_true_labels: Vettore delle vere etichette di classe per il set di test.
  #   - prediction_sets_list: Lista dove ogni elemento è un vettore di caratteri (insieme di predizione).
  #   - test_probs_matrix: Matrice delle probabilità di classe predette per il set di test.
  #   - output_directory: Il percorso della directory dove verrà salvato il file CSV.
  #   - base_filename: Il nome base per il file CSV.
  #
  # Ritorna: Restituisce in modo invisibile il percorso completo al file salvato, o NULL se il salvataggio è fallito.
  
  # Step 1: Prepara per il salvataggio dei risultati dettagliati.
  cat(paste0("INFO: Preparazione per il salvataggio delle predizioni di test dettagliate in '", base_filename, "'...\n"))
  
  # Step 2: Esegue un controllo di coerenza sugli argomenti di input.
  # Assicura che tutte le liste e matrici abbiano lunghezze/righe corrispondenti.
  if (length(test_true_labels) != length(prediction_sets_list) ||
      length(test_true_labels) != nrow(test_probs_matrix)) {
    stop("ERRORE: Gli argomenti di input per save_detailed_test_predictions hanno lunghezze/righe non corrispondenti.")
  }
  
  # Step 3: Converte le liste di insiemi di predizione in stringhe concatenate per il CSV.
  # Se un set è vuoto, viene rappresentato da una stringa vuota; altrimenti, gli elementi sono ordinati e uniti da "; ".
  pred_set_strings <- sapply(prediction_sets_list, function(s) {
    if (length(s) == 0) "" else paste(sort(s), collapse = "; ")
  })
  
  # Step 4: Crea il data frame dettagliato dei risultati.
  # Aggiunge un prefisso "Prob_" ai nomi delle colonne di probabilità.
  detailed_results_df <- as.data.frame(test_probs_matrix)
  names(detailed_results_df) <- paste0("Prob_", names(detailed_results_df))
  detailed_results_df <- cbind(
    data.frame(
      SampleID = 1:length(test_true_labels),
      TrueLabel = as.character(test_true_labels),
      PredictionSet = pred_set_strings
    ),
    detailed_results_df
  )
  
  # Step 5: Crea la directory di output se non esiste.
  # `showWarnings = FALSE` evita avvisi se la directory esiste già.
  # `recursive = TRUE` crea le sottodirectory necessarie.
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  detailed_filename_path <- file.path(output_directory, base_filename)
  
  # Step 6: Tenta di scrivere il data frame in un file CSV.
  # Utilizza `tryCatch` per gestire gli errori durante il salvataggio.
  tryCatch({
    write.csv(detailed_results_df, detailed_filename_path, row.names = FALSE)
    cat(paste0("INFO: Predizioni di test dettagliate salvate in '", detailed_filename_path, "'\n"))
    return(invisible(detailed_filename_path)) # Restituisce il percorso in modo invisibile
  }, error = function(e) {
    warning(paste0("WARN: Fallito il salvataggio delle predizioni di test dettagliate. Errore: ", e$message))
    return(invisible(NULL)) # Restituisce NULL in caso di fallimento
  })
}