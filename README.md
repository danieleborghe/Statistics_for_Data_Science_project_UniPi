# Conformal Prediction for Uncertainty Quantification in R

## Panoramica del Progetto

Questo progetto offre un'implementazione e un'analisi approfondita di diverse tecniche di **Conformal Prediction** (Predizione Conforme), come descritto nell'articolo scientifico _"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"_ di Anastasios N. Angelopoulos e Stephen Bates.

Il progetto utilizza il classico dataset **Iris** e implementa i metodi di predizione conforme su modelli **Support Vector Machine (SVM)**, sia per compiti di **classificazione** che di **regressione**.

Il codice è scritto interamente in **R** e si articola in tre fasi principali:
1.  **Esecuzione di Esperimenti**: Simulazioni multiple (100 run) per ogni metodo, al fine di valutarne la copertura marginale, e un'esecuzione singola dettagliata per analizzare le performance in modo riproducibile.
2.  **Valutazione delle Performance**: Analisi dei risultati attraverso metriche rigorose come la **Feature-Stratified Coverage (FSC)** e la **Set-size Stratified Coverage (SSC)** per misurare l'adattività dei metodi.
3.  **Analisi Comparativa**: Creazione di grafici e tabelle per confrontare l'efficienza (dimensione degli insiemi, larghezza degli intervalli) e la robustezza dei diversi approcci.

---

## Struttura delle Cartelle

Il progetto è organizzato in modo modulare per garantire chiarezza e manutenibilità.

.
├── R/
│   ├── comparative_analyses/ # Script per analisi comparative tra esperimenti
│   ├── experiments/          # Script per eseguire i singoli esperimenti
│   ├── conformal_predictors.R
│   ├── evaluation_utils.R
│   └── experimentation_utils.R
├── results/
│   ├── tables/               # Tabelle CSV con i risultati grezzi e aggregati
│   └── plots/                # Grafici e istogrammi comparativi
├── 26.ConformalPrediction.pdf
└── README.md

### Spiegazione delle Cartelle

* **`/R`**: Contiene tutto il codice sorgente in R.
    * **`/R/experiments`**: Ogni script in questa cartella esegue un esperimento completo per uno specifico metodo di predizione conforme descritto nel paper.
    * **`/R/comparative_analyses`**: Contiene gli script che caricano i risultati salvati dagli esperimenti per creare visualizzazioni e analisi comparative.
* **`/results`**: Contiene tutti gli output generati dagli script.
    * **`/results/tables`**: Sottocartelle separate per ogni esperimento contenenti file CSV dettagliati, come le distribuzioni di copertura, le dimensioni degli insiemi, i dati per l'analisi di adattività, ecc.
    * **`/results/plots`**: Grafici (in formato `.png`) che visualizzano i confronti tra i metodi, come istogrammi di copertura e analisi di adattività.

---

## Spiegazione degli Script

### Script di Utilità (`R/`)

Questi script contengono funzioni riutilizzate in tutto il progetto, separando la logica di base dall'esecuzione degli esperimenti.

* `R/experimentation_utils.R`
    * **Scopo**: Fornisce funzioni ausiliarie per la gestione degli esperimenti.
    * **Funzioni Principali**:
        * `check_and_load_packages()`: Gestisce le dipendenze, installando e caricando i pacchetti R necessari.
        * `load_iris_for_classification()` / `load_iris_for_regression()`: Caricano e preparano il dataset Iris per i rispettivi task (es. rimozione della colonna `Species` per la regressione).
        * `train_svm_classification_model()` / `train_svm_regression_model()`: Addestrano i modelli SVM di base.
        * `predict_svm_probabilities()` / `predict_svm_regression_values()`: Eseguono predizioni utilizzando i modelli SVM addestrati.
        * Funzioni specializzate come `train_quantile_models()`, `train_primary_and_uncertainty_models()` e `train_mean_and_stddev_models()` per addestrare i modelli specifici richiesti dagli esperimenti di regressione.

* `R/conformal_predictors.R`
    * **Scopo**: Implementa la logica fondamentale dei diversi algoritmi di predizione conforme.
    * **Funzioni Principali**:
        * `calculate_q_hat()`: Funzione centrale che calcola il quantile `q_hat` corretto per la dimensione del campione di calibrazione. Questa funzione è condivisa da tutti i metodi.
        * `get_non_conformity_scores_*()`: Una serie di funzioni, una per ogni metodo (es. `_basic`, `_adaptive`, `_bayes`, `_quantile`), che calcolano i punteggi di non-conformità specifici per quel metodo.
        * `create_prediction_sets_*()` / `create_prediction_intervals_*()`: Funzioni che, dato un `q_hat`, costruiscono gli insiemi o gli intervalli di predizione finali per i nuovi dati.

* `R/evaluation_utils.R`
    * **Scopo**: Contiene le funzioni per valutare le performance dei predittori conformi, seguendo le metriche della Sezione 4 del paper.
    * **Funzioni Principali**:
        * `calculate_empirical_coverage()`: Calcola la copertura marginale (la percentuale di volte in cui il valore vero è contenuto nell'insieme di predizione).
        * `get_set_sizes()`: Estrae le dimensioni degli insiemi di predizione.
        * `calculate_fsc()`: Implementa la **Feature-Stratified Coverage**, che valuta se la copertura è mantenuta stratificando i dati in base ai valori di una feature (es. `Sepal.Length`).
        * `calculate_ssc()`: Implementa la **Set-size Stratified Coverage**, che valuta la copertura stratificando i dati in base alla dimensione dell'insieme di predizione.

### Script degli Esperimenti (`R/experiments/`)

Questi script sono il cuore esecutivo del progetto. Ognuno è autonomo e dedicato a un singolo metodo.

* `R/experiments/sec1_run_experiment.R`: Esegue l'esperimento per il metodo di **Predizione Conforme Base** (Classificazione).
* `R/experiments/sec2_1_run_experiment.R`: Esegue l'esperimento per gli **Insiemi di Predizione Adattivi** (Classificazione), che mirano a regolare la dimensione dell'insieme in base alla difficoltà dell'istanza.
* `R/experiments/sec2_2_run_experiment.R`: Esegue l'esperimento per la **Regressione Quantile Conforme** (Regressione), che calibra gli intervalli predetti da modelli di regressione quantile.
* `R/experiments/sec2_3_scalar_run_experiment.R`: Esegue l'esperimento di **Incertezza Scalare** (Regressione), dove l'incertezza è modellata da un secondo SVM che predice i residui assoluti del modello primario.
* `R/experiments/sec2_3_stddev_run_experiment.R`: Simile al precedente, ma l'incertezza è modellata stimando la deviazione standard dell'errore (predicendo i residui al quadrato).
* `R/experiments/sec2_4_run_experiment.R`: Esegue l'esperimento **Conformalizing Bayes** (Classificazione), che utilizza la densità predittiva a posteriori come score per ottenere insiemi teoricamente ottimali in termini di dimensione.

### Script di Analisi Comparativa (`R/comparative_analyses/`)

Questi script aggregano i risultati dei singoli esperimenti per effettuare confronti diretti.

* `R/comparative_analyses/run_comparative_analysis.R`: Genera istogrammi che confrontano le **distribuzioni di copertura marginale** tra i metodi di classificazione e tra quelli di regressione.
* `R/comparative_analyses/run_ssc_fsc_comparative_analysis.R`: Crea grafici a barre affiancati che confrontano le metriche di copertura condizionale **FSC** e **SSC** tra i diversi metodi, sia per la classificazione che per la regressione.
* `R/comparative_analyses/run_width_comparative_analysis.R`: Analizza e visualizza la **distribuzione delle dimensioni degli insiemi di predizione** per i metodi di classificazione, evidenziando le differenze nella loro efficienza.
* `R/comparative_analyses/run_adaptiveness_analysis.R`: Genera grafici a dispersione per analizzare visivamente l'**adattività** dei metodi di regressione, mettendo in relazione i punteggi di non-conformità con l'errore di predizione assoluto. Una correlazione positiva indica una buona adattività.