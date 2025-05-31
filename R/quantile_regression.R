# predizione di petallength in regressione
# Installazione pacchetti se non già presenti
if (!require("quantreg")) install.packages("quantreg")
if (!require("dplyr")) install.packages("dplyr")

# Caricamento pacchetti
library(quantreg)
library(dplyr)

# Lettura del dataset
data <- read.csv("C:/Users/PC/Desktop/STATS/Iris dataset.csv")
# Usiamo Petal.Length come target

# Suddivisione in train (60%), calibrazione (20%) e test (20%)
set.seed(42)
n <- nrow(data)
train_index <- sample(1:n, size = 0.6 * n)
remaining <- setdiff(1:n, train_index)
calib_index <- sample(remaining, size = 0.5 * length(remaining))
test_index <- setdiff(remaining, calib_index)

# Rinominare il target per comodità
names(data)[which(names(data) == "PetalLengthCm")] <- "Petal.Length"

train_data <- data[train_index, ]
calib_data <- data[calib_index, ]
test_data <- data[test_index, ]

# Rimuoviamo le colonne inutili
data <- data %>% select(-Id, -Species)

# Conferma dimensioni
cat("Train:", nrow(train_data), "Calibrazione:", nrow(calib_data), "Test:", nrow(test_data), "\n")

if (!require("e1071")) install.packages("e1071")  # SVM

head(data)
str(data)


# 1. QUANTILE REGRESSION
library(quantreg)

# Allenamento dei due modelli quantili (5° e 95° percentile)
mod_05 <- rq(Petal.Length ~ ., data = train_data, tau = 0.05)
mod_95 <- rq(Petal.Length ~ ., data = train_data, tau = 0.95)

# Predizione su calibrazione
q_05_calib <- predict(mod_05, newdata = calib_data)
q_95_calib <- predict(mod_95, newdata = calib_data)

# Score conformali (distanza dai quantili)
scores <- pmax(q_05_calib - calib_data$Petal.Length,
               calib_data$Petal.Length - q_95_calib)

# Calcolo quantile conformale (es. alpha = 0.1 → intervallo al 90%)
alpha <- 0.1
q_conformal <- quantile(scores, probs = 1 - alpha, type = 1)
cat("Quantile conformale q^ =", q_conformal, "\n")

# plot results
if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)
# Dataframe con i risultati
results <- data.frame(
  index = 1:length(Y_test),
  true_value = Y_test,
  lower = lower,
  upper = upper,
  midpoint = (lower + upper) / 2
)
# try on test con  quantili 0.05 e 0.95
# Previsioni dei quantili sul test set
q_05_test <- predict(mod_05, newdata = test_data)
q_95_test <- predict(mod_95, newdata = test_data)

# Costruzione intervallo conformale con q̂
lower <- q_05_test - abs(q_conformal)
upper <- q_95_test + abs(q_conformal)

# Estrai il valore reale di Petal.Length
Y_test <- test_data$Petal.Length

# Verifica se il valore reale rientra nell'intervallo
covered <- (Y_test >= lower) & (Y_test <= upper)
coverage <- mean(covered)
cat("Copertura empirica sul test set =", round(coverage, 3), "\n")
if (!require("ggplot2")) install.packages("ggplot2")
library(ggplot2)

results <- data.frame(
  index = 1:length(Y_test),
  true_value = Y_test,
  lower = lower,
  upper = upper,
  midpoint = (lower + upper) / 2
)

ggplot(results, aes(x = index)) +
  geom_point(aes(y = true_value), color = "black", size = 1.8) +
  geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "red") +
  labs(title = "Intervalli predittivi conformali (Quantile Regression)",
       x = "Osservazione nel test set",
       y = "Petal.Length") +
  theme_minimal()

# try on test con altri valori di tau (es. 0.025 / 0.975) dato che prima abbiamo
#ottenuto un 80% di copertura su 90%

# Nuovi quantili più estremi
mod_025 <- rq(Petal.Length ~ ., data = train_data, tau = 0.025)
mod_975 <- rq(Petal.Length ~ ., data = train_data, tau = 0.975)

# Predizioni su calibrazione
q_025_calib <- predict(mod_025, newdata = calib_data)
q_975_calib <- predict(mod_975, newdata = calib_data)

# Score conformali
scores_extreme <- pmax(q_025_calib - calib_data$Petal.Length,
                       calib_data$Petal.Length - q_975_calib)

# Quantile conformale
q_conformal_extreme <- quantile(scores_extreme, probs = 1 - alpha, type = 1)
cat("Nuovo quantile conformale q^ =", q_conformal_extreme, "\n")

# Previsioni su test set
q_025_test <- predict(mod_025, newdata = test_data)
q_975_test <- predict(mod_975, newdata = test_data)

# Costruzione dell’intervallo conforme sul test set
lower_ext <- q_025_test - abs(q_conformal_extreme)
upper_ext <- q_975_test + abs(q_conformal_extreme)

# Valori reali di Petal.Length nel test set
Y_test <- test_data$Petal.Length

# plot 
# Costruzione dataframe con intervalli estremi
results_extreme <- data.frame(
  index = 1:length(Y_test),
  true_value = Y_test,
  lower = lower_ext,
  upper = upper_ext,
  midpoint = (lower_ext + upper_ext) / 2
)

# Grafico
library(ggplot2)

ggplot(results_extreme, aes(x = index)) +
  geom_point(aes(y = true_value), color = "black", size = 1.8) +
  geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "darkred") +
  labs(
    title = "Intervalli predittivi conformali (Quantile Regression estrema)",
    x = "Osservazione nel test set",
    y = "Petal.Length"
  ) +
  theme_minimal()
# calcolo copertura empirica
covered_ext <- (Y_test >= lower_ext) & (Y_test <= upper_ext)
coverage_ext <- mean(covered_ext)
cat("Copertura empirica con intervalli estremi =", round(coverage_ext, 3), "\n")

#l'intervallo si adatta e compensa l'errore residuo nel modello quantile,
#come previsto dal metodo conformale, con una copertura empirica con intervalli 
# estremi = 0.967 very good

# EVALUATING PREDICTIONS SETS

# 1. Copertura empirica
covered_cqr <- (Y_test >= lower_ext) & (Y_test <= upper_ext)
coverage_cqr <- mean(covered_cqr)

# Ampiezza intervallo
width_cqr <- upper_ext - lower_ext
avg_width_cqr <- mean(width_cqr)

cat("Copertura empirica (CQR) =", round(coverage_cqr, 3), "\n")
cat("Ampiezza media dell'intervallo (CQR) =", round(avg_width_cqr, 3), "\n")
# alta copertura (96,7%), superiore a quella richiesta (90%) e intervalli
# non troppo larghi, ragionevoli 

# 2. VALUTAZIONE ADATTIVITA ATTRAVERSO Relazione tra errore e ampiezza
residuals_abs <- abs(Y_test - (lower_ext + upper_ext) / 2)

# Relazione tra residuo e ampiezza
plot(width_cqr, residuals_abs,
     xlab = "Ampiezza intervallo predittivo (CQR)",
     ylab = "Errore assoluto |Y - centro intervallo|",
     main = "Adattività degli intervalli (Quantile Regression)")
abline(lm(residuals_abs ~ width_cqr), col = "red", lty = 2)

# VALUTAZIONE ADATTIVITA ATTRAVERSO FSC/SSC

# FSC:
# Prendiamo Sepal.Width dal test set
sepal_width <- test_data$SepalWidthCm

# Suddividiamo in due gruppi
group <- ifelse(sepal_width <= median(sepal_width), "Basso", "Alto")

# Calcoliamo copertura per ogni gruppo
group_df <- data.frame(group = group, covered = covered_cqr)

# Media copertura per ciascun gruppo
fsc_coverage <- tapply(group_df$covered, group_df$group, mean)
print(fsc_coverage)
# copertura molto elevata in entrambi i gruppi:
# gruppo con Sepal.Width più alto  ha copertura prefetta
#  gruppo con Sepal.Width più basso è comunque elevata 93%
# metodo CQR garantisce copertura stabile anche tra sottogruppi di input,
# modello è robusto e ben calibrato in termini di conditional validity

# PLOT FSC
fsc_coverage <- tapply(group_df$covered, group_df$group, mean)
library(ggplot2)

# Preparazione dati per il grafico
fsc_df <- data.frame(
  gruppo = names(fsc_coverage),
  copertura = as.numeric(fsc_coverage)
)

# Grafico a barre
ggplot(fsc_df, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "Feature-Stratified Coverage (FSC) – Quantile Regression",
       x = "Gruppo Sepal.Width",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")

# SSC:
# Crea intervalli di ampiezza
quantiles <- quantile(width_cqr, probs = c(0.33, 0.66))

# Bin su ampiezza
interval_size <- cut(width_cqr,
                     breaks = c(-Inf, quantiles[1], quantiles[2], Inf),
                     labels = c("Piccolo", "Medio", "Grande"))

# Calcoliamo copertura per ampiezza
ssc_df <- data.frame(bin = interval_size, covered = covered_cqr)

ssc_coverage <- tapply(ssc_df$covered, ssc_df$bin, mean)
print(ssc_coverage)
# copertura ≥ 90% in tutti i sottogruppi. anche intervalli piccoli coprono bene
# l metodo adatta la larghezza degli intervalli alla difficoltà, mantenendo la
# validità 

# PLOT SSC 
ssc_df_plot <- data.frame(gruppo = names(ssc_coverage),
                          copertura = as.numeric(ssc_coverage))

ggplot(ssc_df_plot, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "Set-Stratified Coverage (SSC) – Quantile Regression",
       x = "Ampiezza intervallo",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")




