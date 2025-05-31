# Installazione pacchetti se non già presenti
if (!require("dplyr")) install.packages("dplyr")

# Caricamento pacchetti
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

# SCALAR UNCERATINTY ESTIMATES WITH RESIDUALS
library(e1071)

# Modello di regressione SVM
svm_model <- svm(Petal.Length ~ ., data = train_data, type = "eps-regression")

# Predizione su calibrazione
pred_calib <- predict(svm_model, newdata = calib_data)

# Calcolo residui assoluti
residuals_calib <- abs(calib_data$Petal.Length - pred_calib)

# Alleniamo un secondo modello per predire i residui (û(X))
residual_model <- svm(residuals_calib ~ ., data = calib_data[, -which(names(calib_data) == "Petal.Length")])


# CALCOLO SCORE CONFORMALE
# Stima dell'incertezza û(X) su calibrazione
u_hat_calib <- predict(residual_model, newdata = calib_data)

# Score conformale (come nel paper, sezione 2.3.2)
scores_svm <- residuals_calib / u_hat_calib

# Calcolo quantile conformale
q_svm <- quantile(scores_svm, probs = 1 - alpha, type = 1)
cat("Quantile conformale (SVM) q̂ =", q_svm, "\n")

# COSTRUZIONE INTERVALLO SU TEST SET
# Predizione punto centrale f̂(X)
pred_test <- predict(svm_model, newdata = test_data)

# Stima incertezza û(X) sul test
u_hat_test <- predict(residual_model, newdata = test_data)

# Intervallo conforme
lower_svm <- pred_test - q_svm * u_hat_test
upper_svm <- pred_test + q_svm * u_hat_test
Y_test <- test_data$Petal.Length

# COPERTURA EMPIRICA
covered_svm <- (Y_test >= lower_svm) & (Y_test <= upper_svm)
coverage_svm <- mean(covered_svm)
cat("Copertura empirica (SVM conformale) =", round(coverage_svm, 3), "\n")

# PLOT 
library(ggplot2)

# Costruzione dataframe dei risultati
results_svm <- data.frame(
  index = 1:length(Y_test),
  true_value = Y_test,
  lower = lower_svm,
  upper = upper_svm,
  midpoint = pred_test
)

# Grafico
ggplot(results_svm, aes(x = index)) +
  geom_point(aes(y = true_value), color = "black", size = 1.8) +  # Valore reale
  geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +  # Predizione
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "darkgreen") +
  labs(
    title = "Intervalli predittivi conformali (SVM – Scalar Uncertainty Estimates)",
    x = "Osservazione nel test set",
    y = "Petal.Length"
  ) +
  theme_minimal()
# copertura empirica bassa (73.3%),intervalli sono più stretti e meno adattivi
# rispetto alla quantile regression -->  stima dell’incertezza u(X) non molto
# efficace in questo caso

# SCALAR UNCERATINTY ESTIMATES WITH STANDARD DEVIATION DEI RESIDUI
svm_mean <- svm(Petal.Length ~ ., data = train_data, type = "eps-regression")
fhat_calib <- predict(svm_mean, newdata = calib_data)

# Calcola residui al quadrato (simili a varianza)
residuals_sq <- (calib_data$Petal.Length - fhat_calib)^2

# Modello che predice la varianza → deviazione standard stimata = sqrt(predizione)
stddev_model <- svm(residuals_sq ~ ., data = calib_data[, -which(names(calib_data) == "Petal.Length")])
sigma_hat_calib <- sqrt(predict(stddev_model, newdata = calib_data))

scores_stddev <- abs(calib_data$Petal.Length - fhat_calib) / sigma_hat_calib
q_stddev <- quantile(scores_stddev, probs = 1 - alpha, type = 1)
cat("Quantile conformale q̂ =", q_stddev, "\n")

fhat_test <- predict(svm_mean, newdata = test_data)
sigma_hat_test <- sqrt(predict(stddev_model, newdata = test_data))

lower_stddev <- fhat_test - q_stddev * sigma_hat_test
upper_stddev <- fhat_test + q_stddev * sigma_hat_test
Y_test <- test_data$Petal.Length

covered_stddev <- (Y_test >= lower_stddev) & (Y_test <= upper_stddev)
coverage_stddev <- mean(covered_stddev)
cat("Copertura empirica (Estimated Std Dev) =", round(coverage_stddev, 3), "\n")

# PLOT
library(ggplot2)

# Costruzione dataframe dei risultati
results_stddev <- data.frame(
  index = 1:length(Y_test),
  true_value = Y_test,
  lower = lower_stddev,
  upper = upper_stddev,
  midpoint = fhat_test
)

# Grafico
ggplot(results_stddev, aes(x = index)) +
  geom_point(aes(y = true_value), color = "black", size = 1.8) +
  geom_line(aes(y = midpoint), color = "blue", linetype = "dashed") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.25, color = "orange") +
  labs(
    title = "Intervalli predittivi conformali (SVM – Estimated Standard Deviation)",
    x = "Osservazione nel test set",
    y = "Petal.Length"
  ) +
  theme_minimal()

# EVALUATING PREDICTIONS SETS

# 1. MODELLO SOLO RESIDUI
# ADATTIVITA: ERRORE VS AMPIEZZA
covered_svm <- (Y_test >= lower_svm) & (Y_test <= upper_svm)
width_svm <- upper_svm - lower_svm

coverage_svm <- mean(covered_svm)
avg_width_svm <- mean(width_svm)

cat("Copertura empirica (residui) =", round(coverage_svm, 3), "\n")
cat("Ampiezza media (residui) =", round(avg_width_svm, 3), "\n")
residuals_abs_svm <- abs(Y_test - (lower_svm + upper_svm) / 2)

plot(width_svm, residuals_abs_svm,
     xlab = "Ampiezza intervallo (SVM residui)",
     ylab = "Errore assoluto |Y - centro|",
     main = "Adattività (SVM + residui)")
abline(lm(residuals_abs_svm ~ width_svm), col = "red", lty = 2)
#  relazione molto debole o nulla tra ampiezza dell’intervallo e errore 
# assoluto, errori grandi si verificano anche con intervalli stretti,
# metodo non riesce ad adattare bene la larghezza dell’intervallo alla
# difficoltà del punto

# FSC
group_svm <- ifelse(test_data$SepalWidthCm <= median(test_data$SepalWidthCm),
                    "Basso", "Alto")
group_df_svm <- data.frame(group = group_svm, covered = covered_svm)
fsc_coverage_svm <- tapply(group_df_svm$covered, group_df_svm$group, mean)
print(fsc_coverage_svm)

# SSC
quantiles_svm <- quantile(width_svm, probs = c(0.33, 0.66))
interval_size_svm <- cut(width_svm,
                         breaks = c(-Inf, quantiles_svm[1], quantiles_svm[2], Inf),
                         labels = c("Piccolo", "Medio", "Grande"))
ssc_df_svm <- data.frame(bin = interval_size_svm, covered = covered_svm)
ssc_coverage_svm <- tapply(ssc_df_svm$covered, ssc_df_svm$bin, mean)
print(ssc_coverage_svm)
# FSC:  Copertura bassa, specialmente nel gruppo Basso → problemi di validità 
# condizionata
# SSC: nessun gruppo raggiunge 90% di copertura,  intervalli non si adattano 
# bene alla difficoltà

# PLOT FSC
fsc_coverage_svm <- tapply(group_df_svm$covered, group_df_svm$group, mean)
fsc_df_svm <- data.frame(
  gruppo = names(fsc_coverage_svm),
  copertura = as.numeric(fsc_coverage_svm)
)

ggplot(fsc_df_svm, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "FSC – Scalar Uncertainty Estimates (SVM residui)",
       x = "Gruppo Sepal.Width",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")

# PLOT SSC
ssc_coverage_svm <- tapply(ssc_df_svm$covered, ssc_df_svm$bin, mean)
ssc_df_plot_svm <- data.frame(
  gruppo = names(ssc_coverage_svm),
  copertura = as.numeric(ssc_coverage_svm)
)

ggplot(ssc_df_plot_svm, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "SSC – Scalar Uncertainty Estimates (SVM residui)",
       x = "Ampiezza intervallo",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")

# MODELLO DELLA DEV STD DEI RESIDUI
coverage_stddev <- mean(covered_stddev)
width_stddev <- upper_stddev - lower_stddev
avg_width_stddev <- mean(width_stddev)

cat("Copertura empirica (Estimated Std Dev) =", round(coverage_stddev, 3), "\n")
cat("Ampiezza media intervallo (Std Dev) =", round(avg_width_stddev, 3), "\n")

residuals_abs_stddev <- abs(Y_test - (lower_stddev + upper_stddev) / 2)

plot(width_stddev, residuals_abs_stddev,
     xlab = "Ampiezza intervallo (Std Dev)",
     ylab = "Errore assoluto |Y - centro|",
     main = "Adattività (Estimated Std Dev)")
abline(lm(residuals_abs_stddev ~ width_stddev), col = "red", lty = 2)
#  correlazione positiva, anche se non fortissima, intervalli tendono ad 
# allargarsi dove il modello è più incerto.  metodo è parzialmente adattivo,
# meglio del modello basato su residui assoluti


# FSC
group_stddev <- ifelse(test_data$SepalWidthCm <= median(test_data$SepalWidthCm),
                       "Basso", "Alto")
group_df_stddev <- data.frame(group = group_stddev, covered = covered_stddev)
fsc_coverage_stddev <- tapply(group_df_stddev$covered, group_df_stddev$group, mean)
print(fsc_coverage_stddev)

#SSC
quantiles_stddev <- quantile(width_stddev, probs = c(0.33, 0.66))
interval_size_stddev <- cut(width_stddev,
                            breaks = c(-Inf, quantiles_stddev[1], quantiles_stddev[2], Inf),
                            labels = c("Piccolo", "Medio", "Grande"))
ssc_df_stddev <- data.frame(bin = interval_size_stddev, covered = covered_stddev)
ssc_coverage_stddev <- tapply(ssc_df_stddev$covered, ssc_df_stddev$bin, mean)
print(ssc_coverage_stddev)
# FSC:  copertura non è uniforme e scende sotto il 90% → validità condizionata
# rispettata
# SSC: intervalli più ampi non garantiscono copertura migliore, quindi il metodo
# non adattivo

# PLOT FSC
fsc_coverage_stddev <- tapply(group_df_stddev$covered, group_df_stddev$group, mean)
library(ggplot2)

fsc_df_stddev <- data.frame(
  gruppo = names(fsc_coverage_stddev),
  copertura = as.numeric(fsc_coverage_stddev)
)

ggplot(fsc_df_stddev, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "FSC – Estimated Std Dev",
       x = "Gruppo Sepal.Width",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")


# PLOT SSC
ssc_coverage_stddev <- tapply(ssc_df_stddev$covered, ssc_df_stddev$bin, mean)
ssc_df_plot_stddev <- data.frame(
  gruppo = names(ssc_coverage_stddev),
  copertura = as.numeric(ssc_coverage_stddev)
)

ggplot(ssc_df_plot_stddev, aes(x = gruppo, y = copertura, fill = gruppo)) +
  geom_col(width = 0.6) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "red") +
  ylim(0, 1.05) +
  labs(title = "SSC – Estimated Std Dev",
       x = "Ampiezza intervallo",
       y = "Copertura empirica") +
  theme_minimal() +
  theme(legend.position = "none")








