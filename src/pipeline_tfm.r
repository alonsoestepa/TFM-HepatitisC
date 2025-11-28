# ------------------------------------------------------------
# TFM Hepatitis C - Pipeline completo en R
# ------------------------------------------------------------

set.seed(12345)

# Paquetes necesarios
library(timeDate)
library(caret)
library(tidyverse)
library(dplyr)
library(janitor)
library(ggplot2)
library(tidyr)
library(pROC)
library(randomForest)
library(psych)
library(knitr)
library(MASS)   # stepAIC
library(broom)  # tidy()

# Carpeta de salida
dir.create("output", showWarnings = FALSE)

# 1) Carga de datos y limpieza básica -------------------------

df <- read_csv("hepatitisC.csv")

if ("Unnamed: 0" %in% names(df)) {
  df <- df %>% select(-`Unnamed: 0`)
}

df <- janitor::clean_names(df)

if ("x1" %in% names(df)) {
  df <- df %>% select(-x1)
}

# 2) Definición de la variable resultado ----------------------

df <- df %>%
  mutate(
    status = case_when(
      category %in% c("0=Blood Donor", "0s=suspect Blood Donor") ~ "Control",
      category %in% c("1=Hepatitis", "2=Fibrosis", "3=Cirrhosis") ~ "HCV",
      TRUE ~ NA_character_
    ),
    status = factor(status, levels = c("Control", "HCV"))
  )

table(df$status, useNA = "ifany")
write_csv(as.data.frame(table(df$status)), "output/class_counts.csv")

# 3) Gestión de valores perdidos ------------------------------

num_vars <- names(df)[sapply(df, is.numeric)]
cat_vars <- names(df)[!sapply(df, is.numeric)]

na_before <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing")

write_csv(na_before, "output/missing_counts_before_imputation.csv")

# Imputación numéricas (media por grupo Control/HCV)
for (v in num_vars) {
  df <- df %>%
    group_by(status) %>%
    mutate(
      !!v := ifelse(
        is.na(.data[[v]]),
        mean(.data[[v]], na.rm = TRUE),
        .data[[v]]
      )
    ) %>%
    ungroup()
}

# Imputación categórica (modo en Sex)
if ("sex" %in% names(df)) {
  mode_sex <- names(sort(table(df$sex), decreasing = TRUE))[1]
  df$sex[is.na(df$sex)] <- mode_sex
  df$sex <- factor(df$sex)
}

na_after <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing")

write_csv(na_after, "output/missing_counts_after_imputation.csv")

# Recalcular numéricas por si acaso
num_vars <- names(df)[sapply(df, is.numeric)]

# 4) Descriptivos por grupo -----------------------------------

desc_by_group <- df %>%
  group_by(status) %>%
  summarise(
    across(
      all_of(num_vars),
      list(
        n      = ~ sum(!is.na(.)),
        mean   = ~ mean(.),
        sd     = ~ sd(.),
        median = ~ median(.),
        iqr    = ~ IQR(.)
      ),
      .names = "{.col}_{.fn}"
    )
  )

write_csv(desc_by_group, "output/desc_by_group.csv")

# 5) Gráficos descriptivos (facetados) ------------------------

df_long <- df %>%
  pivot_longer(
    cols = all_of(num_vars),
    names_to = "variable",
    values_to = "value"
  )

p_hist_all <- ggplot(df_long, aes(x = value, fill = status)) +
  geom_histogram(alpha = 0.5, position = "identity", bins = 30) +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_minimal() +
  labs(
    title = "Distribución de los parámetros bioquímicos por grupo",
    x = "",
    y = "Frecuencia",
    fill = "Grupo"
  )

ggsave("output/hist_all.png", p_hist_all, width = 10, height = 8)

p_box_all <- ggplot(df_long, aes(x = status, y = value)) +
  geom_boxplot() +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  theme_minimal() +
  labs(
    title = "Distribución de los parámetros bioquímicos por grupo",
    x = "",
    y = ""
  )

ggsave("output/box_all.png", p_box_all, width = 10, height = 8)

# 6) Matriz de correlaciones ----------------------------------

cor_mat <- cor(
  df %>% select(all_of(num_vars)),
  use = "pairwise.complete.obs"
)

write_csv(as.data.frame(cor_mat), "output/correlation_matrix.csv")

png("output/correlation_heatmap.png", width = 800, height = 800)
heatmap(cor_mat, symm = TRUE, main = "Matriz de correlaciones (Pearson)")
dev.off()

# 7) Contrastes de hipótesis ----------------------------------

contrast_list <- list()

for (v in num_vars) {
  x <- df %>% filter(status == "Control") %>% pull(!!sym(v))
  y <- df %>% filter(status == "HCV")     %>% pull(!!sym(v))
  
  t_res <- tryCatch(t.test(x, y), error = function(e) NULL)
  w_res <- wilcox.test(x, y, exact = FALSE)
  
  contrast_list[[v]] <- tibble(
    variable    = v,
    t_stat      = if (!is.null(t_res)) t_res$statistic else NA_real_,
    t_p         = if (!is.null(t_res)) t_res$p.value   else NA_real_,
    wilcox_stat = w_res$statistic,
    wilcox_p    = w_res$p.value
  )
}

contrast_numeric <- bind_rows(contrast_list)
write_csv(contrast_numeric, "output/contrast_numeric.csv")

if ("sex" %in% names(df)) {
  tab_sex <- table(df$sex, df$status)
  chi_sex <- chisq.test(tab_sex)
  
  contrast_cat <- tibble(
    variable = "sex",
    chi2     = chi_sex$statistic,
    df       = chi_sex$parameter,
    p_value  = chi_sex$p.value
  )
  
  write_csv(contrast_cat, "output/contrast_categorical.csv")
}

# 8) Preparación para modelos --------------------------------

features <- df %>%
  select(-category, -status)

if ("sex" %in% names(features)) {
  features$sex <- factor(features$sex)
}

set.seed(123)
train_index <- caret::createDataPartition(df$status, p = 0.8, list = FALSE)
train_data  <- df[train_index, ]
test_data   <- df[-train_index, ]

train_x <- train_data %>% select(-category, -status)
train_y <- train_data$status

test_x <- test_data %>% select(-category, -status)
test_y <- test_data$status

# 9) Control de entrenamiento (caret) -------------------------

ctrl <- trainControl(
  method        = "repeatedcv",
  number        = 5,
  repeats       = 3,
  classProbs    = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# 9a) Regresión logística (caret, para comparación global) ----

set.seed(123)
glm_fit <- train(
  x          = train_x,
  y          = train_y,
  method     = "glm",
  family     = "binomial",
  metric     = "ROC",
  trControl  = ctrl,
  preProcess = c("center", "scale")
)

saveRDS(glm_fit, "output/glm_fit.rds")

# 9b) Random Forest (caret) -----------------------------------

set.seed(123)

grid_rf <- expand.grid(
  mtry = c(2, 3, 4, 5)
)

rf_fit <- train(
  x          = train_x,
  y          = train_y,
  method     = "rf",
  metric     = "ROC",
  trControl  = ctrl,
  tuneGrid   = grid_rf,
  ntree      = 500,
  importance = TRUE
)

saveRDS(rf_fit, "output/rf_fit.rds")

rf_varimp <- varImp(rf_fit, scale = TRUE)

png("output/rf_varimp.png", width = 800, height = 600)
plot(rf_varimp, main = "Importancia de variables - Random Forest")
dev.off()

rf_varimp_df <- rf_varimp$importance %>%
  tibble::rownames_to_column("variable")

write_csv(rf_varimp_df, "output/rf_varimp.csv")

# 9c) Regresión logística detallada (stepwise) ----------------

predictor_vars <- setdiff(
  names(train_data),
  c("status", "category")
)

form_logit <- as.formula(
  paste("status ~", paste(predictor_vars, collapse = " + "))
)

logit_full <- glm(
  formula = form_logit,
  data    = train_data,
  family  = binomial(link = "logit")
)

logit_full_tidy <- broom::tidy(logit_full) %>%
  mutate(
    OR      = exp(estimate),
    OR_low  = exp(estimate - 1.96 * std.error),
    OR_high = exp(estimate + 1.96 * std.error)
  )

write_csv(logit_full_tidy, "output/logit_full_tidy.csv")

logit_step <- stepAIC(
  logit_full,
  direction = "both",
  trace     = FALSE
)

logit_step_tidy <- broom::tidy(logit_step) %>%
  mutate(
    OR      = exp(estimate),
    OR_low  = exp(estimate - 1.96 * std.error),
    OR_high = exp(estimate + 1.96 * std.error)
  )

write_csv(logit_step_tidy, "output/logit_step_tidy.csv")

test_data$prob_logit_step <- predict(
  logit_step,
  newdata = test_data,
  type    = "response"
)

test_data$pred_logit_step <- ifelse(
  test_data$prob_logit_step >= 0.5, "HCV", "Control"
)

test_data$pred_logit_step <- factor(
  test_data$pred_logit_step,
  levels = c("Control", "HCV")
)

cm_logit_step <- caret::confusionMatrix(
  data      = test_data$pred_logit_step,
  reference = test_data$status,
  positive  = "HCV"
)

logit_step_metrics <- data.frame(
  Accuracy    = cm_logit_step$overall["Accuracy"],
  Kappa       = cm_logit_step$overall["Kappa"],
  Sensitivity = cm_logit_step$byClass["Sensitivity"],
  Specificity = cm_logit_step$byClass["Specificity"],
  PPV         = cm_logit_step$byClass["Pos Pred Value"],
  NPV         = cm_logit_step$byClass["Neg Pred Value"]
)

write_csv(logit_step_metrics, "output/logit_step_metrics_test.csv")

roc_logit_step <- roc(
  response  = test_data$status,
  predictor = test_data$prob_logit_step,
  levels    = c("Control", "HCV"),
  direction = "<"
)

auc_logit_step <- auc(roc_logit_step)

write_csv(
  data.frame(Model = "Logistic_stepwise", AUC = as.numeric(auc_logit_step)),
  "output/logit_step_auc_test.csv"
)

roc_step_df <- data.frame(
  fpr = 1 - roc_logit_step$specificities,
  tpr = roc_logit_step$sensitivities
)

p_roc_logit_step <- ggplot(roc_step_df, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  theme_minimal() +
  labs(
    title = "Curva ROC - Regresión logística (modelo stepwise)",
    x = "1 - Especificidad",
    y = "Sensibilidad"
  )

ggsave(
  filename = "output/roc_logit_step.png",
  plot     = p_roc_logit_step,
  width    = 5,
  height   = 5
)

# 10) Evaluación del Random Forest en test --------------------

rf_probs <- predict(rf_fit, newdata = test_x, type = "prob")[, "HCV"]
rf_preds <- predict(rf_fit, newdata = test_x)

roc_rf <- roc(
  response  = test_y,
  predictor = rf_probs,
  levels    = c("Control", "HCV"),
  direction = "<"
)

auc_rf <- auc(roc_rf)

cm_rf <- table(Predicted = rf_preds, Observed = test_y)

tn <- cm_rf["Control", "Control"]
fp <- cm_rf["HCV",    "Control"]
fn <- cm_rf["Control", "HCV"]
tp <- cm_rf["HCV",    "HCV"]

rf_metrics <- tibble(
  Model       = "Random Forest",
  AUC         = as.numeric(auc_rf),
  Accuracy    = (tp + tn) / sum(cm_rf),
  Sensitivity = tp / (tp + fn),
  Specificity = tn / (tn + fp),
  TN = tn,
  FP = fp,
  FN = fn,
  TP = tp
)

write_csv(rf_metrics, "output/rf_metrics_test.csv")

png("output/roc_random_forest.png", width = 700, height = 600)
plot(roc_rf, main = "Curva ROC - Random Forest")
abline(a = 0, b = 1, lty = 2)
dev.off()

# 11) Reproducibilidad ----------------------------------------

sink("output/sessionInfo.txt")
print(sessionInfo())
sink()
