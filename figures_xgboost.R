# install.packages(c("xgboost", "ranger", "viridis", "cowplot"))
library(xgboost)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(viridis)
library(cowplot)
library(ranger)

set.seed(321)

########################################################################################
# Define theme
########################################################################################

custom_theme <- function(base_size = 16) { 
  theme_minimal_hgrid(font_size = base_size) +
    theme(
      legend.position="bottom",
      legend.justification = "center",
      plot.margin=unit(c(0.5,1,0.5,0.5), "cm"),
      plot.title = element_text(hjust = 0.5),
      panel.background = element_rect(fill = 'white', colour = 'gray40'),
      plot.background = element_rect(fill = 'white', colour = 'white'),
      panel.spacing = unit(3, "mm")
      # panel.grid.major.x = element_line(
      #   color = "gray70",
      #   size = 0.5),
      # panel.grid.major.y = element_line(
      #   color = "gray70",
      #   size = 0.5),
      # panel.grid.minor.x = element_line(
      #   color = "gray80",
      #   size = 0.5),
      # panel.grid.minor.y = element_line(
      #   color = "gray80",
      #   size = 0.5)
    )
}

theme_set(custom_theme())
update_geom_defaults("line", list(size = 0.75))

options(ggplot2.continuous.colour="viridis")
options(ggplot2.discrete.fill="viridis")
options(ggplot2.discrete.color="viridis")
fill_viridis_d_color <- "E"

# Taille des pdf
width_png <- 200
height_png <- 120
options(OutDec= ".")



########################################################################################
# Generate data
########################################################################################

data <- data.frame(
  x = seq(0, 10, 10 / 500)
) |>
  mutate(id = row_number()) |>
  mutate(
    target_clean = sin(x) + 0.02 * x^2 + 0.5 * cos((x - 1.5)^2/2.5) + 0.2,
    train = as.integer(runif(n()) > 0.2) 
  ) |>
  mutate(
    target_noisy_temp = target_clean + runif(n(), -3, 3) * (runif(n()) > 0.95)
  ) |>
  mutate(
    target_noisy = case_when(
      target_noisy_temp |> between(-1.5, 3.5) ~ target_noisy_temp,
      target_noisy_temp > 3.5   ~ 3.5,
      target_noisy_temp <= -1.5 ~ 1.5
    )
  )

# Clean target
ggplot(data) + 
  geom_line(aes(x = x, y = target_clean), size = 0.5) + 
  labs(x = "x", y = "Target", color = NULL) +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "bottom")

# Noisy target
ggplot(data) + 
  geom_line(aes(x = x, y = target_noisy), size = 0.5) + 
  labs(x = "x", y = "Target", color = NULL) +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "bottom")


########################################################################################
# Train the models
########################################################################################

max.depth_rf <- 5
max.depth_gb <- 3

train <- data |> filter(train == 1)
test  <- data |> filter(train == 0)

model_xgb_clean <- xgboost(
  data = train  |> select(x) |> as.matrix(), 
  label = train |> pull(target_clean),
  max.depth = max.depth_gb, eta = 1, 
  nthread = 2, nrounds = 50, objective = "reg:squarederror")

model_xgb_noisy <- xgboost(
  data = train  |> select(x) |> as.matrix(), 
  label = train |> pull(target_noisy),
  max.depth = max.depth_gb, eta = 1, 
  nthread = 2, nrounds = 50, objective = "reg:squarederror")

model_rf_clean <- ranger(
  target_clean ~ ., data = train  |> select(target_clean, x), num.trees = 50, max.depth = max.depth_rf, replace = TRUE, sample.fraction = 0.2
)

model_rf_noisy <- ranger(
  target_noisy ~ ., data = train  |> select(target_noisy, x), num.trees = 50, max.depth = max.depth_rf, replace = TRUE, sample.fraction = 0.2
)

########################################################################################
# Prepare data
########################################################################################

test2 <- test |>
  mutate(
    prediction_xgb_clean_1_tree     = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 2)),
    prediction_xgb_clean_2_trees    = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 3)),
    prediction_xgb_clean_3_trees    = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 4)),
    prediction_xgb_clean_4_trees    = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 5)),
    prediction_xgb_clean_5_trees    = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 6)),
    prediction_xgb_clean_10_trees   = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 11)),
    prediction_xgb_clean_20_trees   = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 21)),
    prediction_xgb_clean_50_trees   = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 1)),
    prediction_xgb_noisy_1_tree     = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 2)),
    prediction_xgb_noisy_2_trees    = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 3)),
    prediction_xgb_noisy_3_trees    = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 4)),
    prediction_xgb_noisy_4_trees    = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 5)),
    prediction_xgb_noisy_5_trees    = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 6)),
    prediction_xgb_noisy_10_trees   = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 11)),
    prediction_xgb_noisy_20_trees   = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 21)),
    prediction_xgb_noisy_50_trees   = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 1)),
    
    prediction_xgb_clean_single_tree1      = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(1, 2)),
    prediction_xgb_clean_single_tree2      = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(2, 3)),
    prediction_xgb_clean_single_tree3      = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(3, 4)),
    prediction_xgb_clean_single_tree4      = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(4, 5)),
    prediction_xgb_clean_single_tree50     = predict(model_xgb_clean, test |> pull(x) |> as.matrix(), iterationrange = c(50, 51)),
    prediction_xgb_noisy_single_tree1      = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(1, 2)),
    prediction_xgb_noisy_single_tree2      = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(2, 3)),
    prediction_xgb_noisy_single_tree3      = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(3, 4)),
    prediction_xgb_noisy_single_tree4      = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(4, 5)),
    prediction_xgb_noisy_single_tree50     = predict(model_xgb_noisy, test |> pull(x) |> as.matrix(), iterationrange = c(50, 51)),
    
    prediction_rf_clean_1_tree      = predict(model_rf_clean, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_clean_2_trees     = predict(model_rf_clean, data = test |> select(x), num.trees =  2)$predictions,
    prediction_rf_clean_3_trees     = predict(model_rf_clean, data = test |> select(x), num.trees =  3)$predictions,
    prediction_rf_clean_4_trees     = predict(model_rf_clean, data = test |> select(x), num.trees =  4)$predictions,
    prediction_rf_clean_5_trees     = predict(model_rf_clean, data = test |> select(x), num.trees =  5)$predictions,
    prediction_rf_clean_10_trees    = predict(model_rf_clean, data = test |> select(x), num.trees = 10)$predictions,
    prediction_rf_clean_20_trees    = predict(model_rf_clean, data = test |> select(x), num.trees = 20)$predictions,
    prediction_rf_clean_50_trees    = predict(model_rf_clean, data = test |> select(x), num.trees = 50)$predictions,
    prediction_rf_noisy_1_tree      = predict(model_rf_noisy, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_noisy_2_trees     = predict(model_rf_noisy, data = test |> select(x), num.trees =  2)$predictions,
    prediction_rf_noisy_3_trees     = predict(model_rf_noisy, data = test |> select(x), num.trees =  3)$predictions,
    prediction_rf_noisy_4_trees     = predict(model_rf_noisy, data = test |> select(x), num.trees =  4)$predictions,
    prediction_rf_noisy_5_trees     = predict(model_rf_noisy, data = test |> select(x), num.trees =  5)$predictions,
    prediction_rf_noisy_10_trees    = predict(model_rf_noisy, data = test |> select(x), num.trees = 10)$predictions,
    prediction_rf_noisy_20_trees    = predict(model_rf_noisy, data = test |> select(x), num.trees = 20)$predictions,
    prediction_rf_noisy_50_trees    = predict(model_rf_noisy, data = test |> select(x), num.trees = 50)$predictions,
    
    prediction_rf_clean_single_tree1     = predict(model_rf_clean, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_clean_single_tree2     = 2 *  predict(model_rf_clean, data = test |> select(x), num.trees =   2)$predictions  - 1 * predict(model_rf_clean, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_clean_single_tree3     = 3 *  predict(model_rf_clean, data = test |> select(x), num.trees =   3)$predictions  - 2 * predict(model_rf_clean, data = test |> select(x), num.trees =  2)$predictions,
    prediction_rf_clean_single_tree4     = 4 *  predict(model_rf_clean, data = test |> select(x), num.trees =   4)$predictions  - 3 * predict(model_rf_clean, data = test |> select(x), num.trees =  3)$predictions,
    prediction_rf_clean_single_tree50    = 50 * predict(model_rf_clean, data = test |> select(x), num.trees =  50)$predictions - 49 * predict(model_rf_clean, data = test |> select(x), num.trees =  49)$predictions,
    prediction_rf_noisy_single_tree1     = predict(model_rf_noisy, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_noisy_single_tree2     = 2 *  predict(model_rf_noisy, data = test |> select(x), num.trees =   2)$predictions  - 1 * predict(model_rf_noisy, data = test |> select(x), num.trees =  1)$predictions,
    prediction_rf_noisy_single_tree3     = 3 *  predict(model_rf_noisy, data = test |> select(x), num.trees =   3)$predictions  - 2 * predict(model_rf_noisy, data = test |> select(x), num.trees =  2)$predictions,
    prediction_rf_noisy_single_tree4     = 4 *  predict(model_rf_noisy, data = test |> select(x), num.trees =   4)$predictions  - 3 * predict(model_rf_noisy, data = test |> select(x), num.trees =  3)$predictions,
    prediction_rf_noisy_single_tree50    = 50 * predict(model_rf_noisy, data = test |> select(x), num.trees =  50)$predictions - 49 * predict(model_rf_noisy, data = test |> select(x), num.trees =  49)$predictions
  ) |>
  pivot_longer(
    cols = starts_with("pred"),
    values_to = "prediction",
    names_to = "prediction_name"
  ) |>
  mutate(
    model_type   = prediction_name |> str_extract("(xgb|rf)"),
    number_trees = prediction_name |> str_extract("\\d+") |> as.integer(),
    target_type  = prediction_name |> str_extract("(noisy|clean)"),
    single_tree  = prediction_name |> str_detect("single_tree") |> as.integer()
  ) |>
  mutate(
    label_target = if_else(target_type == "clean", "Clean data", "Noisy data") |> 
      factor(levels = c("Clean data", "Noisy data"), ordered = TRUE),
    label_model = if_else(model_type == "xgb", "Gradient boosting", "Random forest") |> 
      factor(levels = c("Gradient boosting", "Random forest"), ordered = TRUE),
    label_single_tree = if_else(single_tree == 1, "Single tree", "Full model") |> 
      factor(levels = c("Single tree", "Full model"), ordered = TRUE)
  )


########################################################################################
########################################################################################
########################################################################################
# Plots
########################################################################################
########################################################################################
########################################################################################

########################################################################################
# Plot the target
########################################################################################

data |>
  pivot_longer(
    cols = c("target_clean", "target_noisy"),
    names_to = "target_type",
    values_to = "target_value"
  ) |>
  mutate(
    label_target = if_else(target_type == "target_clean", "Clean", "Noisy") |> factor(levels = c("Clean", "Noisy"), ordered = TRUE)
  ) |>
  ggplot() + 
  geom_point(aes(x = x, y = target_value), size = 0.5) + 
  labs(x = "x", y = "Target", color = NULL) +
  scale_color_viridis_d(end = 0.7) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("label_target")


########################################################################################
########################################################################################
# Plot the predictions of the single trees
########################################################################################
########################################################################################

selection0 <- c(1, 2, 3, 50)

########################################################################################
# Clean target
########################################################################################

# Random forest model trained on the clean target
test2 |> 
  filter(label_model == "Random forest" & label_target == "Clean data" & number_trees %in% selection0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = NULL, y = NULL, color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  facet_grid(label_single_tree ~ number_trees, switch = "y") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

ggsave(
  "./figures/single_trees_clean_rf.png", 
  width = width_png, height = height_png, units = "mm"
)


# Gradient boosting model trained on the clean target
test2 |> 
  filter(label_model == "Gradient boosting" & label_target == "Clean data" & number_trees %in% selection0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = NULL, y = NULL, color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  facet_grid(label_single_tree ~ number_trees, switch = "y") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

ggsave(
  "./figures/single_trees_clean_gb.png", 
  width = width_png, height = height_png, units = "mm"
)


########################################################################################
# Noisy target
########################################################################################

# Random forest model trained on the noisy target
test2 |> 
  filter(label_model == "Random forest" & label_target == "Noisy data" & number_trees %in% selection0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = NULL, y = NULL, color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  facet_grid(label_single_tree ~ number_trees, switch = "y") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

ggsave(
  "./figures/single_trees_noisy_rf.png", 
  width = width_png, height = height_png, units = "mm"
)


# Gradient boosting model trained on the noisy target
test2 |> 
  filter(label_model == "Gradient boosting" & label_target == "Noisy data" & number_trees %in% selection0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = NULL, y = NULL, color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) +
  scale_y_continuous(expand = c(0, 0), limits = c(-1.6, 3.6), labels = NULL) +
  facet_grid(label_single_tree ~ number_trees, switch = "y") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

ggsave(
  "./figures/single_trees_noisy_gb.png", 
  width = width_png, height = height_png, units = "mm"
)



########################################################################################
########################################################################################
# Plot the predictions of the full models
########################################################################################
########################################################################################

selection <- c(1, 2, 5, 10)

########################################################################################
# Random forest
########################################################################################

# Random forest model trained on the clean target
test2 |> 
  filter(label_model == "Random forest" & label_target == "Clean data" & number_trees %in% selection & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("number_trees") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

# Random forest model trained on the noisy target
test2 |> 
  filter(label_model == "Random forest" & label_target == "Noisy data" & number_trees %in% selection & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("number_trees") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )


########################################################################################
# Gradient boosting
########################################################################################

# Gradient boosting model trained on the clean target
test2 |> 
  filter(label_model == "Gradient boosting" & label_target == "Clean data" & number_trees %in% selection & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("number_trees") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )


# Gradient boosting model trained on the noisy target
test2 |> 
  filter(label_model == "Gradient boosting" & label_target == "Noisy data" & number_trees %in% selection & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("number_trees") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )


########################################################################################
# Comparison Gradient boosting versus random forest
########################################################################################


# Gradient boosting model and RF model trained on the clean target
test2 |> 
  filter(label_target == "Clean data" & number_trees == 10 & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = label_model), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  scale_color_viridis_d(end = 0.7) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Gradient boosting" = viridis::viridis(10)[1], "Random forest" = viridis::viridis(10)[7])
  )



# Gradient boosting model and RF model trained on the noisy target
test2 |> 
  filter(label_target == "Noisy data" & number_trees == 10 & single_tree == 0) |>
  ggplot() + 
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  geom_line(aes(x = x, y = prediction, color = label_model), size = 0.5) +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  scale_color_viridis_d(end = 0.7) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Gradient boosting" = viridis::viridis(10)[1], "Random forest" = viridis::viridis(10)[7])
  )


# Gradient boosting model trained on the noisy target
test2 |> 
  filter(label_model == "Gradient boosting" & label_target == "Noisy data" & number_trees %in% selection) |>
  ggplot() + 
  geom_line(aes(x = x, y = prediction, color = "Prediction"), size = 0.5) +
  geom_line(aes(x = x, y = target_clean, color = "Target"), size = 0.5, linetype = "dashed") +
  labs(x = "x", y = "Prediction", color = "Type of target") +
  scale_x_continuous(expand = c(0, 0), labels = NULL) + 
  scale_y_continuous(expand = c(0, 0), labels = NULL) + 
  theme_minimal() +
  theme(legend.position = "bottom") +
  facet_wrap("number_trees") +
  scale_color_manual(
    name = NULL, 
    values = c("Target" = "red", "Prediction" = "black")
  )

