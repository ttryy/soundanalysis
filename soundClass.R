source("renv.R")

library(keras)
library(tidyverse)
library(caret)
library(e1071)
library(pheatmap)
library(RColorBrewer)

# Read processed data
base::load("prepAudio.RData")


# Build model
model <- keras_model_sequential() %>% 
  layer_conv_2d(input_shape = dim(train$X)[2:4], 
                filters = 16, kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = .2) %>% 
  
  layer_conv_2d(filters = 32, kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = .2) %>% 
  
  layer_conv_2d(filters = 64, kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = .2) %>% 
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3),
                activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(28, 2)) %>%
  layer_dropout(rate = .2) %>%
  
  layer_flatten() %>% 
  
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(rate = .5) %>% 
  layer_dense(units = ncol(train$Y), activation = "softmax")






# Print summary
summary(model)
model %>% compile(optimizer = optimizer_adam(decay = 1e-5),
                  loss = "categorical_crossentropy",
                  metrics = "accuracy")

filepath <- file.path("checkpoints/", "weights.{epoch:02d}.hdf5")

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = filepath,
  save_weights_only = TRUE,
  save_freq = 5,
  period = 5,
  verbose = 1
)

history <- fit(model, x = train$X, y = train$Y,
               batch_size = 16, epochs = 80,
               validation_data = list(val$X, val$Y), 
               callbacks = list(cp_callback))

plot(history)

# Save model
save(history, file="history.RData")
model %>% save_model_hdf5("model.h5")

base::load("history.RData")
model <- load_model_hdf5("model.h5")



# Grep species, set colors for heatmap
genreClass <- gsub(colnames(train$Y), pat = "genre", rep = "")
cols <- colorRampPalette(rev(brewer.pal(n = 7, name = "RdGy")))

# Validation predictions
predProb <- predict(model, val$X)
predClass <- genreClass[apply(predProb, 1, which.max)]
trueClass <- genreClass[apply(val$Y, 1, which.max)]

# Plot confusion matrix
confMat <- confusionMatrix(data = factor(predClass, levels = genreClass),
                           reference = factor(trueClass, levels = genreClass))

pheatmap(confMat$table, cluster_rows = F, cluster_cols = F,
         border_color = NA, show_colnames = F,
         labels_row = genreClass,
         color = cols(max(confMat$table)+1))

# Accuracy in validation set
mean(predClass == trueClass)







# Test set prediction
predXProb <- predict(model, test$X)
predXClass <- genreClass[apply(predXProb, 1, which.max)]
trueXClass <- genreClass[apply(test$Y, 1, which.max)]

# Plot confusion matrix
confMatTest <- confusionMatrix(data = factor(predXClass, levels = genreClass),
                               reference = factor(trueXClass, levels = genreClass))

pheatmap(confMatTest$table, cluster_rows = F, cluster_cols = F,
         border_color = NA, show_colnames = F,
         labels_row = genreClass,
         color = cols(max(confMatTest$table)+1))

# Accuracy in test set
mean(predXClass == trueXClass)



