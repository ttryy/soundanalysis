source("renv.R")

library(parallel)
library(tidyverse)
library(abind)
library(caret)
library(tuneR)
source("funs.R")

#### Pre-processing ####
# Read files
fnames <- list.files("mp3/", full.names = T, patt = "*.mp3")

# Write metadata for Kaggle dataset
ids <- str_extract(fnames, pattern = "[0-9]{3,}")
classes <- str_replace_all(str_extract(fnames, pattern = "/[A-Za-z]*-"), "[/-]", "")

query <- data.frame(Genre=classes, Id=ids, Path=fnames)




# Encode species from fnames regex
genre <- str_extract(fnames, pattern = "/[A-Za-z]*-") %>%
  gsub(patt = "[/-]", rep = " ") %>% factor()

# Stratified sampling: train (80%), val (10%) and test (10%)
idx <- createFolds(genre, k = 10)
valIdx <- idx$Fold01
testIdx <- idx$Fold02
# Define samples for train, val and test
fnamesTrain <- fnames[-c(valIdx, testIdx)]
fnamesVal <- fnames[valIdx]
fnamesTest <- fnames[testIdx]

# Take multiple readings per sample for training
Xtrain <- audioProcess(files = fnamesTrain, ncores = 1,
                       limit = 60, ws = 30, stride = 15)
Xval <- audioProcess(files = fnamesVal, ncores = 2,
                     limit = 60, ws = 30, stride = 15)
Xtest <- audioProcess(files = fnamesTest, ncores = 1,
                      limit = 60, ws = 30, stride = 15)






# Define targets and augment data
target <- model.matrix(~0+genre)

targetTrain <- do.call("rbind", lapply(1:(dim(Xtrain)[1]/length(fnamesTrain)),
                                       function(x) target[-c(valIdx, testIdx),]))
targetVal <- do.call("rbind", lapply(1:(dim(Xval)[1]/length(fnamesVal)),
                                     function(x) target[valIdx,]))
targetTest <- do.call("rbind", lapply(1:(dim(Xtest)[1]/length(fnamesTest)),
                                      function(x) target[testIdx,]))
# Assemble Xs and Ys
train <- list(X = Xtrain, Y = targetTrain)
val <- list(X = Xval, Y = targetVal)
test <- list(X = Xtest, Y = targetTest)






# Plot spectrogram from random training sample - range is 0-22.05 kHz
image(train$X[sample(dim(train$X)[1], 1),,,],
      xlab = "Time (s)",
      ylab = "Frequency (kHz)",
      axes = F)
# Generate mel sequence from Hz points, standardize to plot
freqs <- c(0, 1, 5, 15, 22.05)
mels <- 2595 * log10(1 + (freqs*1e3) / 700) # https://en.wikipedia.org/wiki/Mel_scale
mels <- mels - min(mels)
mels <- mels / max(mels)

axis(1, at = seq(0, 1, by = .2), labels = seq(0, 10, by = 2))
axis(2, at = mels, las = 2,
     labels = round(freqs, 2))
axis(3, labels = F); axis(4, labels = F)






#### Save ####
save(train, val, test, file = "prepAudio.RData")
