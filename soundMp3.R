source("renv.R")

library(parallel)
library(tidyverse)
library(abind)
library(caret)
library(tuneR)

genre <- "HipHop"
path <- "test/HipHop-006.mp3"

melspec <- function(x, start, end){
  mp3 <- readMP3(filename = x) %>% 
    extractWave(xunit = "time",
                from = start, to = end)
  
  # return log-spectrogram with 256 Mel bands and compression
  sp <- melfcc(mp3, nbands = 256, usecmp = T,
               spec_out = T,
               hoptime = (end-start) / 256)$aspectrum
  
  # Median-based noise reduction
  noise <- apply(sp, 1, median)
  sp <- sweep(sp, 1, noise)
  sp[sp < 0] <- 0
  
  # Normalize to max
  sp <- sp / max(sp)
  
  return(sp)
}

Xsong <- lapply(X = path, FUN = melspec,
                start = 0, end = 15) %>%
  simplify2array()

Xsong <- array(Xsong, dim=c(1, 256, 256, 1))

predXSong <- predict(model, Xsong)
predXClass <- genreClass[apply(predXSong, 1, which.max)]
base::print(predXClass)
