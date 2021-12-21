# ABC analysis based feature selection based on variable importance
# for one classifier
#' @importFrom ABCanalysis ABCanalysis
SelectFeaturesOneClassifier <- function(DataToProcessPrepared = DataToProcessPrepared, Classifier,
                                        nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees, Seeds) {

  # Internal functions
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }

  #
  ABCselectFeatures <- function(VariableImportance) {

    is.integer0 <- function(x) {
      is.integer(x) && length(x) == 0L
    }

    ABC_GT0 <- data.frame(X1 = VariableImportance)
    ABC_GT0 <- subset(ABC_GT0, ABC_GT0$X1 > 0)
    if (nrow(ABC_GT0) > 0) {
      ABC_Features <- ABCanalysis::ABCanalysis(as.vector(ABC_GT0$X1), PlotIt = F)
      ABC_A <- c(ABC_Features$Aind)
      if (is.integer0(ABC_A) | length(ABC_A) == 0) {
        ABC_A <- c(ABC_Features$Aind, ABC_Features$Bind)
      }
      if (is.integer0(ABC_A) | length(ABC_A) == 0) {
        ABC_A <- max(ABC_GT0)  #1:nrow(ABC_GT0)
      }
      if (is.integer0(ABC_A) | length(ABC_A) == 0) {
        ABC_A_names <- names(VariableImportance)
      } else {
        ABC_A_names <- rownames(ABC_GT0)[ABC_A]
      }
    } else {
      ABC_A_names <- names(VariableImportance)
    }
    return(ABC_A_names)
  }

  ### Feature selection
  ABCResamplinAallIter <- vector()
  ABCResamplinAcountallIter <- vector()
  nColTrainingData <- ncol(DataToProcessPrepared[[i]]$TrainDataAndClsScaled)
  FeatureNamesTrainingData <- names(DataToProcessPrepared[[i]]$TrainDataAndClsScaled)[2:nColTrainingData]

  FeatureSelectionPerIteration <-
    parallel::mclapply(list.of.seeds, function(i) {
      ABCResamplinA <- vector()
      ABCResamplinAcount <- vector()

      # Leave one feature out assessment of classifier performance
      LOOaccuracies <- unlist(
        parallel::mclapply(1:nColTrainingData, function(i1) {
          TrainDataAndClsScaled1 <- TrainDataAndClsScaled
          TestValidationDataAndClsScaled1 <- TestValidationDataAndClsScaled
          if(i1 > 1) {
            TrainDataAndClsScaled1 <- TrainDataAndClsScaled[,-i1]
            TestValidationDataAndClsScaled1 <- TestValidationDataAndClsScaled[,-i1]
          }
          AccuracyActual <- mean(na.omit(
            TrainClassifierAndEstimatePerformance(
              TrainDataAndClsScaled = TrainDataAndClsScaled1,
              TestValidationDataAndClsScaled = TestValidationDataAndClsScaled1,
              Classifier = Classifier, Type = "Classes", Measure = "CM",
              nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
              Seed = i)$CM["Balanced Accuracy", ]))

        }, mc.cores = nProc )
      )
      names(LOOaccuracies) <- c("All", FeatureNamesTrainingData)
      LOOaccuracies <- (LOOaccuracies[1] - LOOaccuracies)[2:length(LOOaccuracies)]

      # ABC feature selection per iteration
      ABC_A_names <- ABCselectFeatures(VariableImportance = LOOaccuracies)
      ABCResamplinA <- append(ABCResamplinA, ABC_A_names)
      ABCResamplinAcount <- append(ABCResamplinAcount, length(ABC_A_names))

      return(list(ABCResamplinA = ABCResamplinA, ABCResamplinAcount = ABCResamplinAcount))
    }, mc.cores = nProc)

  if (length(which(grepl("Error", lapply(FeatureSelectionPerIteration, "[[",
                                         1)) == T)) > 0) {
    FeatureSelectionPerIteration <-
      FeatureSelectionPerIteration[-which(grepl("Error", lapply(FeatureSelectionPerIteration, "[[", 1)) == T)]
  }

  # ABC feature selection across all iterations
  ABCResamplinAallIter <- append(ABCResamplinAallIter, unlist(lapply(FeatureSelectionPerIteration, "[[", "ABCResamplinA")))
  ABCResamplinAcountallIter <- append(ABCResamplinAcountallIter, unlist(lapply(FeatureSelectionPerIteration, "[[", "ABCResamplinAcount")))
  ABCResamplinAcountallIterMode <- getmode(ABCResamplinAcountallIter)

  dfABCResamplinAallIter <- cbind.data.frame(Features = FeatureNamesTrainingData, Freq = 0)
  dfABCResamplinAallIterOnlySelected <- data.frame(table(ABCResamplinAallIter))
  names(dfABCResamplinAallIterOnlySelected) <- c("Features", "Freq")
  dfABCResamplinAallIter$Freq[match(dfABCResamplinAallIterOnlySelected$Features, dfABCResamplinAallIter$Features)] <-
    dfABCResamplinAallIterOnlySelected$Freq
  ABCselectedFeatures <- as.vector(dfABCResamplinAallIter$Features[order(-dfABCResamplinAallIter$Freq)])[1:ABCResamplinAcountallIterMode]

  return(list(FeaturesTimesinABCA = dfABCResamplinAallIter, FeaturesSelected = ABCselectedFeatures))
}

