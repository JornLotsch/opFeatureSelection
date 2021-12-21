# Extract features and balanced accuracies per algorithm
CreateCombinedFeaturesSelected <- function(PreparedData, FeaturesPreSelected, FeatureSelectionAlgorithms,
                                           Type, Measure, nClasses, kNNk, RFtrees, Seeds) {

  FeaturesPreSelected <- lapply(FeaturesPreSelected, "[[", "FeaturesSelected")
  names(FeaturesPreSelected) <- FeatureSelectionAlgorithms
  FeaturesSelectedUnion <- Reduce(f = union, x = FeaturesPreSelected)
  FeaturesSelectedIntersection <- Reduce(f = intersect, x = FeaturesPreSelected)
  FeaturesSelectedNoIntersection <- setdiff(FeaturesSelectedUnion, FeaturesSelectedIntersection)

  FeaturesSelectedCombinations <-
    parallel::mclapply(seq(FeaturesSelectedNoIntersection) + 1, function(x) {
      if (!identical(FeaturesSelectedIntersection, character(0))) {
        Comb1 <- combn(c("XFeaturesSelectedIntersection", FeaturesSelectedNoIntersection), x)
        if (prod(size(Comb1)) != x) {
          Xcontained <- grepl("XFeaturesSelectedIntersection", Comb1[1, ])
          Comb1 <- Comb1[, Xcontained == TRUE]
        }
      } else {
        Comb1 <- combn(FeaturesSelectedNoIntersection, x - 1)
      }
      list.of.feature.combinations <- lapply(seq_len(ncol(Comb1)), function(x) Comb1[, x])

      # Performance testing with all combinations of selected features
      PerformanceTestingCombinations <-
        parallel::mclapply(list.of.feature.combinations,
                           function(ActualFeatures) {
                             if ("XFeaturesSelectedIntersection" %in% ActualFeatures) {
                               ActualFeatures <- ActualFeatures[-"XFeaturesSelectedIntersection"]
                               ActualFeatures <- c(ActualFeatures, FeaturesSelectedIntersection)
                             }
                             BalAccPerAlgorithm <-
                               parallel::mclapply(seq(FeatureSelectionAlgorithms),
                                                  function(i1) {
                                                    BalAccSingleClassifier <-
                                                      parallel::mclapply(seq(Seeds), function(i) {
                                                        BalAcc <- vector(mode = "list", length = 1)
                                                        ActualTrainDataReduced <- lapply(PreparedData[[i1]], function(x) {
                                                          lapply(x, function(y) {
                                                            y[c("Classes", ActualFeatures)]
                                                          }) })[[i]]$TrainDataAndClsScaled
                                                        ActualTestDataReduced <- lapply(PreparedData[[i1]], function(x) {
                                                          lapply(x, function(y) {
                                                            y[c("Classes", ActualFeatures)]
                                                          }) })[[i]]$TestDataAndClsScaled
                                                        PerfMeas <-
                                                          TrainClassifierAndEstimatePerformance(TrainDataAndClsScaled = ActualTrainDataReduced,
                                                                                                TestValidationDataAndClsScaled = ActualTestDataReduced,
                                                                                                Classifier = FeatureSelectionAlgorithms[i1],
                                                                                                Type = "Classes", Measure = "CM",
                                                                                                nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
                                                                                                Seed = Seeds[i] )
                                                        BalAcc <- PerfMeas$CM["Balanced Accuracy", ]
                                                        return(BalAcc)
                                                      }, mc.cores = nProc)
                                                    BalAccAlgorithm <- mean(unlist(BalAccSingleClassifier), na.rm = TRUE)
                                                    return(BalAccAlgorithm)
                                                  }, mc.cores = nProc)
                             BalAccCombination <- round(mean(unlist(BalAccPerAlgorithm), trim = 0.1, na.rm = TRUE), 2)
                             return(list(BA = BalAccCombination, Features = ActualFeatures))
                           }, mc.cores = nProc)
    }, mc.cores = nProc)

  BAcomb <- unlist(lapply(
    FeaturesSelectedCombinations, function(x) lapply(x, "[[", "BA") ))
  FeatueComb <- do.call(c, unlist(lapply( FeaturesSelectedCombinations, function(x) lapply(x, "[", "Features") ), recursive = FALSE))
  nFeaturesMaxBAcomb <- lapply(FeatueComb, length)
  BestBA <- which(BAcomb == max(BAcomb) )
  BestFeatureSet <-   unlist(FeatueComb[BestBA[which.min(nFeaturesMaxBAcomb)]])

  return(BestFeatureSet)
}

