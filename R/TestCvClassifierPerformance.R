# Classifier performance testing in different scenarios of
# full and reduced feature sets with original or permuted training data
TestCvClassifierPerformance <- function(DataToProcessPrepared, CvChoice,
                                        SelectedFeatures, ClassifierChoice,
                                        Type, Measure, nClasses, kNNk, RFtrees, Seeds) {

  PerformanceTestingClassifiersAndCvChoises <- parallel::mclapply(seq(ClassifierChoice), function(Classifier_i) {
    DataToProcessPreparedActualClassifier <- DataToProcessPrepared[[ClassifierChoice[Classifier_i]]]
    PerformanceTesting_i <- parallel::mclapply(seq(Seeds), function(i) {
      PerformanceTestingCvChoises <- parallel::mclapply(CvChoice, function(CvChoice_i) {
        PerformanceMeasures <- vector(mode = "list", length = 2)
        names(PerformanceMeasures) <- c(paste0(CvChoice_i, "_CM"), paste0(CvChoice_i, "_ROC_AUC"))
        switch(CvChoice_i, all = {
          ActualTrainDataReduced <- DataToProcessPreparedActualClassifier[[i]]$TrainDataAndClsScaled
          ActualTestDataReduced <- DataToProcessPreparedActualClassifier[[i]]$ValidationDataScaled
        }, allPermuted = {
          ActualTrainDataReduced <- DataToProcessPreparedActualClassifier[[i]]$TrainDataAndClsScaledPermuted
          ActualTestDataReduced <- DataToProcessPreparedActualClassifier[[i]]$ValidationDataScaled
        }, reduced = {
          ActualTrainDataReduced <- lapply(DataToProcessPreparedActualClassifier, function(x) {
            lapply(x, function(y) {y[c("Classes", SelectedFeatures)] })
          })[[i]]$TrainDataAndClsScaled
          ActualTestDataReduced <- lapply(DataToProcessPreparedActualClassifier, function(x) {
            lapply(x, function(y) {y[c("Classes", SelectedFeatures)] })
          })[[i]]$TestDataAndClsScaled
        }, reducedPermuted = {
          ActualTrainDataReduced <- lapply(DataToProcessPreparedActualClassifier, function(x) {
            lapply(x, function(y) {y[c("Classes", SelectedFeatures)] })
          })[[i]]$TrainDataAndClsScaledPermuted
          ActualTestDataReduced <- lapply(DataToProcessPreparedActualClassifier, function(x) {
            lapply(x, function(y) {y[c("Classes", SelectedFeatures)] })
          })[[i]]$TestDataAndClsScaled
        })

        PerfMeas <- TrainClassifierAndEstimatePerformance(TrainDataAndClsScaled = ActualTrainDataReduced,
                                                          TestValidationDataAndClsScaled = ActualTestDataReduced,
                                                          Classifier = ClassifierChoice[Classifier_i],
                                                          Type = c("Classes", "Probs"), Measure = c("CM", "ROC"),
                                                          nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
                                                          Seed = i)
        PerformanceMeasures[[1]] <- PerfMeas$CM
        PerformanceMeasures[[2]] <- PerfMeas$ROC

        return(PerformanceMeasures)
      }, mc.cores = nProc)
    }, mc.cores = nProc)
    PerformanceMeasuresAllperClassifier <- lapply(CvChoice, function(CvChoice_i) {
      rbind.data.frame(do.call(cbind.data.frame,
                               lapply(lapply(PerformanceTesting_i, "[[", which(CvChoice == CvChoice_i)), "[[", 1)),
                       "ROC AUC" = suppressWarnings(as.numeric(unlist(strsplit(paste(unlist(lapply(lapply(PerformanceTesting_i,
                                                                                                          "[[", which(CvChoice == CvChoice_i)), "[[", 2)),
                                                                                     paste(rep(NA, nClasses - 1), collapse = ","), sep = ","), ","))), classes = "warning"))
    })
    names(PerformanceMeasuresAllperClassifier) <- CvChoice
    return(PerformanceMeasuresAllperClassifier)
  }, mc.cores = nProc)
  names(PerformanceTestingClassifiersAndCvChoises) <- ClassifierChoice
  return(PerformanceTestingClassifiersAndCvChoises)
}


xx <- TestCvClassifierPerformance(DataToProcessPrepared = )


TrainClassifierAndEstimatePerformance <- function(TrainDataAndClsScaled, TestValidationDataAndClsScaled,
                                                  Classifier, Type, Measure, nClasses, kNNk, RFtrees, Seed)



names(PerformanceMeasuresAll) <- CvChoice
ResultsPerClassifier[[Classifier_i]] <- list(PerformanceMeasuresAll = PerformanceMeasuresAll,
                                             FeaturesTimesinABCA = dfABCResamplinAallIter, FeaturesSelected = SelectedFeatures)
setTxtProgressBar(pb, Classifier_Counter)
