# Train classifier and estimate its performance
TrainClassifierAndEstimatePerformance <- function(TrainDataAndClsScaled, TestValidationDataAndClsScaled,
                                          Classifier, Type, Measure, nClasses, kNNk, RFtrees, Seed) {

  ActualClassifierObject <- CreateTrainedClassifierObject(TrainDataAndClsScaled = TrainDataAndClsScaled,
                                                          TestValidationDataAndClsScaled = TestValidationDataAndClsScaled,
                                                          Classifier = Classifier, kNNk = kNNk, RFtrees = RFtrees,
                                                          Seed = Seed)

  ActualClassPrediction <- PredictClassesAndProbs(TestValidationDataAndClsScaled = TestValidationDataAndClsScaled,
                                                  Classifier = Classifier,
                                                  TrainedClassifierObject = ActualClassifierObject, Type = Type)

  PerformanceMeasures <- CalculatePerformanceMeasures(PredClasses = ActualClassPrediction$PredClasses,
                                                      PredProbs = ActualClassPrediction$PredProbs,
                                                      TestValidationDataAndClsScaled = TestValidationDataAndClsScaled,
                                                      Measure = Measure, nClasses = nClasses)
  return(PerformanceMeasures)
}
