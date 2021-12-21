# Calculate classification performance measures
#' @importFrom caret confusionMatrix
#' @importFrom pROC multiclass.roc
CalculatePerformanceMeasures <- function(PredClasses, PredProbs, TestValidationDataAndClsScaled,
                                         nClasses, Measure = c("CM", "ROC")) {
  rocAUC <- 0
  if (missing(Measure)) {
    Measure <- c("CM", "ROC")
  }
  if ("ROC" %in% Measure & missing(PredProbs)) {
    stop("opFeatureSelection: ROC requires probability matrix of class assigment. Stopping.")
  }
  if ("CM" %in% Measure) {
    cTab <- table(factor(PredClasses, levels = 1:nClasses), factor(TestValidationDataAndClsScaled$Classes,
                                                                   levels = 1:nClasses))

    ifelse(nClasses > 2, cMat <- t(caret::confusionMatrix(cTab)$byClass),
           cMat <- caret::confusionMatrix(cTab)$byClass)
  }
  if ("ROC" %in% Measure) {
    if (!"try-error" %in% class(PredProbs)) {
      rocAUC <- as.numeric(pROC::multiclass.roc(TestValidationDataAndClsScaled$Classes,
                                                PredProbs, quiet = T)$auc)
    } else rocAUC <- 0
  }
  return(list(CM = cMat, ROC = rocAUC))
}
