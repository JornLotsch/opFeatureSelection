# Predict class membership using the trained classifier object
#' @importFrom stats predict
#' @importFrom methods is
#' @importFrom scales rescale
PredictClassesAndProbs <- function(TestValidationDataAndClsScaled, Classifier, TrainedClassifierObject,
                           Type = c("Classes", "Probs")) {
  Pred0 <- rep(1, length(TestValidationDataAndClsScaled$Classes))
  Pred2 <- matrix(0, ncol = length(unique(TestValidationDataAndClsScaled$Classes)), nrow = length(TestValidationDataAndClsScaled$Classes))
  colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)

  if ("Classes" %in% Type) {
    switch(Classifier, ADA = {
      Pred0 <- predict(TrainedClassifierObject, TestValidationDataAndClsScaled, type = "response")
      Pred0 <- ifelse(Pred0 > 0.5, unique(TestValidationDataAndClsScaled$Classes)[2],
                      unique(TestValidationDataAndClsScaled$Classes)[1])
    }, CTREE = {
      Pred0 <- predict(TrainedClassifierObject, TestValidationDataAndClsScaled, type = "response")
    }, kNN = {
      Pred0 <- apply(TrainedClassifierObject, 1, which.max)
    }, BinREG = {
      Pred0 <- predict(TrainedClassifierObject, TestValidationDataAndClsScaled, type = "response")
      Pred0 <- ifelse(Pred0 > 0.5, unique(TestValidationDataAndClsScaled$Classes)[2],
                      unique(TestValidationDataAndClsScaled$Classes)[1])
    }, {
      Pred0 <- try(predict(TrainedClassifierObject, TestValidationDataAndClsScaled,
                           type = "class"), TRUE)
      if (!is(Pred0[1], "try-error")) {
        Pred0 <- as.integer(Pred0)
      }
      Pred0
    })
  }

  if ("Probs" %in% Type) {
    switch(Classifier, ADA = {
      Pred2 <- matrix(rep(scales::rescale(Pred0, to = c(0, 1)), length(unique(TestValidationDataAndClsScaled$Classes))),
                      ncol = length(unique(TestValidationDataAndClsScaled$Classes)), nrow = length(TestValidationDataAndClsScaled$Classes))
      colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)
    }, kNN = {
      Pred2 <- matrix(rep(scales::rescale(Pred0, to = c(0, 1)), length(unique(TestValidationDataAndClsScaled$Classes))),
                      ncol = length(unique(TestValidationDataAndClsScaled$Classes)), nrow = length(TestValidationDataAndClsScaled$Classes))
      colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)
    }, MARS = {
      Pred2 <- matrix(rep(scales::rescale(Pred0, to = c(0, 1)), length(unique(TestValidationDataAndClsScaled$Classes))),
                      ncol = length(unique(TestValidationDataAndClsScaled$Classes)), nrow = length(TestValidationDataAndClsScaled$Classes))
      colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)
    }, SVM = {
      Pred2 <- matrix(rep(scales::rescale(Pred0, to = c(0, 1)), length(unique(TestValidationDataAndClsScaled$Classes))),
                      ncol = length(unique(TestValidationDataAndClsScaled$Classes)), nrow = length(TestValidationDataAndClsScaled$Classes))
      colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)
    }, CTREE = {
      Pred2 <- matrix(unlist(predict(TrainedClassifierObject, TestValidationDataAndClsScaled,
                                     type = "prob")), ncol = length(unique(TestValidationDataAndClsScaled$Classes)),
                      nrow = length(TestValidationDataAndClsScaled$Classes))
      colnames(Pred2) <- unique(TestValidationDataAndClsScaled$Classes)
    }, BinREG = {
      Pred2 <- predict(TrainedClassifierObject, TestValidationDataAndClsScaled, type = "response")
    }, nBayes = {
      Pred2 <- try(predict(TrainedClassifierObject, TestValidationDataAndClsScaled,
                           type = "raw"), TRUE)
    }, {
      Pred2 <- try(predict(TrainedClassifierObject, TestValidationDataAndClsScaled,
                           type = "prob"), TRUE)
    })
  }
  return(list(PredClasses = Pred0, PredProbs = Pred2))
}
