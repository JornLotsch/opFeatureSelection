# Train the classifier
#' @importFrom stats na.omit binomial
#' @importFrom rattle xgboost
#' @importFrom stats glm
#' @importFrom RWeka J48
#' @importFrom C50 C5.0
#' @importFrom party ctree ctree_control
#' @importFrom KernelKnn KernelKnn
#' @importFrom nnet multinom
#' @importFrom earth earth
#' @importFrom e1071 naiveBayes svm
#' @importFrom rpart rpart rpart.control
#' @importFrom randomForest randomForest
CreateTrainedClassifierObject <- function(TrainDataAndClsScaled, TestValidationDataAndClsScaled,
                                                Classifier, kNNk, RFtrees, Seed) {
  if (missing(kNNk))
    kNNk = 7
  set.seed(Seed)
  switch(Classifier, ADA = {
    TrainedClassifierObject <- rattle::xgboost(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                              max_depth = 10, eta = 0.25, num_parallel_tree = 5, nthread = 1, nround = 50,
                                              verbose = F)
  }, BinREG = {
    TrainedClassifierObject <- glm(factor(Classes) ~ ., family = binomial(link = "logit"),
                                  data = TrainDataAndClsScaled)
  }, C4.5 = {
    TrainedClassifierObject <- RWeka::J48(as.factor(Classes) ~ ., data = TrainDataAndClsScaled)
  }, C5.0 = {
    TrainedClassifierObject <- C50::C5.0(as.factor(Classes) ~ ., data = TrainDataAndClsScaled)
  }, C5.0rules = {
    TrainedClassifierObject <- C50::C5.0(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                        rules = T)
  }, CTREE = {
    TrainedClassifierObject <- party::ctree(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                           control = party::ctree_control(minbucket = 1, mincriterion = 0.9,
                                                                          minsplit = 5))
  }, kNN = {
    y = as.vector(as.integer(TrainDataAndClsScaled$Classes))
    TrainedClassifierObject <- KernelKnn::KernelKnn(data = as.matrix(TrainDataAndClsScaled[,
                                                                                          2:ncol(TrainDataAndClsScaled)]), TEST_data = as.matrix(TestValidationDataAndClsScaled[,
                                                                                                                                                                      2:ncol(TestValidationDataAndClsScaled)]), y = as.vector(as.integer(TrainDataAndClsScaled$Classes)),
                                                   method = "euclidean", regression = F, k = kNNk, Levels = unique(y),
                                                   threads = 1)
  }, loglinREG = {
    TrainedClassifierObject <- nnet::multinom(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                             trace = FALSE, maxit = 1000)
  }, MARS = {
    TrainedClassifierObject <- earth::earth(as.factor(Classes) ~ ., data = TrainDataAndClsScaled)
  }, nBayes = {
    TrainedClassifierObject <- e1071::naiveBayes(as.factor(Classes) ~ ., data = as.data.frame(TrainDataAndClsScaled),
                                                usekernel = F)
  }, PART = {
    TrainedClassifierObject <- RWeka::PART(as.factor(Classes) ~ ., data = TrainDataAndClsScaled)
  }, RIPPER = {
    TrainedClassifierObject <- RWeka::JRip(as.factor(Classes) ~ ., data = TrainDataAndClsScaled)
  }, Rpart = {
    TrainedClassifierObject <- rpart::rpart(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                           method = "class", xval = 1000, parms = list(split = "information"),
                                           control = rpart::rpart.control(cp = 0.01, maxdepth = 30, minsplit = 5))
  }, RF = {
    TrainedClassifierObject <- randomForest::randomForest(as.factor(Classes) ~
                                                           ., data = TrainDataAndClsScaled, ntree = RFtrees, mtry = 1 * sqrt(length(names(TrainDataAndClsScaled)) -
                                                                                                                               1), na.action = randomForest::na.roughfix, strata = TrainDataAndClsScaled$Classes,
                                                         replace = T)
  }, SVM = {
    TrainedClassifierObject <- e1071::svm(as.factor(Classes) ~ ., data = TrainDataAndClsScaled,
                                         method = "C-classification", kernel = "linear", probability = TRUE)
  })
  return(TrainedClassifierObject)
}
