# Preparing the data per classifier Scaling, nesting, splitting into training,
# test and validation data sets If no nested cross-validation is requested, i
# e., nNestings == 1, then the train and test data subsets and the validation
# data subset are the same
#' @importFrom stats na.omit
#' @importFrom caTools sample.split
PrepaireDataForClassification <-
  function(DataToProcess, SeparateValidationDataToProcess,
           Scaling = "none", SplitRatio = 0.67,
           nTrialsopdisDownsampling = nTrialsopdisDownsampling,
           SeparateValidationFraction = 0, list.of.seeds.nestings) {

    ### Internal functions
    # Scaling function
    ScaleData <- function(Data, Scaling) {
      switch(Scaling, z = {
        cbind.data.frame(Classes = Data$Classes,
                         apply(Data[, 2:ncol(Data)], 2, function(x) scale(x, center = T, scale = T)))
      }, range = {
        rangeScale <- function(x) {
          return(diff(range(na.omit(x))))
        }
        ranges <- apply(Data[2:ncol(Data)], 2, rangeScale)
        cbind.data.frame(Classes = Data$Classes, scale(Data[, 2:ncol(Data)],
                                                       scale = ranges))
      }, Data)
    }

    # Splitting function
    SplitTrainingTest <- function(Data, Seed) {
      set.seed(Seed)
      sample <- caTools::sample.split(Data$Classes, SplitRatio = SplitRatio)
      TrainDataScaled <- subset(Data, sample == TRUE)
      TestDataScaled <- subset(Data, sample == FALSE)
      TrainDataScaledPermuted <-
        cbind.data.frame(Classes = TrainDataScaled$Classes,
                         apply(TrainDataScaled[-1], 2, sample))
      return(list(TrainDataAndClsScaled = TrainDataScaled, TestDataAndClsScaled = TestDataScaled,
                  TrainDataAndClsScaledPermuted = TrainDataScaledPermuted))
    }

    # Nesting function
    CreateNestedData <-
      function(Data, DataToProcessScaledSeparateValidation, nNestings, list.of.seeds.nestings) {

        # Mark data set for nesting
        list.of.nestings.perCls <-
          unlist(lapply(unique(Data$Classes), function(x) {
            set.seed(list.of.seeds.nestings[1, 1])
            sample(rep(seq(nNestings), length(Data$Classes[Data$Classes == x]) / length(Data$Classes) *
                         length(Data$Classes) / nNestings, length.out = length(Data$Classes[Data$Classes == x])))
          }))
        DataToProcessScaledNestingsMarked <- cbind.data.frame(Data, list.of.nestings.perCls)

        DataToProcessNested <- lapply(1:nNestings, function(i) {
          # Split the data set into two parts, for either sub sampling of training
          # and test data sets, or as separated validation sample
          if (nNestings > 1) {
            DataToProcessNestedTrainingTest <-
              DataToProcessScaledNestingsMarked[DataToProcessScaledNestingsMarked$list.of.nestings.perCls != i, ]
          } else {
            DataToProcessNestedTrainingTest <-
              DataToProcessScaledNestingsMarked[DataToProcessScaledNestingsMarked$list.of.nestings.perCls == i, ]
          }
          DataToProcessNestedValidation <-
            DataToProcessScaledNestingsMarked[DataToProcessScaledNestingsMarked$list.of.nestings.perCls == i, ]
          DataToProcessNestedTrainingTest <-
            DataToProcessNestedTrainingTest[, - ncol(DataToProcessNestedTrainingTest)]
          DataToProcessNestedValidation <-
            DataToProcessNestedValidation[, - ncol(DataToProcessNestedValidation)]

          return(list(DataToProcessNestedTrainingTest = DataToProcessNestedTrainingTest,
                      DataToProcessNestedValidation = DataToProcessNestedValidation))
        })
        return(DataToProcessNested)
      }

    # Function to assemble the data sets
    AssembleNestedDataSets <- function(DataToProcessNested, SeparateValidationDataScaled,
                                       SplitRatio = 0.67, Seed, Nesting, nNestings) {
      set.seed(Seed)
      # Split die training and test data subsets
      sample <-
        caTools::sample.split(DataToProcessNested[[Nesting]]$DataToProcessNestedTrainingTest$Classes,
                              SplitRatio = 0.67)
      TrainDataScaled <-
        subset(DataToProcessNested[[Nesting]]$DataToProcessNestedTrainingTest,
               sample == TRUE)
      TestDataScaled <-
        subset(DataToProcessNested[[Nesting]]$DataToProcessNestedTrainingTest,
               sample == FALSE)

      # Select the validation sample, either defined during nested cross
      # validation or place the test data as validation data
      if (nNestings > 1) {
        ValidationDataScaled <-
          DataToProcessNested[[Nesting]]$DataToProcessNestedValidation
      } else {
        ValidationDataScaled <- TestDataScaled
      }

      # Add the separate validation sample, only drawn once to the return list or
      # just take the validation sample from the nested cross validation
      if (dim(DataToProcessScaledSeparateValidation)[1] > 0) {
        SeparateValidationDataScaled <- DataToProcessScaledSeparateValidation
      } else {
        SeparateValidationDataScaled <- ValidationDataScaled
      }

      # Create permuted data sets from all subsets
      TrainDataScaledPermuted <-
        cbind.data.frame(Classes = TrainDataScaled$Classes,
                         apply(TrainDataScaled[-1], 2, sample))
      TestDataScaledPermuted <-
        cbind.data.frame(Classes = TestDataScaled$Classes,
                         apply(TestDataScaled[-1], 2, sample))
      ValidationDataScaledPermuted <-
        cbind.data.frame(Classes = ValidationDataScaled$Classes,
                         apply(ValidationDataScaled[-1], 2, sample))
      SeparateValidationDataScaledPermuted <-
        cbind.data.frame(Classes = SeparateValidationDataScaled$Classes,
                         apply(SeparateValidationDataScaled[-1], 2, sample))

      return(list(TrainDataAndClsScaled = TrainDataScaled, TestDataAndClsScaled = TestDataScaled,
                  TrainDataAndClsScaledPermuted = TrainDataScaledPermuted, TestDataAndClsScaledPermuted = TestDataScaledPermuted,
                  ValidationDataScaled = ValidationDataScaled, ValidationDataScaledPermuted = ValidationDataScaledPermuted,
                  SeparateValidationDataScaled = SeparateValidationDataScaled, SeparateValidationDataScaledPermuted = SeparateValidationDataScaledPermuted))
    }

    ### Main function
    nNestings = max(list.of.seeds.nestings[2,])
    DataToProcess$Classes <- factor(as.integer(DataToProcess$Classes))
    DataToProcessScaled <- ScaleData(Data = DataToProcess, Scaling = Scaling)

    if (dim(SeparateValidationDataToProcess)[1] > 0) {
      DataToProcessScaledSeparateValidation <- SeparateValidationDataToProcess
      if (SeparateValidationFraction > 0) {
        warning("opFeatureSelection: SeparateValidationData provided, which will be used.
              SeparateValidationFraction will be ignored.",
                call. = FALSE)
        SeparateValidationFraction <- 0
      }
    } else {
      DataToProcessScaledDownsampled <-
        opdisDownsampling::opdisDownsampling(Data = DataToProcessScaled[, -1],
                                             Cls = DataToProcessScaled$Classes,
                                             Size = length(DataToProcess$Classes) * (1 - SeparateValidationFraction),
                                             Seed = list.of.seeds.nestings[1, 1], nTrials = nTrialsopdisDownsampling)
      DataToProcessScaled <-
        cbind.data.frame(Classes = DataToProcessScaledDownsampled$ReducedData$Cls,
                         DataToProcessScaledDownsampled$ReducedData[, - ncol(DataToProcessScaledDownsampled$ReducedData)])
      DataToProcessScaledSeparateValidation <-
        cbind.data.frame(Classes = DataToProcessScaledDownsampled$RemovedData$Cls,
                         DataToProcessScaledDownsampled$RemovedData[, - ncol(DataToProcessScaledDownsampled$RemovedData)])
    }

    DataToProcessNested <- CreateNestedData(Data = DataToProcessScaled,
                                            DataToProcessScaledSeparateValidation = DataToProcessScaledSeparateValidation,
                                            nNestings = nNestings,
                                            list.of.seeds.nestings = list.of.seeds.nestings)

    # Create data sets with required splits
    PrepairedData <- lapply(seq(list.of.seeds.nestings[1, ]), function(x) {
      AssembleNestedDataSets(DataToProcessNested = DataToProcessNested,
                             SeparateValidationDataScaled = SeparateValidationDataScaled,
                             SplitRatio = SplitRatio, nNestings = nNestings,
                             Seed = list.of.seeds.nestings[1, x], Nesting = list.of.seeds.nestings[2, x] )
    })
    names(PrepairedData) <- list.of.seeds.nestings[2, ]

    return(PrepairedData)
  }
