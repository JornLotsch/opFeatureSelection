# Performs feature selection based on the variables' importance for the performance of algorithms
# combines the thus selected features across several different algorithms
# and evaluates the performance of the combinaed final feature set in different algorithms
opFeatureSelection <- function(Data, Cls, ClassifierChoice = c("RF", "kNN"),
                               FeatureSelectionAlgorithm = "each", SelectedFeatures,
                               CvChoice = c("all", "allPermuted", "reduced", "reducedPermuted"),
                               Scalings = "none", nIter = 100, MaxCores = 2048,
                               SplitRatio = 0.67, SeparateValidationRatio = 0,
                               kNNk = 5, RFtrees = 1500, Seed) {
  # Check input
  list.of.possible.classifiers <- c("ADA", "BinREG", "C4.5", "C5.0", "C5.0rules", "CTREE",
                                    "kNN", "loglinREG", "MARS", "nBayes", "PART", "RIPPER", "Rpart", "RF", "SVM")
  if (min(ClassifierChoice %in% list.of.possible.classifiers) < 1 |
      min(FeatureSelectionAlgorithm %in% c(list.of.possible.classifiers, "each", "none") < 1)) {
    stop(paste0("opFeatureSelection: ClassifierChoice can contain only the following items",
                list.of.possible.classifiers,
                " FeatureSelectionAlgorithm can also contain 'each' or 'none'. Stopping."))
  }
  if (min(CvChoice %in% c("all", "allPermuted", "reduced", "reducedPermuted", "none")) < 1) {
    stop(paste0("opFeatureSelection: CvChoice can be only 'all', 'allPermuted', 'reduced',
                'reducedPermuted' and/or 'none'. Stopping."))
  }
  list.of.possible.scalings <- c("none", "range", "z")
  if (min(Scalings %in% list.of.possible.scalings) < 1) {
    stop(paste0("opFeatureSelection: Only the follwing scalings are implemented",
                list.of.possible.scalings, ". Stopping."))
  }
  if (hasArg("Data") == FALSE | hasArg("Cls") == FALSE) {
    stop("opFeatureSelection: No data or classes provided. Stopping.")
  }
  if (sum(is.na(Data)) > 0) {
    stop("opFeatureSelection: Data must not contain NAs. Stopping.")
  }
  if (sum(is.na(Data)) > 0) {
    stop("opFeatureSelection: Data must not contain NAs. Stopping.")
  }
  nClasses <- length(unique(Cls))
  if (nClasses < 2) {
    stop("opFeatureSelection: At least two classes must be provided. Stopping.")
  }
  if ("BinREG" %in% ClassifierChoice & nClasses != 2) {
    if (length(unique(ClassifierChoice)) > 1) {
      ClassifierChoice <- ClassifierChoice[-"BinREG"]
      warning("opFeatureSelection: More than two classes provided.
              Binary regression dropped from ClassifierChoice.",
              call. = FALSE)
    } else {
      stop("opFeatureSelection: More than two classes provided. Binary regression impossible. Stopping.")
    }
  }

  num_workers <- parallel::detectCores()
  nProc <- min(num_workers - 1, MaxCores)

  # Prepare internal variable values
  DataToProcess <- cbind.data.frame(Classes = Cls, Data)

  if (FeatureSelectionAlgorithm == "each" & FeatureSelectionAlgorithm != "none") {
    list.of.FeatureSelectionAlgorithms <- ClassifierChoice
  } else {
    list.of.FeatureSelectionAlgorithms <- FeatureSelectionAlgorithm
  }

  FeaturesSelected <- names(Data)
  if (hasArg("SelectedFeatures") == TRUE) {
    if (min(SelectedFeatures %in% names(Data)) < 1) {
      stop(paste0("opFeatureSelection: Names of 'SelectedFeatures' do not match names of the variables. Stopping."))
    } else {
      FeaturesSelected <- SelectedFeatures
    }
  }

  if (length(Scalings) > 1) {
    if (length(Scalings) == length(ClassifierChoice)) {
      list.of.scalings <- Scalings
    } else {
      warning("opFeatureSelection: Number of scaling methods != number of classifiers.
        Scaling defaulting to 'none' for all classifiers.", call. = FALSE)
      list.of.scalings <- rep("none", length(ClassifierChoice))
    }
  } else {
    list.of.scalings <- rep(Scalings, length(ClassifierChoice))
  }

  if (!missing(Seed)) {
    ActualSeed <- Seed
  } else {
    ActualSeed <- tail(get(".Random.seed", envir = globalenv()), 1)
  }
  list.of.seeds <- 1:nIter + ActualSeed - 1
  list.of.nestings <- rep(1:nNestings, each = round(length(list.of.seeds)/nNestings), length.out = length(list.of.seeds))
  list.of.seeds.nestings <- rbind(list.of.seeds, list.of.nestings)


  ### Main function
  # Prepare spitted data sets for each algorithm abd for each iteration
  DataToProcessPrepared <- parallel::mclapply(seq(list.of.scalings), function(x) {
    PrepaireDataForClassification(DataToProcess = DataToProcess,
                                  Scaling = list.of.scalings[x], SplitRatio = SplitRatio,
                                  nNestings = nNestings, Seeds = list.of.seeds, SeparateValidationRatio = SeparateValidationRatio)
  }, mc.cores = nProc)
  names(DataToProcessPrepared) <- ClassifierChoice

  # Select features
  if (FeatureSelectionAlgorithm != "none") {
    DataToProcessPreparedFeatureSelection <- DataToProcessPrepared[which(list.of.FeatureSelectionAlgorithms %in% ClassifierChoice)]
    FeaturesSelectedAll <- parallel::mclapply(seq(list.of.FeatureSelectionAlgorithms), function(x) {
      SelectFeaturesOneClassifier(DataToProcessPrepared = DataToProcessPreparedFeatureSelection[[x]],
                                  Classifier = list.of.FeatureSelectionAlgorithms[x],
                                  nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
                                  Seeds = list.of.seeds)
    }, mc.cores = nProc)
    names(FeaturesSelectedAll) <- list.of.FeatureSelectionAlgorithms

    FeaturesSelected <- CreateCombinedFeaturesSelected(PreparedData = DataToProcessPreparedFeatureSelection,
                                                       FeaturesPreSelected = FeaturesSelectedAll,
                                                       FeatureSelectionAlgorithms = list.of.FeatureSelectionAlgorithms,
                                                       Type = "Classes", Measure = "CM", nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
                                                       Seeds = list.of.seeds)
  }

  # Test and validate the algorithms with all and with the selected features
  ClassifierPerformances <- TestCvClassifierPerformance(DataToProcessPrepared = DataToProcessPrepared,
                                    CvChoice = c("all", "allPermuted", "reduced", "reducedPermuted"),
                                    SelectedFeatures = FeaturesSelected, ClassifierChoice = ClassifierChoice,
                                    Type = c("Classes", "Probs"), Measure = c("CM", "ROC"),
                                    nClasses = nClasses, kNNk = kNNk, RFtrees = RFtrees,
                                    Seeds = list.of.seeds)


xx <- ClassifierPerformances[[1]]

for(i1 in ClassifierChoice) {
  for (i2 in CvChoice) {
    print(
      matrixStats::rowQuantiles(as.matrix(ClassifierPerformances[[i1]][[i2]]), probs = c(0.025, 0.5,0.975), na.rm = T) * 100)
  }
}


yy <- lapply(CvChoice, function(x) matrixStats::rowQuantiles(ClassifierPerformances[[1]][[x]],
                                                             probs = c(0.025, 0.5,0.975), na.rm = T) * 100)


apply(yy,1,mean)


  lapply(xx[x],"[[", 1))

xx[["all"]]

lapply(ClassifierChoice, function(x)
  lapply(CvChoice, function(y)
    lapply(xx, "[[", )
         )

       )

}

