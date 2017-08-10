# spark-imbalance


- Stratified cross validation

```
  val randomForestCrossValidation = new RandomForestCrossValidation(labelIndexer, indexerPipeline, trainingData, testData)
  randomForestCrossValidation.avgMetricsParamGrid
  randomForestCrossValidation.combined
```

- Tree printer

TODO

```
  private lazy val continuousFeatures = Seq()
  private lazy val categoricalFeatures = Seq()

  val labelNames = new Labels(labelIndexer)
  val featuresNames = new Labels(
    (continuousFeatures ++ categoricalFeatures).toArray
  )
  val categories = categoricalFeatures.toArray.indices
    .map(index => index -> new Labels(categoricalFeaturesIndexer.labels)).toMap

  val treePrinter: TreePrinter = new TreePrinter(bestRandomForestModel.trees(0), labelNames, featuresNames, categories)
  val tree: String = treePrinter.toJson
```

- SMOTE-NC

TODO

- Useful UDFs (ensemble)

TODO