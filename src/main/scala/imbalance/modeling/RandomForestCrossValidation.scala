package imbalance.modeling

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidatorModel, ParamGridBuilder, StratifiedCrossValidator}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame

class RandomForestCrossValidation(labelIndexer: StringIndexerModel,
                                  indexerPipeline: PipelineModel,
                                  trainingData: DataFrame,
                                  testData: DataFrame) {

  //TODO: read this values + param grid from a config file
  val numTrees = 100
  val maxDepth = 16
  val maxBins = 150
  val subSamplingRate = 0.4
  val impurity = "entropy"
  val featureSubsetStrategy = "onethird" //changed from sqrt

  lazy val modelParams: Map[String, Any] = Map(
    "numTrees" -> numTrees,
    "maxDepth" -> maxDepth,
    "maxBins" -> maxBins,
    "subSamplingRate" -> subSamplingRate,
    "impurity" -> impurity,
    "featureSubsetStrategy" -> featureSubsetStrategy
  )

  lazy private val rf = new RandomForestClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("features")
    .setNumTrees(modelParams("numTrees").asInstanceOf[Int])
    .setMaxDepth(modelParams("maxDepth").asInstanceOf[Int])
    .setMaxBins(modelParams("maxBins").asInstanceOf[Int])
    .setSubsamplingRate(modelParams("subSamplingRate").asInstanceOf[Double])
    .setImpurity(modelParams("impurity").asInstanceOf[String])
    .setFeatureSubsetStrategy(modelParams("featureSubsetStrategy").asInstanceOf[String])

  // Convert indexed labels back to original labels.
  lazy private val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and forest in a Pipeline.
  lazy private val pipeline: Pipeline = new Pipeline()
    .setStages(
      Array(indexerPipeline, rf, labelConverter)
    )

  private val paramGrid = new ParamGridBuilder()
    .addGrid(rf.featureSubsetStrategy, Array("4", "log2", "sqrt", "onethird", "40"))
    .addGrid(rf.impurity, Array("gini", "entropy"))
    .build()

  private lazy val areaUnderPREvaluator = new BinaryClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("rawPrediction")
    .setMetricName("areaUnderPR")

  // No parameter search
  private val cv = new StratifiedCrossValidator(
    majorityClass = "NOT FRAUD",
    minorityClass = "FRAUD",
    samplingFraction = 80.0,
    samplingSeed = 11L
  )
    // ml.Pipeline with ml.classification.RandomForestClassifier
    .setEstimator(pipeline)
    // ml.evaluation.MulticlassClassificationEvaluator
    .setEvaluator(areaUnderPREvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)

  val modelCV: CrossValidatorModel = cv.fit(trainingData)

  // Print the average metrics per ParamGrid entry
  val avgMetricsParamGrid: Array[Double] = modelCV.avgMetrics
  // Combine with paramGrid to see how they affect the overall metrics
  val combined: Array[(ParamMap, Double)] = paramGrid.zip(avgMetricsParamGrid)

  val bestModel: PipelineModel = modelCV.bestModel.asInstanceOf[PipelineModel]
  val bestRandomForestModel: RandomForestClassificationModel = bestModel.stages(1).asInstanceOf[RandomForestClassificationModel]
  //  val bestHashingTFNumFeatures : String = bestModel.stages(1).asInstanceOf[RandomForestClassificationModel].explainParams

  val predictions: DataFrame = modelCV.transform(testData)
}
