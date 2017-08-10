package org.apache.spark.ml.tuning

import com.github.fommil.netlib.F2jBLAS
import org.apache.spark.ml.Model
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}

class StratifiedCrossValidator(
                                majorityClass: String,
                                minorityClass: String,
                                samplingFraction: Double,
                                samplingSeed: Long
                              ) extends CrossValidator {

  private val f2jBLAS = new F2jBLAS

  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val sparkSession = dataset.sparkSession
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = $(estimatorParamMaps)
    val numModels = epm.length
    val metrics = new Array[Double](epm.length)

    //Divide the dataset in frauds and non frauds
    val frauds = dataset.toDF.filter(s"label == '$minorityClass'").distinct.rdd //FRAUD
    val nonFrauds = dataset.toDF.filter(s"label == '$minorityClass'").rdd //NOT FRAUD
    //Run k-fold in both RDDs. Each split contains the training and test.
    val fraudSplits: Array[(RDD[Row], RDD[Row])] = MLUtils.kFold(frauds, $(numFolds), $(seed))
    val nonFraudsSplits: Array[(RDD[Row], RDD[Row])] = MLUtils.kFold(nonFrauds, $(numFolds), $(seed))
    //Combine the frauds and non frauds splits. The frauds are oversampled.
    val zippedData = fraudSplits.zip(nonFraudsSplits)
    val splits: Array[(RDD[Row], RDD[Row])] = zippedData
      .map { case ((fraudsTraining, fraudsTest), (nonFraudsTraining, nonFraudsTest)) =>
        //TODO: the seed and the fraction could be parameters taken from outside
        val fraudsTrainingOversampled = fraudsTraining.sample(
          withReplacement = true, fraction = samplingFraction, seed = samplingSeed)  //seed 11L and fraction 80.0
        (fraudsTrainingOversampled.union(nonFraudsTraining), fraudsTest.union(nonFraudsTest))
      }

    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache()
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache()
      // multi-model training
      logDebug(s"Train split $splitIndex with multiple sets of parameters.")
      val models = est.fit(trainingDataset, epm).asInstanceOf[Seq[Model[_]]]
      trainingDataset.unpersist()
      var i = 0
      while (i < numModels) {
        val metric = eval.evaluate(models(i).transform(validationDataset, epm(i)))
        logDebug(s"Got metric $metric for model trained with ${epm(i)}.")
        metrics(i) += metric
        i += 1
      }
      validationDataset.unpersist()
    }
    //scales a vector by a constant. DSCAL(N,DA,DX,INCX)
    f2jBLAS.dscal(numModels, 1.0 / $(numFolds), metrics, 1)
    logInfo(s"Average cross-validation metrics: ${metrics.toSeq}")
    val (bestMetric, bestIndex) =
      if (eval.isLargerBetter) metrics.zipWithIndex.maxBy(_._1)
      else metrics.zipWithIndex.minBy(_._1)
    logInfo(s"Best set of parameters:\n${epm(bestIndex)}")
    logInfo(s"Best cross-validation metric: $bestMetric.")
    val bestModel = est.fit(dataset, epm(bestIndex)).asInstanceOf[Model[_]]
    copyValues(new CrossValidatorModel(uid, bestModel, metrics).setParent(this))
  }
}