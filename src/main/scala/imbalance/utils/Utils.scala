package imbalance.utils

import org.apache.spark.ml.linalg.{Vector => SparkVector}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

object Utils {

  //TODO: explain how it is used
  def featuresImportanceOrdered(featuresMetadata : String, importance : SparkVector) : Seq[(String, Double)] = {
    val pattern = "\"idx\":(\\d+),\"name\":\"([^\"]+)\"".r
    var featuresNames  : Map[String,String] = Map()
    pattern.findAllIn(featuresMetadata).matchData foreach {
      m => featuresNames += (m.group(1) -> m.group(2))
    }
    var featuresImportance  : Map[String,Double] = Map()
    featuresNames foreach {
      case (k,v) => featuresImportance += (v -> importance(k.toInt))
    }
    featuresImportance.toSeq.sortBy(_._2).reverse
  }

  //TODO: rename to hard ensemble
  val ensembleLabels: Seq[Double] => String = predictions => {
    val totalPredictions = predictions.sum
    val threshold = predictions.length / 2.0
    totalPredictions match {
      case _ if totalPredictions > threshold => "FRAUD"
      case _ => "NOT FRAUD"
    }
  }
  val ensembleLabelsUDF: UserDefinedFunction = udf(ensembleLabels)

  //TODO: rename to soft ensemble
  //Called soft ensemble (better if well calibrated classifiers)
  val ensembleProbabilities: Seq[Double] => String = probabilities => {
    val fraudProbability = probabilities.sum / probabilities.length
    val threshold = 0.5
    fraudProbability match {
      case probability if probability > threshold => "FRAUD"
      case _ => "NOT FRAUD"
    }
  }
  val ensembleProbabilitiesUDF: UserDefinedFunction = udf(ensembleProbabilities)

}
