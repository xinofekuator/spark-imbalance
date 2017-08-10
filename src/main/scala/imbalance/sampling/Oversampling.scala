package imbalance.sampling

import breeze.linalg._
import breeze.linalg.functions._
import breeze.stats.{median, _}

import scala.util.Random

//TODO: maybe receive directly a regular breeze vector not to depend on Spark
// (if they are compatible and work locally for Unit Tests I don't care)

/**
  * *
  *
  * @param denseFeatures         : minority class data.
  * @param firstCategoricalIndex : this index indicates the position of the first categorical feature.
  *                              The first columns are continuous features and the last ones categorical.
  * @param neighborsNumber       : number of neighbors to apply KNN.
  * @param oversamplingRate      : oversampling rate (2 = 200% of the minority data as synthetic samples).
  */


//TODO: see if foreach can be changed to MAPS!!!!
//TODO: change Array to Seq!!!!

//TODO: be careful. The neighborsNumber can't be greater than the number of samples.
//TODO: change mode to enum
class Oversampling(denseFeatures: Array[DenseVector[Double]],
                   firstCategoricalIndex: Int,
                   misclassifiedObservations: Seq[Boolean],
                   neighborsNumber: Int = 5,
                   oversamplingRate: Int = 80,
                   mode: String = "normal") {

  val featuresNumber: Int = denseFeatures.head.length

  //TODO: create matrix and slice
  val continuousFeatures: Array[DenseVector[Double]] = denseFeatures.map(_ (0 until firstCategoricalIndex))
  val categoricalFeatures: Array[DenseVector[Double]] = denseFeatures.map(_ (firstCategoricalIndex until featuresNumber))

  val features: Features = new Features(continuousFeatures, categoricalFeatures)

  private val neighbors = features.neighbors(neighborsNumber)

  //TODO: pass Feature object as argument instead
  val syntheticSamples: Array[DenseVector[Double]] = new SMOTE(
    features.normalizedContinuous,
    features.categorical,
    neighbors,
    oversamplingRate,
    features.columnStats,
    features.isInteger,
    misclassifiedObservations,
    mode)
    .syntheticSamples

  val oversampledFeatures: Array[DenseVector[Double]] = syntheticSamples ++ denseFeatures

  class Features(continuousFeatures: Array[DenseVector[Double]], categoricalFeatures: Array[DenseVector[Double]]) {

    private val observationsNumber: Int = continuousFeatures.length
    private val observationsIndices: List[Int] = (0 until observationsNumber).toList

    private val continuousFeaturesNumber: Int = continuousFeatures.head.length
    //    private val categoricalFeaturesNumber: Int = categoricalFeatures.head.length

    val continuous: DenseMatrix[Double] = DenseMatrix(continuousFeatures: _*)
    val categorical: DenseMatrix[Double] = DenseMatrix(categoricalFeatures: _*)

    val isInteger: Seq[Boolean] = continuous(::, *)
      .map(column => column.toArray.forall(featureValue => featureValue == Math.floor(featureValue))).t
      .toArray
      .toSeq

    /**
      * https://en.wikipedia.org/wiki/Feature_scaling
      * x' = (x - min) / (max - min)
      */

    //TODO: min-max scaler exists in Spark also (that could speed it up)
    //TODO: this is WRONG!!!!
    //TODO: refactor this. First column stats and then map it applying minmax method.
    val (normalizedContinuous: DenseMatrix[Double], columnStats: Map[Int, ColumnStats]) = {
      val normalizedMatrix = DenseMatrix.zeros[Double](continuous.rows, continuous.cols)
      val rowIndices: List[Int] = (0 until observationsNumber).toList
      val columnIndices = (0 until continuousFeaturesNumber).toList
      var columnStats: Map[Int, ColumnStats] = Map()
      //TODO: remove foreach
      columnIndices.foreach {
        columnIndex =>
          val columnValues: DenseVector[Double] = continuous(::, columnIndex)
          val maxValue: Double = max(columnValues)
          val minValue: Double = min(columnValues)
          columnStats += (columnIndex -> ColumnStats(minValue, maxValue))
          rowIndices.foreach {
            rowIndex =>
              val normalizedValue: Double = (continuous(rowIndex, columnIndex) - minValue) / (maxValue - minValue)
              normalizedMatrix(rowIndex, columnIndex) = normalizedValue
          }
      }
      (normalizedMatrix, columnStats)
    }

    val standardDeviationMedian: Double = {
      val standardDeviations: DenseVector[Double] = DenseVector.zeros[Double](continuousFeaturesNumber)
      val indices: List[Int] = (0 until continuousFeaturesNumber).toList
      indices.foreach { index =>
        val featureVector: DenseVector[Double] = normalizedContinuous(::, index)
        standardDeviations(index) = stddev(featureVector)
      }
      median(standardDeviations)
    }

    private val distances: DenseMatrix[Double] = distanceMatrix(normalizedContinuous, categorical)

    //TODO: move somewhere else
    def cart[T](listOfLists: List[List[T]]): List[List[T]] = listOfLists match {
      case Nil => List(List())
      case xs :: xss => for (y <- xs; ys <- cart(xss)) yield y :: ys
    }

    //TODO: receive an RDD[SparkDenseVector] instead to speed the program up. Could be parallelized.
    def distanceMatrix(
                        continuousFeatures: DenseMatrix[Double],
                        categoricalFeatures: DenseMatrix[Double]
                      ): DenseMatrix[Double] = {

      val distanceMatrix = DenseMatrix.zeros[Double](observationsNumber, observationsNumber)

      //generate the index I want to consider
      val indices: List[(Int, Int)] = cart(List(observationsIndices, observationsIndices))
        .map(list => (list.head, list.tail.head))
        .filter { case (row, column) => row > column }

      indices.foreach {
        case (row, column) =>
          val continuousFeatureVector1 = continuousFeatures(row, ::).t
          val continuousFeatureVector2 = continuousFeatures(column, ::).t
          val categoricalFeatureVector1 = categoricalFeatures(row, ::).t
          val categoricalFeatureVector2 = categoricalFeatures(column, ::).t

          val distance: Double = euclideanDistance(continuousFeatureVector1, continuousFeatureVector2) +
            (categoricalFeatureVector1 :== categoricalFeatureVector2).activeSize.toDouble *
              Math.pow(standardDeviationMedian, 2)

          //diagonal of distance matrix is always zero (d(x,x) = 0)
          distanceMatrix(row, column) = distance
          distanceMatrix(column, row) = distance
      }
      distanceMatrix
    }


    //TODO: use the misclassified information previously gathered.
    def neighbors(neighborsNumber: Int): Array[Neighbors] = {
      observationsIndices
        .map({
          index =>
            val observationDistances: Array[(Double, Int)] = distances(index, ::).t.toArray.zipWithIndex
            //Remove the element in the index
            val nearestNeighbors = (observationDistances.take(index) ++ observationDistances.drop(index + 1))
              .sorted
              .take(neighborsNumber)
              .map(_._2)
            Neighbors(index, nearestNeighbors)
        })
        .toArray
    }
  }

  /**
    * For a given observation and its set of neighbors:
    * IF CONTINUOUS:
    * 1: choose one of the neighbors randomly.
    * 2: for each feature compute:
    * dif = Double value of the neighbor - Double value of the observation
    * gap = random number between 0 and 1
    * synthetic = Double value of the neighbor + gap * dif
    *
    * IF CATEGORICAL: get the majority value in the neighbors (randomly picked if there is a tie)
    */

  class SMOTE(continuousFeatures: DenseMatrix[Double],
              categoricalFeatures: DenseMatrix[Double],
              neighbors: Array[Neighbors],
              oversamplingRate: Int,
              minMaxScaling: Map[Int, ColumnStats],
              integerFeatures: Seq[Boolean],
              misclassifiedObservations: Seq[Boolean],
              mode: String
             ) {

    val oversamplingSteps: Array[Int] = (1 to oversamplingRate).toArray
    val syntheticSamples: Array[DenseVector[Double]] = oversamplingSteps.flatMap(
      _ => neighbors.map(generateSyntheticSample))

    def generateSyntheticSample(neighbors: Neighbors): DenseVector[Double] = {


      val continuousObservation: DenseVector[Double] = continuousFeatures(neighbors.observationIndex, ::).t

      val misclassifiedNeighbors = neighbors.neighbors.filter(index => misclassifiedObservations(index))
      val neighborsIndices: Array[Int] = mode match {
        case "normal" => neighbors.neighbors
        case "hard_bias" => if (misclassifiedNeighbors.isEmpty) neighbors.neighbors else misclassifiedNeighbors
        case "soft_bias" => neighbors.neighbors ++ misclassifiedNeighbors
      }

      val continuousSelectedNeighbor: DenseVector[Double] = {
        val neighborIndex: Int = neighborsIndices(Random.nextInt(neighborsIndices.length))
        continuousFeatures(neighborIndex, ::).t
      }

      val categoricalNeighbors: Array[DenseVector[Double]] = neighborsIndices.map(index => categoricalFeatures(index, ::).t)

      val continuousSyntheticSample: DenseVector[Double] = generateContinuousSyntheticSample(
        continuousObservation,
        continuousSelectedNeighbor)

      val categoricalSyntheticSample: DenseVector[Double] = generateCategoricalSyntheticSample(categoricalNeighbors)

      val syntheticSample: DenseVector[Double] = DenseVector.vertcat(
        continuousSyntheticSample,
        categoricalSyntheticSample)

      syntheticSample
    }

    def generateContinuousSyntheticSample(observation: DenseVector[Double],
                                          neighbor: DenseVector[Double]): DenseVector[Double] = {
      val gap: Double = Random.nextDouble
      val difference: DenseVector[Double] = neighbor - observation
      //TODO: if integer (see in Stats?) then round the result
      val continuousSyntheticSample: DenseVector[Double] = observation + gap * difference
      denormalize(continuousSyntheticSample)
    }

    //TODO: neighbors number not used!!!???
    def generateCategoricalSyntheticSample(neighbors: Array[DenseVector[Double]]): DenseVector[Double] = {
      val neighborsNumber: Int = neighbors.length
      val neighborsMatrix: DenseMatrix[Double] = DenseMatrix(neighbors: _*)
      val indices: Array[Int] = (0 until neighborsMatrix.cols).toArray

      val syntheticValues: Array[Double] = indices
        .map(index => neighborsMatrix(::, index).toArray)
        .map(neirestNeighbors)
        .map(valuesCandidates => valuesCandidates(Random.nextInt(valuesCandidates.length)))

      val categoricalSyntheticSample: DenseVector[Double] = DenseVector(syntheticValues)
      categoricalSyntheticSample
    }

    //TODO: move elsewhere
    def neirestNeighbors(numbers: Array[Double]): Array[Double] = {
      val numbersCounter: Map[Double, Int] = numbers.groupBy(identity).mapValues(_.length)
      val counter: Map[Int, Iterable[Double]] = numbersCounter.groupBy(_._2).mapValues(_.keys)
      counter.max._2.toArray
    }

    //TODO: not very efficient (repeating this operation several times)
    //TODO: remove this foreach (change to map)

    //Preserves max and min for each feature (if it goes beyond this limits then the value is min or max)
    def denormalize(continuousSyntheticSample: DenseVector[Double]): DenseVector[Double] = {
      val denormalizedContinuousSyntheticSample: DenseVector[Double] =
        DenseVector.zeros[Double](continuousSyntheticSample.length)

      // min-max scaling back: x = x' * (max - min) + min
      minMaxScaling.foreach {
        case (index: Int, stats: ColumnStats) =>
          val denormalizedValue: Double = continuousSyntheticSample(index) * (stats.max - stats.min) + stats.min
          denormalizedContinuousSyntheticSample(index) = denormalizedValue match {
            case aboveMax if aboveMax > stats.max => stats.max
            case belowMin if belowMin < stats.min => stats.min
            case _ => denormalizedValue
          }
      }

      val denormalizedWithIntegersContinuousSyntheticSample = denormalizedContinuousSyntheticSample
        .iterator
        .map {
          case (index, integerValue) if integerFeatures(index) => Math.round(integerValue).toDouble
          case (_, doubleValue) => doubleValue
        }
        .toArray

      DenseVector(denormalizedWithIntegersContinuousSyntheticSample)
    }

  }

  case class FeatureKey(featureIndex: Int, continuousIndex: Int, categoricalIndex: Int)

  case class Neighbors(observationIndex: Int, neighbors: Array[Int])

  case class ColumnStats(min: Double, max: Double)

}