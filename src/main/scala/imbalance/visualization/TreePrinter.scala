package imbalance.visualization

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.tree._

//TODO: make test class
//TODO: split this class into several files to get everything organized

case class DecisionNode(
                         featureName: Option[String],
                         gain: Option[Double],
                         impurity: Double,
                         threshold: Option[Double], // Continuous split
                         nodeType: String, // Internal or leaf
                         splitType: Option[String], // Continuous and categorical
                         leftCategories: Option[Array[Double]], // Categorical Split
                         rightCategories: Option[Array[Double]], // Categorical Split
                         prediction: String,
                         leftChild: Option[DecisionNode],
                         rightChild: Option[DecisionNode])


class Labels(labels: Array[String]) {

  /**
    *
    * @param labels : list of string labels
    * @return map with the labels as values and integers enumerating them as keys (starting from zero)
    */
  private def enumerate(labels: Array[String]): Map[Int, String] = labels.zipWithIndex
    .map({ case (name, index) => (index, name) })
    .toMap

  def get(key: Int): String = enumerate(labels)(key)
}

//TODO: add the labels for each indexer to get the categorical categories

class TreePrinter(tree: DecisionTreeClassificationModel,
                  labelNames: Labels,
                  featureNames: Labels,
                  categories: Map[Int, Labels]) {

  private val getInternalNode: PartialFunction[Node, Option[InternalNode]] = {
    case node: InternalNode => Some(node.asInstanceOf[InternalNode])
    case _ => None
  }

  private val getContinuousSplit: PartialFunction[Option[Split], Option[ContinuousSplit]] = {
    case Some(continuousSplit: ContinuousSplit) => Some(continuousSplit.asInstanceOf[ContinuousSplit])
    case _ => None
  }

  private val getCategoricalSplit: PartialFunction[Option[Split], Option[CategoricalSplit]] = {
    case Some(categoricalSplit: CategoricalSplit) => Some(categoricalSplit.asInstanceOf[CategoricalSplit])
    case _ => None
  }

  private var DOT: Array[String] = Array()
  private var idCount: Int = 0
  //  private var ids: Map[Int, String] = Map()

  def getDecisionRules(node: Node): DecisionNode = {
    val internalNode: Option[InternalNode] = getInternalNode(node)
    val continuousSplit: Option[ContinuousSplit] = getContinuousSplit(internalNode.map(_.split))
    val categoricalSplit: Option[CategoricalSplit] = getCategoricalSplit(internalNode.map(_.split))

    val (nodeType: String, splitType: Option[String]) = internalNode match {
      case None => ("leaf", None)
      case Some(_) if continuousSplit.isDefined => ("internal", Some("continuous"))
      case _ => ("internal", Some("categorical"))
    }

    def getIndex(internalNode: Option[InternalNode]): Option[Int] = internalNode.map(_.split.featureIndex)

    val prediction: String = labelNames.get(node.prediction.toInt)
    val gain: Option[Double] = internalNode.map(_.gain)
    val impurity: Double = node.impurity

    val featureIndex: Option[Int] = getIndex(internalNode)
    val featureName: Option[String] = featureIndex.map(featureNames.get)

    val nodeThreshold: Option[Double] = continuousSplit.map(_.threshold)
    val categoricalLabels: Option[Labels] = featureIndex.flatMap(categories.get)

    val leftCategoriesIndexes: Option[Array[Double]] = categoricalSplit.map(_.leftCategories)
    val rightCategoriesIndexes: Option[Array[Double]] = categoricalSplit.map(_.rightCategories)

    //TODO: Fix this. Array issue. FIX leftCategories and rightCategories!
    //    def categoryName(index : Double) : String = categoricalLabels.get.get(index.toInt)
    //    def categoriesNames : PartialFunction[Option[Array[Double]], Option[Array[String]]] = {
    //      case Some(indexes) => Some(indexes.map(index => categoryName(index)))
    //      case None => None
    //    }
    //    val leftCategories: Option[Array[String]] = categoriesNames(leftCategoriesIndexes)
    //    val rightCategories: Option[Array[String]] = categoriesNames(rightCategoriesIndexes)

    val leftChild: Option[DecisionNode] = internalNode.map(_.leftChild) match {
      case Some(child) => Some(getDecisionRules(child))
      case _ => None
    }
    val rightChild: Option[DecisionNode] = internalNode.map(_.rightChild) match {
      case Some(child) => Some(getDecisionRules(child))
      case _ => None
    }

    val labelName: String = idCount.toString
    idCount += 1

    val labelExplanation: String = featureName match {
      case Some(label) => label + "<=" +
        nodeThreshold.getOrElse(leftCategoriesIndexes.grouped(5).toList.map(x => x.mkString(",")).mkString("\n"))
      case None => ""
    }
    //TODO: fill in
    //    val leftRule: String = "" + "->" + "" + "[labeldistance=2.5, labelangle=45, headlabel=True]"
    //    val rightRule: String = "" + "->" + "" + "[labeldistance=2.5, labelangle=-45, headlabel=False]"
    val leftRule: String = ""
    val rightRule: String = ""

    val impurityLabel: String = "Impurity = " + "%.2f".format(impurity)
    val gainLabel: String = gain.map(x => "Gain = %.2f".format(x)).getOrElse("")

    DOT :+=
      s"""
         |$labelName
         |[
         |label="
         |Node ID $labelName
         |$labelExplanation
         |$impurityLabel
         |$gainLabel
         |Prediction = $prediction
         |"
         |]
         |$leftRule
         |$rightRule
         |"""
        .stripMargin

    DecisionNode(
      featureName = featureName,
      gain = gain,
      impurity = impurity,
      threshold = nodeThreshold,
      nodeType = nodeType,
      splitType = splitType,
      leftCategories = leftCategoriesIndexes, // Categorical Split
      rightCategories = rightCategoriesIndexes, // Categorical Split
      prediction = prediction,
      leftChild = leftChild,
      rightChild = rightChild
    )
  }

  //  def toJsonPlotFormat: String = {
  //    implicit val formats = net.liftweb.json.DefaultFormats
  //    net.liftweb.json.Serialization.writePretty(getDecisionRules(tree.rootNode))
  //  }


  def toJson: String = {
    val mapper = new ObjectMapper() with ScalaObjectMapper
    mapper.registerModule(DefaultScalaModule)
    mapper.writeValueAsString(getDecisionRules(tree.rootNode))
  }

  //TODO: this doesn't work yet
  def toDOT: String = {
    DOT = Array()
    getDecisionRules(tree.rootNode)
    val subgraph: String = DOT.reverse.mkString("\n")
    s"""digraph Tree{
       |node [shape=box style=filled]
       |subgraph body {
       |$subgraph
       |}
       |}
    """.stripMargin
  }

}