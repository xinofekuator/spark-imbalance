name := "spark-imbalance"

version := "0.1-SNAPSHOT"

scalaVersion := "2.11.8"

// "provided" when deploying, "compile" when developing
val sparkDependencyScope = "provided"

// spark version to be used
val sparkVersion = "2.1.0"

// force scalaVersion
ivyScala := ivyScala.value map { _.copy(overrideScalaVersion = true) }

// Needed as SBT's classloader doesn't work well with Spark
fork := true

// BUG: unfortunately, it's not supported right now
fork in console := true

// Java version
javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

// add a JVM option to use when forking a JVM for 'run'
javaOptions ++= Seq("-Xmx512m")

// append -deprecation to the options passed to the Scala compiler
scalacOptions ++= Seq("-deprecation", "-unchecked")

// spark modules (should be included by spark-sql, just an example)
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % sparkDependencyScope,
  "org.apache.spark" %% "spark-sql" % sparkVersion % sparkDependencyScope,
  "org.apache.spark" %% "spark-mllib" % sparkVersion % sparkDependencyScope,
  "org.apache.spark" %% "spark-hive" % sparkVersion % sparkDependencyScope
)


//plotting
//libraryDependencies += "co.theasi" %% "plotly" % "0.2.0"
//libraryDependencies += "org.vegas-viz" %% "vegas" % "0.3.9"

// testing
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

libraryDependencies += "org.scalacheck" %% "scalacheck" % "1.12.2" % "test"


/// Compiler plugins

// linter: static analysis for scala
addCompilerPlugin("org.psywerx.hairyfotr" %% "linter" % "0.1.17")

/// console

// define the statements initially evaluated when entering 'console', 'consoleQuick', or 'consoleProject'
// but still keep the console settings in the sbt-spark-package plugin

// If you want to use yarn-client for spark cluster mode, override the environment variable
// SPARK_MODE=yarn-client <cmd>
val sparkMode = sys.env.getOrElse("SPARK_MODE", "local[2]")


initialCommands in console :=
  s"""
     |import org.apache.spark.sql.SparkSession
     |import org.apache.spark.SparkConf
     |
    |@transient val spark = SparkSession
     |    .builder()
     |    .appName("Console Test")
     |    .master("$sparkMode")
     |    .enableHiveSupport()
     |    .getOrCreate()
     |import spark.implicits._
     |
    |def time[T](f: => T): T = {
     |  import System.{currentTimeMillis => now}
     |  val start = now
     |  try { f } finally { println("Elapsed: " + (now - start)/1000.0 + " s") }
     |}
     |
    |""".stripMargin

cleanupCommands in console :=
  s"""
     |spark.stop()
   """.stripMargin


/// scaladoc
scalacOptions in (Compile,doc) ++= Seq("-groups", "-implicits",
  // NOTE: remember to change the JVM path that works on your system.
  "-doc-external-doc:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/rt.jar#http://docs.oracle.com/javase/8/docs/api"
)

autoAPIMappings := true