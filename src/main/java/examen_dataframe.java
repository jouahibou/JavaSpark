import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.api.java.JavaRDD;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.PipelineModel;


public class examen_dataframe {
    
    public static void main(String[] args) {

        // création de la configuration de Spark
         SparkConf conf = new SparkConf()
                 .setAppName("SparkSession")
                 .setMaster("local[*]");
        
//         // création de la Spark Session
         SparkSession spark = SparkSession.builder()
                 .config(conf)
                 .getOrCreate();

         // accéder au Spark Context associé
         SparkContext sc_raw = spark.sparkContext();
         JavaSparkContext sc = JavaSparkContext.fromSparkContext(sc_raw);

         JavaRDD<String> lines = sc.textFile("/home/ubuntu/evaluation_java_Spark/housing.data");
               
         JavaRDD< String[]> RDD = lines.map(line -> line.substring(1).split("\\s+"));
        
       
 

// // Convertir le JavaRDD<String[]> en JavaRDD<Row>
 JavaRDD<Row> rows = RDD.map(line -> RowFactory.create(line));

// // Créer le schéma spécifiant le nom des colonnes et leur type
 

  StructType schema = DataTypes.createStructType(new StructField[] {
          DataTypes.createStructField("CRIM", DataTypes.StringType, true),
          DataTypes.createStructField("ZN", DataTypes.StringType, true),
          DataTypes.createStructField("INDUS", DataTypes.StringType, true),
          DataTypes.createStructField("CHAS", DataTypes.StringType, true),
          DataTypes.createStructField("NOX", DataTypes.StringType, true),
          DataTypes.createStructField("RM",DataTypes.StringType, true),
          DataTypes.createStructField("AGE",DataTypes.StringType, true),
          DataTypes.createStructField("DIS", DataTypes.StringType, true),
          DataTypes.createStructField("RAD", DataTypes.StringType, true),
          DataTypes.createStructField("TAX", DataTypes.StringType, true),
          DataTypes.createStructField("PTRATIO", DataTypes.StringType, true),
          DataTypes.createStructField("B_100_bk_0_63)", DataTypes.StringType, true),
          DataTypes.createStructField("LSTAT", DataTypes.StringType, true),
          DataTypes.createStructField("MEDSV", DataTypes.StringType, true)
      });

      //     // Créer le DataFrame à partir du RDD<Row>
 Dataset<Row> df = spark.createDataFrame(rows, schema);

// // Récupérer les noms de colonnes et convertir en double 
  String[] columnNames = schema.fieldNames();
 for (String columnName : columnNames) {
     df = df.withColumn(columnName, col(columnName).cast("Double"));
 }

 //  // Statistiques 
 
//  // Afficher la moyenne de la proportion de zone résidentielles par lot
 double avgZN = df.select("ZN").filter("ZN is not null").groupBy().avg().first().getDouble(0);
  System.out.println("La moyenne de la proportion de zone résidentielles par lot est " + avgZN);
 
//  // Afficher le nombre de villes avec un taux de criminalité supérieur à 0.1
  long countCrim = df.select("CRIM").filter("CRIM > 0.1").count();
System.out.println("Il y a " + countCrim + " villes avec un taux de criminalité supérieur à 0.1");
 
//  // Afficher la moyenne du nombre moyen de pièces par logement pour les villes au bord de la rivière Charles
  double avgRMChas = df.select("RM", "CHAS").filter("CHAS = 1").groupBy("CHAS").avg("RM").first().getDouble(1);
System.out.println("La moyenne du nombre moyen de pièces par logement pour les villes au bord de la rivière Charles est " + avgRMChas);

      // Vectorisation 
   VectorAssembler assembler = new VectorAssembler()
   .setInputCols(new String[] {"CRIM", "ZN","INDUS", "CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B_100_bk_0_63)","LSTAT","MEDSV"})
   .setOutputCol("features");
   Dataset<Row> assembled = assembler.transform(df);

// // Standarisation 
  StandardScaler scaler = new StandardScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(false);
     StandardScalerModel scalerModel = scaler.fit(assembled);
     Dataset<Row> scaled = scalerModel.transform(assembled);


     // //  big_features est considéré comme varibale explicatif 
  VectorAssembler assembler_fin = new VectorAssembler()
  .setInputCols(new String[] {"scaledFeatures", "ZN", "TAX"})
  .setOutputCol("big_features");
Dataset<Row> data = assembler_fin.transform(scaled).select("MEDSV", "big_features");
data = data.withColumnRenamed("MEDSV", "label");
data = data.withColumnRenamed("big_features", "features");

// // Splitting des données en train et test sets
Dataset<Row>[] dataSplit = data.randomSplit(new double[]{0.75, 0.25}, 12345);
Dataset<Row> train = dataSplit[0];
Dataset<Row> test = dataSplit[1];





   

    //  // Créer un objet LinearRegression
    LinearRegression lr = new LinearRegression();
    

    //  // Créer un objet Pipeline
      PipelineStage[] stages = new PipelineStage[] {assembler_fin};
      stages[stages.length-1] = lr;
      Pipeline pipeline = new Pipeline().setStages(stages);

    // //  // Create a ParamGridBuilder with the parameters to be tuned
  
       ParamMap[] paramGrid = new ParamGridBuilder()
       .addGrid(lr.regParam(), new double[] {0.1, 0.01})
       .addGrid(lr.elasticNetParam(), new double[] {0.0, 0.5, 1.0})
       .build();

    // //  // crossvaliation avec evaluator pour regression
      CrossValidator cv = new CrossValidator()
       .setEstimator(pipeline)
       .setEvaluator(new RegressionEvaluator())
       .setEstimatorParamMaps(paramGrid)
       .setNumFolds(5);

    // //  // entrainement  crossvalidatator avec la donnée d'entrainement 
       CrossValidatorModel cvModel = cv.fit(train);

       PipelineModel bestPipelineModel = (PipelineModel) cvModel.bestModel();
       LinearRegressionModel bestModel = (LinearRegressionModel) bestPipelineModel.stages()[0];

    

    // //  // faire la predictions sur les données de test
       Dataset<Row> predictions = bestModel.transform(test);

       

    // //  // Evaluate the predictions using the RegressionEvaluator
       RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
              .setPredictionCol("prediction")
            .setMetricName("rmse");
       double rmse = evaluator.evaluate(predictions);
       System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

    // //  // Stop the SparkSession
       spark.stop();




    }}
    
//           
        











    

