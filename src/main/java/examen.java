import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import java.io.*;
import java.util.*;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import static org.apache.spark.sql.functions.*;


public class examen {
    public static void main(String[] args) {
        //Initialisation du SparkContext
        JavaSparkContext sparkContext = new JavaSparkContext("local[*]", "RiverCount");

        //Chargement des données depuis un fichier
        JavaRDD<String> lines = sparkContext.textFile("/home/ubuntu/evaluation_java_Spark/housing.data");
               // comptez le nombre de villes près du fleuve et celles n'étant pas près du fleuve, 
        JavaRDD< String[]> RDD = lines.map(line -> line.substring(1).split("\\s+"));
        long near_river = RDD.filter(line ->line[3].equals("0")).count();
        long no_near_river = RDD.filter(line ->line[3].equals("0")).count() ; 
            
        //Compter le nombre d'occurrences le nombre moyen de pièces par logement
    JavaPairRDD<Double, Integer> piecesCounts = lines.mapToPair(line -> {
        String[] parts = line.substring(1).split("\\s+");
        Double pieces = Double.parseDouble(parts[5]);
        return new Tuple2<>(pieces, 1);
    }).reduceByKey((count1, count2) -> count1 + count2);

    JavaPairRDD<Integer, Double> countsByPieces = piecesCounts.mapToPair(tuple -> {
        Integer count = tuple._2();
        Double pieces = tuple._1();
        return new Tuple2<>(count, pieces);
    }).reduceByKey((pieces1, pieces2) -> pieces1 + pieces2);

    JavaPairRDD<Double, Double> avgPiecesByCount = countsByPieces.mapToPair(tuple -> {
        Integer count = tuple._1();
        Double pieces = tuple._2();
        Double avgPieces = pieces / count;
        return new Tuple2<>(avgPieces, pieces);
    });

    avgPiecesByCount.foreach(tuple -> {
        System.out.println("Nombre d'occurrences : " + tuple._2() + ", nombre moyen de pièces par logement : " + tuple._1());
    });
     // affichez les différentes modalités de la variable RAD
    JavaRDD<String> rad = lines.map(line -> line.split("\\s+")[8]).distinct();
        rad.foreach(value -> System.out.println(value));   

        
    

        sparkContext.stop();
        
    }
}
