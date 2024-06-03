import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import expr, col, rank

def main(spark, userID):
    best_model = ALSModel.load(f"hdfs:/user/{userID}/best_als_model")
    test = spark.read.csv(f"hdfs:/user/{userID}/test_ratings_new.csv", header=True).drop('tag') \
        .withColumn("userId", col("userId").cast("int")) \
        .withColumn("movieId", col("movieId").cast("int")) \
        .withColumn("rating", col("rating").cast("int"))
    
    userRecs = best_model.recommendForAllUsers(30)
    test_interactions = test.groupBy("userId").agg(expr("collect_list(movieId) as items"))
    predictions_and_labels = userRecs.join(test_interactions, "userId")
    predictions_and_labels = predictions_and_labels.select("recommendations.movieId", "items")
    predictions_and_labels_rdd = predictions_and_labels.rdd.map(lambda row: (row[0], row[1]))
    metrics = RankingMetrics(predictions_and_labels_rdd)
    map_scores = {k: metrics.precisionAt(k) for k in range(1, 31)}

    for k, score in map_scores.items():
        print(f"MAP@{k}: {score}")

if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("Ratings Data Partitioning")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "4g")
        .config("spark.shutdown.hook.timeout", "1h")
        .getOrCreate()
    )

    userID = os.environ["USER"]
    main(spark, userID)
