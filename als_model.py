import sys, os
#import pickle
#import dill as pickle
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import expr, col, rank

def main(spark, userID):
    train = spark.read.csv(f"hdfs:/user/{userID}/train_ratings_new.csv", header = True).drop('tag') \
        .withColumn("userId", col("userId").cast("int")) \
        .withColumn("movieId", col("movieId").cast("int")) \
        .withColumn("rating", col("rating").cast("int"))
    validation = spark.read.csv(f"hdfs:/user/{userID}/validation_ratings_new.csv", header = True).drop('tag') \
        .withColumn("userId", col("userId").cast("int")) \
        .withColumn("movieId", col("movieId").cast("int")) \
        .withColumn("rating", col("rating").cast("int"))
    test = spark.read.csv(f"hdfs:/user/{userID}/test_ratings_new.csv", header = True).drop('tag') \
        .withColumn("userId", col("userId").cast("int")) \
        .withColumn("movieId", col("movieId").cast("int")) \
        .withColumn("rating", col("rating").cast("int"))

    als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    param_grid = ParamGridBuilder() \
        .addGrid(als.rank, [10, 20, 50]) \
        .addGrid(als.regParam, [0.01, 0.1, 0.5]) \
        .build()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    cross_val = CrossValidator(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )

    cv_model = cross_val.fit(train)
    best_model = cv_model.bestModel
    val_predictions = best_model.transform(validation)
    test_predictions = best_model.transform(test)

    val_rmse = evaluator.evaluate(val_predictions)
    test_rmse = evaluator.evaluate(test_predictions)
    print()
    print("Root Mean Square Error (RMSE) on validation data:", val_rmse)
    print("Root Mean Square Error (RMSE) on test data:", test_rmse)
    
    filename = 'best_als_model'
    best_model.save(filename)
    print(f"Model has been saved into {filename}.")

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
