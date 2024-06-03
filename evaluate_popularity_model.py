# spark-submit --deploy-mode client evaluate_popularity_model.py
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, lit, when
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)


def main(spark, userID):
    file_schema = StructType(
        [
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            StructField("tag", StringType(), True),
        ]
    )
    validation = spark.read.csv(
        f"hdfs:/user/{userID}/validation_ratings_new.csv",
        header=True,
        schema=file_schema,
    )
    print("Validation data:")
    validation.printSchema()
    validation.show(10)
    print("---------------------------------")

    testing = spark.read.csv(
        f"hdfs:/user/{userID}/test_ratings_new.csv",
        header=True,
        schema=file_schema,
    )
    print("Testing data:")
    testing.printSchema()
    testing.show(10)
    print("---------------------------------")

    popular_movies = spark.read.csv(
        f"hdfs:/user/{userID}/popular_movies.csv", header=True, inferSchema=True
    )
    print("Popular movies:")
    popular_movies.printSchema()
    popular_movies.show(10)
    print("---------------------------------")

    # take popular_movies and convert it into a list of movie IDs
    popular_movie_ids = [row.movieId for row in popular_movies.collect()]

    def set_high_rating(df, threshold=3):
        return df.withColumn(
            "high_rating", when(col("rating") >= threshold, 1).otherwise(0)
        )

    # set a threshold that if a rating is 3.0 or higher, it is considered as a high rating (1), otherwise 0.
    validation = set_high_rating(validation)
    testing = set_high_rating(testing)

    def precision_at_k(recommended_movies, actual_movies, k):
        if k > len(recommended_movies):
            return 0.0
        relevant_items = set(actual_movies)
        recommended_k = recommended_movies[:k]
        relevant_and_recommended = [
            1 if movie in relevant_items else 0 for movie in recommended_k
        ]
        return sum(relevant_and_recommended) / k

    def map_at_k_for_user(user_row, k):
        actual_movies = user_row["movieId_list"]
        return precision_at_k(popular_movie_ids, actual_movies, k)

    def compute_map_at_k(data, k):
        # collect actual movies watched by each user (ground truth)
        user_actual_movies = (
            data.filter(col("high_rating") == 1)
            .groupBy("userId")
            .agg(F.collect_list("movieId").alias("movieId_list"))
        )

        # compute precision@k for each user
        user_precision_at_k = user_actual_movies.rdd.map(
            lambda row: map_at_k_for_user(row, k)
        ).collect()

        return sum(user_precision_at_k) / len(user_precision_at_k)

    # prepare a list to store MAP@k values
    map_at_k_values = []

    # compute MAP@k for k from 1 to 30 for the validation dataset
    for k in range(1, 31):
        map_at_k = compute_map_at_k(validation, k)
        map_at_k_values.append((k, map_at_k))
        print(f"MAP@{k}: {map_at_k}")

    # best k based on the highest MAP@k
    best_k = max(map_at_k_values, key=lambda x: x[1])[0]
    print(f"Best k: {best_k}")

    # evaluate the model on the testing dataset using the best k
    map_at_best_k_on_test = compute_map_at_k(testing, best_k)
    print(f"MAP@{best_k} on test dataset: {map_at_best_k_on_test}")
    spark.stop()


if __name__ == "__main__":
    spark = (
        SparkSession.builder.appName("Popularity Based Model Evaluation")
        .config("spark.executor.memory", "6g")
        .config("spark.executor.memoryOverhead", "2g")
        .config("spark.driver.memory", "4g")
        .config("spark.shutdown.hook.timeout", "1h")
        .getOrCreate()
    )

    userID = os.environ["USER"]
    main(spark, userID)
