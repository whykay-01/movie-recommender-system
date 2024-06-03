import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number, when
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)


def main(spark, userID, input_folder):
    # statically define the schema for the ratings file
    ratings_file_struct = StructType(
        [
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("rating", DoubleType(), True),
            StructField("timestamp_rating", IntegerType(), True),
        ]
    )

    tags_file_struct = StructType(
        [
            StructField("userId", IntegerType(), True),
            StructField("movieId", IntegerType(), True),
            StructField("tag", StringType(), True),
            StructField("timestamp_rating", IntegerType(), True),
        ]
    )

    ratings = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/ratings.csv",
        header=True,
        schema=ratings_file_struct,
    )
    print("Ratings Schema:")
    ratings.dropDuplicates()
    ratings.createOrReplaceTempView("ratings")
    ratings.printSchema()
    print("---------------------------------")

    tags = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/tags.csv",
        header=True,
        schema=tags_file_struct,
    )
    print("Tags Schema:")
    tags.dropDuplicates()
    tags.createOrReplaceTempView("tags")
    tags.printSchema()
    print("---------------------------------")

    # partition the ratings data by userId and timestamp
    ratings = ratings.withColumn(
        "rank",
        row_number().over(Window.partitionBy("userId").orderBy("timestamp_rating")),
    )

    # compute total number of ratings per user for partitioning
    total_ratings = (
        ratings.groupBy("userId").count().withColumnRenamed("count", "total_ratings")
    )
    ratings = ratings.join(total_ratings, on="userId")

    # show the first 5 rows of the ratings data
    print("Ratings Data:")
    ratings.show(5)
    print("---------------------------------")

    # prepare tags data
    tags = tags.withColumn(
        "rank",
        row_number().over(Window.partitionBy("userId").orderBy("timestamp_rating")),
    )
    total_tags = tags.groupBy("userId").count().withColumnRenamed("count", "total_tags")
    tags = tags.join(total_tags, on="userId")

    # show the first 5 rows of the ratings data
    print("Tags Data:")
    tags.show(5)
    print("---------------------------------")

    print("Splitting into train, validation, and test sets...")
    ratings = ratings.withColumn(
        "split",
        when(col("rank") <= col("total_ratings") * 0.6, "train")
        .when(col("rank") <= col("total_ratings") * 0.8, "validation")
        .otherwise("test"),
    )
    print(f"Length of ENTIRE dataset: {ratings.count()}")

    # drop the rank and total_ratings columns and keep 3 different datasets
    train = (
        ratings.filter(col("split") == "train")
        .join(tags, ["userId", "movieId"], "left")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    train.dropDuplicates()

    validation = (
        ratings.filter(col("split") == "validation")
        .join(tags, ["userId", "movieId"], "left")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    validation.dropDuplicates()

    test = (
        ratings.filter(col("split") == "test")
        .join(tags, ["userId", "movieId"], "left")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    test.dropDuplicates()

    # show the first 5 rows of the train, validation, and test sets
    print("Train Set:")
    print(f"Length of train set: {train.count()}")
    train.show(5)
    print("---------------------------------")

    print("Validation Set:")
    print(f"Length of validation set: {validation.count()}")
    validation.show(5)
    print("---------------------------------")

    print("Test Set:")
    print(f"Length of test set: {test.count()}")
    test.show(5)
    print("---------------------------------")

    # saving into csv files
    print("Saving into csv files...")
    train.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/train_ratings.csv",
        "train_ratings.csv",
        header=True,
        mode="overwrite",
    )
    validation.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/validation_ratings.csv",
        "validation_ratings.csv",
        header=True,
        mode="overwrite",
    )
    test.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/test_ratings.csv",
        "test_ratings.csv",
        header=True,
        mode="overwrite",
    )
    print("All files are ssaved successfully!")
    spark.stop()


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
    input_folder = sys.argv[1]
    main(spark, userID, input_folder)
