# here is how you can submit this file: spark-submit --deploy-mode client data_partition_full.py ml-latest

import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    row_number,
    when,
    collect_list,
    avg,
    expr,
    desc,
    split,
)
from pyspark.sql.window import Window
from pyspark.ml.feature import CountVectorizer
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
            StructField("timestamp_tag", IntegerType(), True),
        ]
    )

    movies_file_struct = StructType(
        [
            StructField("movieId", IntegerType(), True),
            StructField("title", StringType(), True),
            StructField("genres", StringType(), True),
        ]
    )

    genome_scores_file_struct = StructType(
        [
            StructField("movieId", IntegerType(), True),
            StructField("tagId", IntegerType(), True),
            StructField("relevance", DoubleType(), True),
        ]
    )

    genome_tags_file_struct = StructType(
        [
            StructField("tagId", IntegerType(), True),
            StructField("tag", StringType(), True),
        ]
    )

    ratings = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/ratings.csv",
        header=True,
        schema=ratings_file_struct,
    )
    print("Ratings Schema:")
    ratings = ratings.dropna(how="any")
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
    tags = tags.dropna(how="any")
    tags.dropDuplicates()
    tags.createOrReplaceTempView("tags")
    tags.printSchema()
    print("---------------------------------")

    movies = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/movies.csv",
        header=True,
        schema=movies_file_struct,
    )
    print("Movies Schema:")
    movies = movies.dropna(how="any")
    movies.dropDuplicates()
    movies.createOrReplaceTempView("movies")
    movies.printSchema()
    print("---------------------------------")

    genome_scores = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/genome-scores.csv",
        header=True,
        schema=genome_scores_file_struct,
    )
    print("Genome Scores Schema:")
    genome_scores = genome_scores.dropna(how="any")
    genome_scores.dropDuplicates()
    genome_scores.createOrReplaceTempView("genome_scores")
    genome_scores.printSchema()
    print("---------------------------------")

    genome_tags = spark.read.csv(
        f"hdfs:/user/{userID}/target/{input_folder}/genome-tags.csv",
        header=True,
        schema=genome_tags_file_struct,
    )
    print("Genome Tags Schema:")
    genome_tags = genome_tags.dropna(how="any")
    genome_tags.dropDuplicates()
    genome_tags.createOrReplaceTempView("genome_tags")
    genome_tags.printSchema()
    print("---------------------------------")

    genome_scores_with_tags = genome_scores.join(genome_tags, "tagId").select(
        "movieId", "tag", "relevance"
    )
    genome_scores_aggregated = genome_scores_with_tags.groupBy("movieId").agg(
        collect_list("tag").alias("tags"),
        expr("percentile_approx(relevance, 0.5)").alias("median_relevance"),
    )
    # inner join to include only movies with genome scores
    movies_with_genome = movies.join(genome_scores_aggregated, "movieId", "inner")
    # split the genres string into an array of genres
    movies_with_genome = movies_with_genome.withColumn(
        "genres_array", split(col("genres"), "\\|")
    )
    # extracting genre features
    cv = CountVectorizer(inputCol="genres_array", outputCol="genreFeatures")
    model = cv.fit(movies_with_genome)
    movies = model.transform(movies_with_genome)

    # show the first 5 rows of the movies data
    print("Movies Data:")
    movies.show(5)
    print("---------------------------------")

    # partition the ratings data by userId and timestamp
    ratings = ratings.withColumn(
        "ratings_rank",
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
        "tags_rank",
        row_number().over(Window.partitionBy("userId").orderBy("timestamp_tag")),
    )
    total_tags = tags.groupBy("userId").count().withColumnRenamed("count", "total_tags")
    tags = tags.join(total_tags, on="userId")

    # show the first 5 rows of the ratings data
    print("Tags Data:")
    tags.show(5)
    print("---------------------------------")

    print(
        "Splitting into train, validation, and test sets based on the `ratings_rank`..."
    )
    ratings = ratings.withColumn(
        "split",
        when(col("ratings_rank") <= col("total_ratings") * 0.6, "train")
        .when(col("ratings_rank") <= col("total_ratings") * 0.8, "validation")
        .otherwise("test"),
    )
    ratings = ratings.join(tags, ["userId", "movieId"], "inner")

    movies.join(ratings, "movieId").groupBy("split").agg(
        expr("percentile_approx(median_relevance, 0.5)").alias("median_relevance")
    ).dropDuplicates()

    print("Movies dataset after modifications: ")
    ratings.show(5)
    print("---------------------------------")
    print(f"Length of ENTIRE ratings dataset: {ratings.count()}")

    # drop the rank and total_ratings columns and keep 3 different datasets

    # movieId, userId, rating, timestamp_rating, ratings_rank,
    # total_ratings, split, tag, timestamp_tag, tags_rank, total_tags,
    # title, genres, tags, median_relevance, genres_array, genreFeatures

    train = (
        ratings.filter(col("split") == "train")
        .join(movies, "movieId", "inner")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    train.dropDuplicates()

    validation = (
        ratings.filter(col("split") == "validation")
        .join(movies, "movieId", "inner")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    validation.dropDuplicates()

    test = (
        ratings.filter(col("split") == "test")
        .join(movies, "movieId", "inner")
        .select("userId", "movieId", "rating", "tag")
        .dropDuplicates(["userId", "movieId", "rating"])
    )
    test.dropDuplicates()

    # show the first 5 rows of the train, validation, and test sets
    print("Train Set:")
    print(f"Length of train set: {train.count()}")
    train.show(20)
    print("---------------------------------")

    print("Validation Set:")
    print(f"Length of validation set: {validation.count()}")
    validation.show(20)
    print("---------------------------------")

    print("Test Set:")
    print(f"Length of test set: {test.count()}")
    test.show(20)
    print("---------------------------------")

    # saving into csv files
    print("Saving into csv files...")
    train.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/train_ratings.csv",
        "train_ratings_new.csv",
        header=True,
        mode="overwrite",
    )
    validation.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/validation_ratings.csv",
        "validation_ratings_new.csv",
        header=True,
        mode="overwrite",
    )
    test.coalesce(1).write.csv(
        # f"hdfs:/user/{userID}/target/{input_folder}/test_ratings.csv",
        "test_ratings_new.csv",
        header=True,
        mode="overwrite",
    )
    print("All files are saved successfully!")
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
