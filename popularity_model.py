import sys, os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(spark, userID):
    train = spark.read.csv(f"hdfs:/user/{userID}/train_ratings_new.csv", header=True).drop('tag')
    #train.show(10)
    
    grouped = train.groupby('movieId').agg(
                    F.avg("rating").alias("average_rating"),
                    F.count("rating").alias("count_rating")
                ).filter(
                    F.col("count_rating") >= 100
                ).orderBy(
                    F.col("average_rating").desc()
                ).limit(30)
    grouped.show()

    # saving into csv files
    print("Saving into csv files...")
    grouped.coalesce(1).write.csv(
        "popularity_rec.csv",
        header=True,
        mode="overwrite",
    )

    print('Files saved!')

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







