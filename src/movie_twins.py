import pandas as pd
import numpy as np
import os
import sys
import pickle
#from datasketch import MinHash, MinHashLSH
from pyspark.sql.functions import collect_set, col
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

def main(spark, userID, input_folder):
    ratings = spark.read.csv(f'hdfs:/user/{userID}/target/{input_folder}/ratings.csv', header=True, inferSchema=True).drop('rating', 'timestamp')

    movies_per_user = ratings.groupBy("userId").agg(collect_set("movieId").alias("movies"))
    int_to_str_udf = udf(lambda movies: [str(movie) for movie in movies], ArrayType(StringType()))
    movies_per_user = movies_per_user.withColumn("movies", int_to_str_udf("movies"))
    cv = CountVectorizer(inputCol="movies", outputCol="features", binary=True)
    model = cv.fit(movies_per_user)
    result = model.transform(movies_per_user)

    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    mh_model = mh.fit(result)
    similar_users = mh_model.approxSimilarityJoin(result, result, threshold=1.0, distCol="JaccardDistance") \
        .filter("datasetA.userId > datasetB.userId") \
        .orderBy("JaccardDistance", ascending=True) \
        .selectExpr("datasetA.userId as userIdA", "datasetB.userId as userIdB", "JaccardDistance") \
        .limit(100)

    similar_users.show()
    pandas_df = similar_users.select("*").toPandas()
    output_path = f"top_100_{input_folder}.csv"
    pandas_df.to_csv(output_path)
    print(f"Output CSV file saved at top_100_{input_folder}.csv!")

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("testing") \
        .config("spark.core.connection.auth.wait.timeout", "60000") \
        .config("spark.scheduler.maxRegisteredResourcesWaitingTime", "60000") \
        .config("spark.hadoop.service.shutdown.timeout", "60000") \
        .config("spark.hadoop.mapreduce.task.timeout", "60000") \
        .config("hadoop.service.shutdown.timeout", "60000") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .config("spark.executor.memory", "24g") \
        .config("spark.executor.memoryOverhead", "8g") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

    userID = os.environ['USER']
    input_folder = sys.argv[1]
    main(spark, userID, input_folder)
