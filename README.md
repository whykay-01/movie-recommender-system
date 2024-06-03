## Big Data Capstone Project

# Overview

In the capstone project, I have applied the tools I learned in big data class to solve a realistic, large-scale applied problem.
Specifically, I used the movielens dataset to build and evaluate a **collaborative-filter based recommender as well as a customer segmentation system**. 

In either case, you are encouraged to work in **groups of up to 3 students**:


## The dataset

In this project, we'll use the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset provided by F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

We have prepared two versions of this dataset for you in Dataproc's HDFS: 
A small dataset for prototyping at /user/pw44_nyu_edu/ml-latest-small.zip (9000 movies, 600 users)
The full dataset for scaling up at /user/pw44_nyu_edu/ml-latest.zip (86000 movies and 330000 users)

Each version of the data contains rating and tag interactions, and the larger sample includes "tag genome" data for each movie, which you may consider as additional features beyond
the collaborative filter. Each version of the data includes a README.txt file which explains the contents and structure of the data which are stored in CSV files.
We strongly recommend to thoroughly read through the dataset documentation before beginning, and make note of the documented differences between the smaller and larger datasets.
Knowing these differences in advance will save you many headaches when it comes time to scale up.
Note: In general, use the small dataset for prototyping, but answer the questions below by using the full dataset.
Also note that these files are provided to you as zip files for ease - they both unzip as larger folders with many files. You should copy these files to your local hdfs and unzip them there by commands like these:

hadoop fs -copyToLocal /user/pw44_nyu_edu/ml-latest.zip .

unzip ml-latest.zip

hadoop fs -copyFromLocal ./ml-latest /user/[YOUR_NETID]_nyu_edu/target


## What did I build:

## Customer segmentation

1.  Customer segmentation relies on similarity, so we first want to find the top 100 pairs of users ("movie twins") who have the most similar movie watching style. Note: For the sake of simplicity, we operationalized "movie watching style" simply by the set of movies that was rated, regardless of the actual numerical ratings. We did so with a minHash-based algorithm.
2.  Validated the results from part 1 by checking whether the average correlation of the numerical ratings in the 100 pairs is different from (higher?) than 100 randomly picked pairs of users from the full dataset.

## Movie recommendation

3.  As a first step, I partitioned the ratings data into training, validation, and test sets. This will reduce the complexity of your experiment code down the line, and make it easier to generate alternative splits if you want to assess the stability of your implementation.

4.  Before implementing a sophisticated model, I began with a popularity baseline model. This is simple enough to implement with some basic dataframe computations. Afterwards, I have evaluated my popularity baseline (see below) before moving on to the next step.

5.  THe recommendation model is using Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
    I made sure to thoroughly read through the documentation on the [pyspark.ml.recommendation module](https://spark.apache.org/docs/3.0.1/ml-collaborative-filtering.html) before getting started.
    This model has some hyper-parameters that you should tune to optimize performance on the validation set, notably: 
      - the *rank* (dimension) of the latent factors, and
      - the regularization parameter.

### Evaluation

Once you are able to make predictions—either from the popularity baseline or the latent factor model—you will need to evaluate accuracy on the validation and test data.
Scores for validation and test are reported in the write-up.
Evaluations are based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
I have referred to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for this process.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but choices are reflected in the final report.
As a general rule, I explored ranges of each hyper-parameter that are sufficiently large to produce observable differences in the evaluation score.


## This codebase cannot be run on the local machine as it required an HPC cluster for these goals. So, please make sure to get to know the codebase before adapring it to your use-case.


In addition to all of your code, I have produced a final report, describing my implementation, and answers to parts of the project described earlier.

Here you can find: 
- A list of top 100 most similar pairs (include a suitable estimate of their similarity for each pair), sorted by similarity
- A comparison between the average pairwise correlations between these highly similar pair and randomly picked pairs
- Documentation of how your train/validation splits were generated
- Any additional pre-processing of the data that you decide to implement
- Evaluation of popularity baseline
- Documentation of latent factor model's hyper-parameters and validation
- Evaluation of latent factor model
