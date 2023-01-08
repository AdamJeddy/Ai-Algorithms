# -*- coding: utf-8 -*-
"""
# Machine Learning in PySpark

We will look at implementation of machine learning techniques such as logisitic regression in PySpark. 
The dataset used can be found on https://github.com/sam16tyagi/Machine-Learning-techniques-in-python/blob/master/logistic%20regression%20dataset-Social_Network_Ads.csv

## Exercise 1 Load your dataset
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

spark

df = spark.read.csv("Social_Network_Ads.csv", inferSchema=True, header=True)

"""Use print schema to show a summary of your data. """

df.printSchema()

"""show the top 2"""

df.show(2)

"""
Exercise 2 

Using Select() take all features except UserID.
"""

df2 = df.select("Gender","Age", "EstimatedSalary", "Purchased")
df2.show(2)

"""
Exercise 3 

Use a 70-30 ratio for train and testing split.
"""

test, train = df2.randomSplit([0.3, 0.7], seed=10)

[test.count(), train.count()]

test.show(2)

train.show(2)

"""
Dtypes

In this dataset, any column of type string is treated as a categorical feature, but sometimes we might have numeric features we want treated as categorical or vice versa. We’ll need to carefully identify which columns are numeric and which are categorical.
"""

train.dtypes

"""
One hot encoding for categorical values

StringIndexer:
Converts a single feature to an index feature.
http://spark.apache.org/docs/latest/ml-features#stringindexer

OneHotEncoder:
http://spark.apache.org/docs/latest/ml-features#onehotencoder
"""

#new 

catCols = [x for (x, dataType) in df2.dtypes if dataType == "string"]
numCols = [ x for (x, dataType) in df2.dtypes if (dataType == "int") & (x != "Purchased") ]
print(numCols)
print(catCols)

df2.agg(F.countDistinct("Gender")).show()

df2.groupBy("Gender").count().show()

# ______ OLD _______
#catCols = [x for (x, dataType) in train.dtypes if dataType == "string"]
#numCols = [
#    x for (x, dataType) in train.dtypes if ((dataType == "int") & (x != "Purchased"))
#]
#print(numCols)
#print(catCols)

#train.agg(F.countDistinct("Gender")).show()

#train.groupBy("Gender").count().show()

from pyspark.ml.feature import (
    OneHotEncoder,
    StringIndexer,
)

string_indexer = [
    StringIndexer(inputCol=x, outputCol=x + "_StringIndexer", handleInvalid="skip")
    for x in catCols
]

string_indexer

one_hot_encoder = [
    OneHotEncoder(
        inputCols=[f"{x}_StringIndexer" for x in catCols],
        outputCols=[f"{x}_OneHotEncoder" for x in catCols],
    )
]

one_hot_encoder

"""
Vector assembling

VectorAssembler:
Combines the values of input columns into a single vector.
http://spark.apache.org/docs/latest/ml-features#vectorassembler

"""

from pyspark.ml.feature import VectorAssembler

assemblerInput = list(numCols)
assemblerInput += [f"{x}_OneHotEncoder" for x in catCols]

assemblerInput

vector_assembler = VectorAssembler(
    inputCols=assemblerInput, outputCol="VectorAssembler_features"
)

"""## Exercise 4
Stage together all your process 


"""

stages = [string_indexer[0], one_hot_encoder[0], vector_assembler]

stages

# Commented out IPython magic to ensure Python compatibility.
# %%time
# from pyspark.ml import Pipeline
# 
# pipeline = Pipeline().setStages(stages)
#

"""
Exercise 4 

Describe the timings found above. Refer to pipeline documentation on spark. 
"""

# System time is time spent running code on the OS kernal
#  - 926 microseconds
# CPU time is time spent on the processor running your 
# program's code
#  - 495 microseconds
# In total, sys + CPU Time is 1.42 milliseconds
# Wall time is the elapsed real time of the code running

# fit and transform model
model = pipeline.fit(train)
pp_df = model.transform(test)

"""
Exercise 5 

Select the correct feature vectors 
"""

temp = pp_df.select("Age", "EstimatedSalary", "Gender_OneHotEncoder", "VectorAssembler_features", "Purchased")
temp.show(20)

"""### Logistic Regression"""

# from pyspark.ml.classification import LogisticRegression

"""
Exercise 6

Select and assemble your data
"""

data = pp_df.selectExpr("VectorAssembler_features as features", "Purchased as label")

data.show(5, truncate=False)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# model = LogisticRegression().fit(data)

model.summary.areaUnderROC

model.summary.pr.show()

"""
Exercise 7 

Obtain the confusion matrix of the classifer.
"""

# Commented out IPython magic to ensure Python compatibility.
Summary = model.summary

accuracy = Summary.accuracy
falsePositiveRate = Summary.weightedFalsePositiveRate
truePositiveRate = Summary.weightedTruePositiveRate
fMeasure = Summary.weightedFMeasure()
precision = Summary.weightedPrecision
recall = Summary.weightedRecall
print("Accuracy: %s \nPrecision: %s\nRecall: %s\n\nFalse Positive Rate: %s\nTrue Positive Rate: %s\nF-measure: %s")
#       % (accuracy, precision, recall, falsePositiveRate, truePositiveRate, fMeasure))
