from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#for each line in the text file, this method will be invoked
#and this meth returns LabeledPoint of values
#syntax: LabeledPoint(label, array of features)
def parseToFloat(line):
	value=[float(a) for a in line.split(" ")]
	return LabeledPoint(value[0],value[1:])


sc = SparkContext(appName="LogisticRegressionMLlib")

data = sc.textFile("/home/prab/spark/spark-2.0.1-bin-hadoop2.7/data/mllib/sample_svm_data.txt")

parsedRDD = data.map(parseToFloat).cache() #caching as it might be used in future.

#model
model = LogisticRegressionWithSGD.train(parsedRDD)

#predict
#labeledPredRDD = tuple of  (label , predicted output) - ex: 1,1; 0,1 so on
labeledPredRDD = parsedRDD.map(lambda p : (p.label,model.predict(p.features)))
#to find train error counting number of mismatch in labeled and predicted and div by total data count.
trainError = labeledPredRDD.filter(lambda (l,p) :l!=p).count()/float(parsedRDD.count())

#printing the error
print("Train Error = "+str(trainError))

