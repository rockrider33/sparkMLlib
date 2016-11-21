from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint,LinearRegressionWithSGD
from numpy import array

#for each line in the text file, this method will be invoked
#and this meth returns LabeledPoint of values
#syntax: LabeledPoint(label, array of features)
def parseDataSet(line):
	value=[float(a) for a in line.replace(',',' ').split(" ")]
	return LabeledPoint(value[0],value[1:])


sc = SparkContext(appName="LinearRegressionMLlib")

data = sc.textFile("/home/prab/spark/spark-2.0.1-bin-hadoop2.7/data/mllib/ridge-data/lpsa.data")

parsedRDD = data.map(parseDataSet).cache() #caching as it might be used in future.

#model
model = LinearRegressionWithSGD.train(parsedRDD)

#predict
#labeledPredRDD = tuple of  (label , predicted output)
labeledPredRDD = parsedRDD.map(lambda p : (p.label,model.predict(p.features)))

#Finding MSE- Mean Squared Error
MSE = labeledPredRDD.map(lambda (l,p) :(l-p)**2).reduce(lambda a,b: a+b)/float(labeledPredRDD.count())

#printing the error
print("MSE Error = "+str(MSE))

