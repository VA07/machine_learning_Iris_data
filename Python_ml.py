#Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn import model_selection


def load_data():
	#Load Dataset
	#url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	url = "C:\Users\Vignesh\Documents\ML_mastery\iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length',  'petal-width',  'species']
	dataset = pandas.read_csv(url, names=names) #load comma-seperated values

	#shape
	print("Shape = "),(dataset.shape)

	#Display 20 elements 
	print("head(20)/n"),(dataset.head(20))

	#descriptions
	print ("Descriptions")
	print(dataset.describe())

	#class distribution
	print("Class Distribution")
	print(dataset.groupby('species').size())

	#box and whisker plots
	dataset.plot(title = 'box and Whisker Plot', kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	plt.show()

	#histograms
	dataset.hist(), plt.suptitle("Histogram Plot")
	plt.show()

	#scatter plot matrix
	scatter_matrix(dataset), plt.suptitle("Scatter-Matrix")
	plt.show()
	
	return dataset
	
def ml_algo(data):
	#split Data into test-train sets
	seed = 7
	array = data.values
	X = array[:,0:4]
	Y = array[:,4]
	size = 0.20
	x_train,  x_test, y_train, y_test =  model_selection.train_test_split(X, Y, data, test_size = size, random_state = seed )




