#Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_data(url):
	#Load Dataset

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
	x_train,  x_test, y_train, y_test =  model_selection.train_test_split(X, Y, test_size = size, random_state = seed )

	# Test options and evaluation metric
	seed = 7
	scoring = 'accuracy'

	# Spot Check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))
	
	#evaluating each models
	results = []
	names = []
	
	for (name, model) in models:
		kfold = model_selection.KFold(n_splits = 10, random_state = seed)
		cv_results = model_selection.cross_val_score(model, x_train, y_train, cv = kfold, scoring = scoring )
		results.append(cv_results)
		names.append(cv_results)
		print(name),":",(cv_results.mean()),"(" ,(cv_results.std()) ,")"
    
	
	#Make prediction on validation dataset
	
	knn = KNeighborsClassifier()
	knn.fit(x_train, y_train)
	predictions = knn.predict(x_test)
	print(accuracy_score(y_test, predictions))
	print(classification_report(y_test, predictions))
	
         


	
	
	
if __name__ == '__main__':
	#url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	url = "C:\Users\Vignesh\Documents\ML_mastery\iris.data"
	data = load_data(url)
	ml_algo(data)
	