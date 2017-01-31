#Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt



#Load Dataset
#local- C:\Users\Vignesh\Documents\ML_mastery\iris.data
url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length',  'petal-width',  'species']
dataset = pandas.read_csv(url, names=names)
#shape
print(dataset.shape)
#head
print(dataset.head(20))
#descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('species').size())
#box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#histograms
dataset.hist()
plt.show()
#scatter plot matrix
scatter_matrix(dataset)
plt.show()



