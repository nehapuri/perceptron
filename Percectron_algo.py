import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#extract data from the given location
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header = None)

#printing the data sheet 
print(df.head())

# plt plot the datasheet/dataset
# fig is variable used to plot the dataset
fig = plt.figure()

#giving 3d projection
ax = fig.add_subplot(111, projection = '3d')
y = df.iloc[0:100,4].values
#assigning values ie., if value is between -1 to 1 it is Iris-setosa
y = np.where(y == 'Iris-setosa', -1,1)
X = df.iloc[0:100 , [0,1,2]].values

ax.scatter(X[:50 ,0] , X[:50,1],color = 'red', marker = 'o', label = 'setosa')
ax.scatter(X[50:100,0],X[50:100,1],color = 'blue', marker = 'x', label = 'versicolor')
ax.scatter(X[:100,1] , X[:100,2],color ='yellow', marker='^' ,label='target')
ax.set_xlabel('petal lenght')
ax.set_ylabel('sepal lenght')
ax.set_zlabel('target')

#ploting the above data
plt.show()

class Perceptron(object):
     #Perceptron classifier
     def __init__(self, eta = 0.01 , n_itr = 10):
          self.eta = eta
          self.n_itr = n_itr
          
     def fit(self,X,y):
          self.w_ = np.zeros(1 + X.shape[1])
          self.errors_ = []

          for _ in range(self.n_itr):
               errors = 0
               for xi,target in zip(X,y):
                    update = self.eta*(target - self.predict(xi))
                    self.w_[1:] += update*xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
               self.errors_.append(errors)
          return self

     def net_input(self,X):
          return np.dot(X, self.w_[1:]) + self.w_[0]

     def predict(self,X):
          return np.where(self.net_input(X) >= 0.0, 1 ,-1)













