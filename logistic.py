#import nrceassary functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lo

#load iris dataset
iris=datasets.load_iris()

#use sepal lengths and sepal width attributes 
X=iris.data[:,:2]
Y=iris.target

#fit logistic regression
logreg=LR(C=1e5,solver='lbfgs',multi_class='multinomial')
logreg.fit(X,Y)

#use logistic regression for fitting data
x_min,x_max=X[:,0].min()-.5,X[:,0].max()+.5
y_min,y_max=X[:,1].min()-.5,X[:,0].max()+.5
h=.02 #step size in mesh
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=logreg.predict(np.c_[xx.ravel(),yy.ravel()])

#display output
x_min,x_max=X[:,0].min()-.5,X[:,0].max()+.5
y_min,y_max=X[:,1].min()-.5,X[:,1].max()+.5
h=.02
xx,yy=np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
Z=logreg.predict(np.c_[xx.ravel(),yy.ravel()])


#put the result into a color plot
Z=Z.reshape(xx.shape)
plt.figure(1,figsize=(4,3))
plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)



#Plot training data
plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='k',cmap=plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.xticks(())
plt.yticks(())
plt.show()