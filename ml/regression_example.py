import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#fixing the seed for random generator
rng = np.random.RandomState(42)

x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.rand(50)


#plt.scatter(x,y)
#plt.show()

#create and instance of the model
model = LinearRegression(fit_intercept=True)

#prepare the data
#in order to have a 3d matrix
X = x[:,np.newaxis]

#lets train the model
model.fit(X,y)

#create testing data
xtest = np.linspace(-1,11) #array in which elements have the same distance among each other
Xtest = xtest[:,np.newaxis]

#test the model
y_predictions = model.predict(Xtest)

plt.scatter(x,y)
plt.plot(xtest,y_predictions)
plt.show()



