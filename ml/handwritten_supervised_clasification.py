'''
hadwritten supervised classification
'''

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

#load the dataset
digits = load_digits()

print(digits.images.shape)

# plot some elements of that dataset
'''
fig, axes = plt.subplots(10,10,figsize=(8,8), subplot_kw=({'xticks':[], 'yticks':[]}))


for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary')
    ax.text(0.05,0.05,str(digits.target[i]), color = 'green')

plt.show()
'''

#[n_samples, n_features]
X = digits.data
y = digits.target

#split the ddata into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,train_size=0.7)

# define the model
model = GaussianNB()

# train the model
model.fit(Xtrain,ytrain)

# use the model for predicting new data
y_predictions = model.predict(Xtest)

# compute the accuracy score
accuracy = accuracy_score(ytest,y_predictions)

print(accuracy)

# compute the confusion matrix
mat = confusion_matrix(ytest, y_predictions)

sns.heatmap(mat,square=True,annot=True,cbar=False)
plt.xlabel('Predicted value')
plt.ylabel('True value')
plt.show()

fig, axes = plt.subplots(10,10,figsize=(8,8), subplot_kw=({'xticks':[], 'yticks':[]}))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary')
    ax.text(0.05,0.05,str(digits.target[i]), color = 'green' if (ytest[i]==y.predictions[i]) else 'red')

plt.show()