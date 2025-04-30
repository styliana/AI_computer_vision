import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
#load the iris dataset
# [n_samples, n_features]
# X= data
# y = label
iris = sns.load_dataset('iris') #iris is pandas dataframe

#print(iris.head())

# Create a pairplot with better formatting
# sns.pairplot(iris, hue='species', height=2.5) 
# plt.suptitle("Pair Plot of Iris Dataset", y=1.02)  # Add a title
# plt.show()  # Explicitly show the plot
'''

# Load the iris dataset
iris = sns.load_dataset('iris')  # iris is pandas dataframe

# Create features and labels
x_iris = iris.drop('species', axis=1)  # Features
y_iris = iris['species']  # Labels

# Create the training and test datasets
xtrain, xtest, ytrain, ytest = train_test_split(x_iris, y_iris, test_size=0.5)

# Create the instance of the model - if u want anothe rmodel u change it
model = GaussianNB()

# Train the model
model.fit(xtrain, ytrain)

# Make predictions
y_prediction = model.predict(xtest)

# Check the model's accuracy
accuracy = accuracy_score(ytest, y_prediction)
print("Model accuracy:", accuracy)




'''

- Model Performance (Bias vs. Variance):

Larger train_size (e.g., 80%):
More data for training → Model learns better patterns.
Risk: If training data is too large, the test set becomes small, leading to unreliable accuracy estimates.

Smaller train_size (e.g., 50%):
Less training data → Model may underfit (high bias).
More test data → Better evaluation of generalization.

- Accuracy & Overfitting
If the training set is too small, the model may not learn well (low accuracy).
If the test set is too small, accuracy scores may vary a lot (high variance in evaluation).
'''