'''
Train a ml model able to classify text topic. 
On the chart we can see if the lables were correcly clasisfied or missclassified by topic.
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#get the data
# we wil use a subset of the categories
categories = ['talk.religion.misc', 'sci.space', 'comp.graphics']

#load the dataset
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#create a pipeline
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

#train the model 
model.fit(train.data, train.target)

#test the model
labels = model.predict(test.data)

'''
#print the confusion matrix
conf_matrix = confusion_matrix(test.target, labels)
sns.heatmap(conf_matrix.T, fmt='d', square=True, cbar=False, annot=True, 
xticklabels=train.target_names, yticklabels=train.target_names) #coolor bar -cbar - we dont want to show, we want to show the squares:)
plt.xlabel('true label')
plt.ylabel('predicted labels')
plt.show()
'''


#try the classifier with out text
s = 'I believe i can fly' #assigns that the topic is space. which is right
# s = 'us the texture as you wish' #assigns that the topic is space. which is wrong. we meant comp graphics
# s = 'you have to fo the raytracing of the planets' #planets are more importnat in the code. which assigns space, even though its comp graphics
pred = model.predict([s])
print(train.target_names[pred[0]])