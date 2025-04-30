
#DOESNT WORK, WORKIN ON IT

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer  # Added this import
import nltk

# You might need to download these if you haven't already:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

example_string = "All the speed he tool. All the turns he'd taken"

words = word_tokenize(example_string)

#set the language for the stopwords 
stop_words = set(stopwords.words('english'))
filtered_list = []

#lets remove the stopwords
for word in words:
    if word.casefold() not in stop_words:
        filtered_list.append(word)


#stemming 
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words]
#print(stemmed_words)

#part of speech (POS) tagging
tags = nltk.pos_tag(words)

print(nltk.help.upenn_tagset())

#lemmatizing
lemmatizer = WordNetLemmatizer()
lemmm = lemmatizer.lemmatize('scarves') #scarv
x = lemmatizer.lemmatize('worst')
print(x)