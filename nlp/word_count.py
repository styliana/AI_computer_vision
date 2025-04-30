from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#data
sample = [
    'problem of evil',
    'evil queen',
    'horizon problem'
]

#instance of the counter
vec = TfidfVectorizer()
X = vec.fit_transform(sample)

#convert X to a dataframe for easy reading
x_pd = pd.DataFrame(X.toarray(), columns = vec.get_feature_names_out())
print(x_pd)