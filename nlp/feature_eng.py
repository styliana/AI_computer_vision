from sklearn.feature_extraction import DictVectorizer

#categorical data
data = [
    {'price': 200000, 'rooms': 4, 'neighborhood':'Appia'},
    {'price': 140000, 'rooms': 3, 'neighborhood':'Tuscolana'},
    {'price': 120000, 'rooms': 2, 'neighborhood':'Casilina'}
]

neigh = {'Appia': 1, 'Tuscolana':2, 'Casilina':3} #appia = casilina - tuscolana ---> WRONG!!!!!

#one hot encoding
vec = DictVectorizer(sparse=False, dtype=int)
res = vec.fit_transform(data)

# print(res)
print(vec.get_feature_names_out())