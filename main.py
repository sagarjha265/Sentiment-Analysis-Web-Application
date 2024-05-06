import pandas as pd 
df = pd.read_csv(r'D:\data_Science\balanced_reviews.csv')

df.dropna(inplace=True)
print(df.isnull().any(axis= 0))
import numpy as np
df['postivity']= np.where(df['overall'] > 3,1,0)

features = df['reviewText']
labels= df['postivity']

from sklearn.model_selection import train_test_split
features_train , features_test , labels_train , labels_test = train_test_split(features , labels , test_size=0.2 , random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(min_df=2)
vect=vect.fit(features_train)
print(len(vect.get_feature_names_out()))

features_train_Vectorized = vect.transform(features_train)
print(features_train_Vectorized)

from yellowbrick.text import FreqDistVisualizer
vocab = vect.get_feature_names_out()
visualizer = FreqDistVisualizer(features=vocab ,orient='v')
visualizer.fit(features_train_Vectorized)
visualizer.show()

from yellowbrick.target import ClassBalance
visualizer = ClassBalance(labels=['negative' , 'positive'])
visualizer.fit(labels)
visualizer.show()


from sklearn.linear_model import  LogisticRegression
model = LogisticRegression()
model.fit(features_train_Vectorized , labels_train)
predict = model.predict(vect.transform(features_test))
print(predict)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix( labels_test , predict)
print(cm)

from sklearn.metrics import accuracy_score
score  =accuracy_score(labels_test , predict)
print(score)

# Team A
import pickle
file = open("pickle_model_pkl","wb")
pickle.dump(model , file)
# Team B
file = open(r"D:\data_Science\pickle_model_pkl" ,"rb")
recreated_model = pickle.load(file)
re_pred =recreated_model.predict(vect.transform(features_test))
print(re_pred)

vocab_file = open("features.model_pkl","wb")
vect.vocabulary_
pickle.dump(vect.vocabulary_,vocab_file)
