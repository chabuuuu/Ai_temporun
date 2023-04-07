import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(tf.__version__)

import keras
keras.__version__

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


stopwords_vn = []
with open('vietnamese-stopwords.txt', 'r', encoding='utf8') as f:
    for word in f:
        stopwords_vn.append(word.strip())


dataset=pd.read_table('train_new.txt', delimiter = '\t', header=None, )



train, test_val = train_test_split(dataset, test_size=0.2)
val, test = train_test_split(test_val, test_size=0.2)





#
data = pd.concat([train ,  val , test])


data.columns = ["label", "text"]

#
data.shape
#
data.isna().any(axis=1).sum()
#
# #text preprocessing
ps = PorterStemmer()
#
def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming
    review = [ps.stem(word) for word in review if not word in stopwords_vn] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)
#
# data['text']=data['text'].apply(lambda x: preprocess(x))

data['text'] = data['text'].apply(lambda x: str(x))
data['text'] = data['text'].apply(lambda x: preprocess(x))


#
from sklearn import preprocessing
#
label_encoder = preprocessing.LabelEncoder()
data['N_label'] = label_encoder.fit_transform(data['label'])
#
#
data['text']
#
#
# # Tạo bag of word nè
#
#
from sklearn.feature_extraction.text import CountVectorizer
#
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))#example: the course was long-> [the,the course,the course was,course, course was, course was long,...]
#
data_cv = cv.fit_transform(data['text']).toarray()
#
data_cv
#
#X_train, X_test, y_train, y_test=data_cv,test_cv,train['N_label'],test['N_label']
X_train, X_test, y_train, y_test =train_test_split(data_cv, data['N_label'], test_size=0.25, random_state=42)

# # Tạo model
from keras import Sequential
from keras.layers import Dense
# load dataset
# Chia thành X: text, y: label
# tạo layer cho model:
model = Sequential()
model.add(Dense(12, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='softmax'))
# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(X_train, y_train, epochs=100, batch_size=1)
# evaluate model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))
#
#
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
#
#
text='Thầy dạy hay'
text=preprocess(text)
array = cv.transform([text]).toarray()
pred = model.predict(array)
a=np.argmax(pred, axis=1)
label_encoder.inverse_transform(a)[0]

tf.keras.models.save_model(model,'my_model.h5')

import pickle
pickle.dump(label_encoder, open('encoder.pkl', 'wb'))
pickle.dump(cv, open('CountVectorizer.pkl', 'wb'))
