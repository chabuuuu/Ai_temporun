import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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


dataset=pd.read_table('train_after_relabel.txt', delimiter = '\t', header=None, )



train, test_val = train_test_split(dataset, test_size=0.1)
val, test = train_test_split(test_val, test_size=0.1)





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
    # review = re.sub('[^a-zA-Z]', ' ', line) #chỉ để lại kí tự từ a-z, các dấu câu ! ) (
    # review = re.sub('[^a-zA-Z!)(]', ' ', line)
    review = re.sub('[^a-zA-Z!)(:^áàạảãăắẵặằấầẩẫậâèéẽẻẹềếệểễòóỏõọôổỗộốồớờơợởỡûùúủũụíìỉĩịêđưứừựữửỳýỹỵỷ]', ' ', line)


    review = review.lower() #chuyển chữ hoa thành chữ thường
    review = review.split() #chuyển chuoỗi thành danh sách các từ
    #apply Stemming
    # review = [ps.stem(word) for word in review if not word in stopwords_vn] #delete stop words like I, and ,OR   review = ' '.join(review)
    #chuyển list thành sentences
    return " ".join(review)
#
# data['text']=data['text'].apply(lambda x: preprocess(x))

data['text'] = data['text'].apply(lambda x: str(x))
data['text'] = data['text'].apply(lambda x: preprocess(x))



##
# def preprocess_label(line):
#     line = line.split() #chuyển chuoỗi thành danh sách các từ
#     # if line != "positive" and line != "negative":
#     #     line = "neutral"
#     return "".join(line)
# #
#
#
# data['label'] = data['label'].apply(lambda y: str(y))
# data['label'] = data['label'].apply(lambda y: preprocess_label(y))
#Căn lề trái cho label
data['label'] = data['label'].apply(lambda x: x.ljust(len(max(data['label'], key=len))))



#
from sklearn import preprocessing
#
label_encoder = preprocessing.LabelEncoder()
data['N_label'] = label_encoder.fit_transform(data['label'])

# print(data['N_label'])

#
# data['N_label'].to_csv('N_labels.txt', index=False, header=None)
# data['text'].to_csv('labels.txt', index=False, header=None)




#
# print(data['text'])
# # Balance data using SMOTE
# smote = SMOTE(sampling_strategy='minority')
# X_sm, y_sm = smote.fit_resample(data['text'].values.reshape(-1, 1), data['N_label'])
#
# # Convert X_sm and y_sm to data frame
# data_sm = pd.DataFrame({'text': X_sm[:, 0], 'N_label': y_sm})
#
# # Convert N_label to original label
# data_sm['label'] = label_encoder.inverse_transform(data_sm['N_label'])
#
# # Print value counts of N_label in data_sm
# print(data_sm['N_label'].value_counts())

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


#Cân bằng dữ liệu
smote =SMOTE(sampling_strategy='minority')
text_sm, label_sm = smote.fit_resample(data_cv, data['N_label'])
text_sm2, label_sm2 = smote.fit_resample(text_sm, label_sm)
print(label_sm2.value_counts())
#
#X_train, X_test, y_train, y_test=data_cv,test_cv,train['N_label'],test['N_label']
# X_train, X_test, y_train, y_test =train_test_split(data_cv, data['N_label'], test_size=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(text_sm2, label_sm2, test_size=0.1, random_state=42)
print(X_train[0])

# # Tạo model
# from keras import Sequential
# from keras.layers import Dense
# # load dataset
# # Chia thành X: text, y: label
# # tạo layer cho model:
# model = Sequential()
# model.add(Dense(12, input_shape=(X_train.shape[1],), activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(3, activation='softmax'))


from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout

# Định nghĩa mô hình
model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))  # Áp dụng Dropout để tránh overfitting
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))  # Áp dụng Dropout để tránh overfitting
model.add(Dense(3, activation='softmax'))

# Compile mô hình với optimizer Adam và learning rate 0.001
opt = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])




# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model
model.fit(X_train, y_train, epochs=1000, batch_size=1)
# evaluate model
_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))
#
#
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
#
#

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.svm import LinearSVC
# vectorizer = CountVectorizer(max_features=1000)
# classifier = LinearSVC(C=1.0, class_weight='balanced')
# model_min = Pipeline(
#     [
#         ("vectorizer", vectorizer),
#         ("classifier", classifier),
#     ]
# )
#
# model_min.fit(X_train, y_train)




text='Thầy dạy hay'
text=preprocess(text)
array = cv.transform([text]).toarray()
pred = model.predict(array)
a=np.argmax(pred, axis=1)
label_encoder.inverse_transform(a)[0]

tf.keras.models.save_model(model,'dua_tren_danhgiacaonhat_epoch1000.h5')

import pickle
pickle.dump(label_encoder, open('encoder.pkl', 'wb'))
pickle.dump(cv, open('CountVectorizer.pkl', 'wb'))
