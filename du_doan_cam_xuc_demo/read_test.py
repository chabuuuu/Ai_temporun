import numpy as np
import pickle
import tensorflow as tf
from preprocess import *
model=tf.keras.models.load_model('my_model_balancedata_relabel.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

# Mở file


with open('dataset_test.txt', 'r', encoding='utf8') as file:
    # Đọc từng dòng và in ra
    res = []
    for line in file:
        print(line)
        line = preprocess(line)
        array = cv.transform([line]).toarray()

        pred = model.predict(array)

        all_res = pred

        a = np.argmax(pred, axis=1)
        prediction = encoder.inverse_transform(a)[0]
        print(prediction)
        print(all_res)
        res.append(prediction)

with open('result.txt', 'w') as file:
    for result in res:
        file.write(result + '\n')




