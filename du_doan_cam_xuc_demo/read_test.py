import numpy as np
import pickle
import tensorflow as tf
from preprocess import *
model=tf.keras.models.load_model('dua_tren_danhgiacaonhat_epoch1000.h5')
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))

# Mở file


with open('data_phase_0.txt', 'r', encoding='utf8') as file:
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

# with open('submit.txt', 'w') as file:
#     for result in res:
#         result = result.strip()
#         file.write(result + '\n')
#     file.seek(0, os.SEEK_END)  # Di chuyển con trỏ tới cuối file
#     pos = file.tell() - 1  # Lưu vị trí ký tự cuối cùng
#     file.seek(pos)  # Di chuyển con trỏ tới vị trí ký tự cuối cùng
#     file.truncate()  # Xóa ký tự cuối cùng (dòng cuối cùng)
#
with open('submit.txt', 'w') as file:
    for i, result in enumerate(res):
        result = result.strip()
        file.write(result)
        if i != len(res) - 1: # Nếu dòng hiện tại không phải dòng cuối cùng
            file.write('\n') # Ghi ký tự xuống dòng

