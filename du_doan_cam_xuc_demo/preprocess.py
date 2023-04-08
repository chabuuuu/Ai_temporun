#text preprocessing
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
def preprocess(line):
    # stopwords_vn = []
    # with open('vietnamese-stopwords.txt', 'r', encoding='utf8') as f:
    #     for word in f:
    #         stopwords_vn.append(word.strip())
    #

    ps = PorterStemmer()

    # review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = re.sub('[^a-zA-Z!)(]', ' ', line)

    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming 
    # review = [ps.stem(word) for word in review if not word in stopwords_vn] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)
