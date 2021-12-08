import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize import sent_tokenize
import csv
import pandas as pd
from csv import writer




def classify(text):


	filename = 'finalized_model.sav'
	model = pickle.load(open(filename, 'rb'))





	types = "business", "entertainment", "politics", "science", "sport", "tech"

	labelinlist = model.predict([text])
	labelinlist = int(((str(labelinlist)).replace("[","")).replace("]",""))


	texttype = str(types[int((labelinlist))])


	probs_as_two_dimensional = model.predict_proba([text])

	probs = []


	for x in range(len(types)):
		probs.append(round((probs_as_two_dimensional[0][x])*100))

	texttoreturn = texttype + " probabilities: " + str(probs)





	return texttoreturn
classify("your text to classify")
