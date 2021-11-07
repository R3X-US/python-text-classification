import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
import re

#data link: http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip

DATA_DIR = "./bbc/"

data = load_files(DATA_DIR, encoding="utf-8", decode_error="replace")
# calculate count of each category
labels, counts = np.unique(data.target, return_counts=True)
# convert data.target_names to np array for fancy indexing
labels_str = np.array(data.target_names)[labels]
#print(dict(zip(labels_str, counts)))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
list(t[:80] for t in X_train[:10])


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000, decode_error="ignore")
vectorizer.fit(X_train)

vectorizer.fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

from sklearn.naive_bayes import MultinomialNB
cls = MultinomialNB()
# transform the list of text to tf-idf before passing it to the model
cls.fit(vectorizer.transform(X_train), y_train)
 
from sklearn.metrics import classification_report, accuracy_score
 
y_pred = cls.predict(vectorizer.transform(X_test))
#print(accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score

svc_tfidf = Pipeline([
        ("tfidf_vectorizer", TfidfVectorizer(stop_words="english", max_features=3000)),
        ("linear svc", SVC(kernel="linear"))
    ])



model = svc_tfidf
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("accuracy_score: "+ str(accuracy_score(y_test, y_pred)))
#print(classification_report(y_test, y_pred))

texts = [
"New tech recovers pure silicon from end-of-life solar cells",
"Chris Barnett wallops Gian Villante with a spinning hook kick at UFC 268.",
"Bloomberg article in case it’s gated behind the paywall.China Posts Robust Export Growth in October, Beats EstimateExports up 27.1% in October, beating economists’ expectationsExport demand supports China’s weaker domestic economyChina’s export growth beat expectations in October as foreign demand for its goods continued to surge, despite global supply chain disruptions."
]

percentofcontent = [0.0,0.0,0.0,0.0,0.0]
sizeofcontent = [0,0,0,0,0]

types = "business", "entertainment", "politics", "sport", "tech"
for text in texts:
	
	labelinlist = model.predict([text])
	labelinlist = int(((str(labelinlist)).replace("[","")).replace("]",""))
	texttype = types[int((labelinlist))]


	sizeofcontent[labelinlist] = int(sizeofcontent[labelinlist]) + 1 

	print(text +"\nClass: " +texttype+"\n\n\n")

for i in range(len(percentofcontent)):
	x = sizeofcontent[i]

	percentofcontent[i] = (x/(len(texts) + 1)) * 100
matplotlibpercents = []
matplotliblabels = []
for i in range(len(types)):
	if percentofcontent[i] != 0:
		matplotlibpercents.append(percentofcontent[i])
		matplotliblabels.append(types[i])

explode = [0] * (len(matplotliblabels) )
explode = tuple(explode)
fig1, ax1 = plt.subplots()
ax1.pie(matplotlibpercents, explode=explode, labels=matplotliblabels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()

