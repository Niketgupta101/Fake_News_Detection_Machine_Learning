# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:18:03 2021

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r'news.csv')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df.label, test_size=0.2,random_state = 7)

import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

tv=TfidfVectorizer(stop_words='english',max_df=0.7)
tv_train= tv.fit_transform(X_train)
tv_test= tv.transform(X_test)

pac=PassiveAggressiveClassifier(max_iter=1000)
pac.fit(tv_train,y_train)

y_pred= pac.predict(tv_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)

score=accuracy_score(y_test,y_pred)