import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('C:\Users\kumar\Desktop\DS\breast-cancer.csv')

X = np.array(df.iloc[:, 2:32])
y = np.array(df.iloc[:, 1:2])
# initialize attributes of the dataset for prediction
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
sv = SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv, open('model.pkl', 'wb'))