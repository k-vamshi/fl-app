import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv('taxi.csv')
print(data.head())

data_x = data.iloc[:, 0:-1]
data_y = data.iloc[:,-1]
print(data_y)

x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.3, random_state=0)

reg = LinearRegression()
reg.fit(x_train, y_train)

print("Training score: ", reg.score(x_train, y_train))
print("Training score: ", reg.score(x_test, y_test))

pickle.dump(reg, open('taxi.pkl', 'wb'))
model = pickle.load(open('taxi.pkl', 'rb'))

print(model.predict([[80, 1730000, 6000, 45]]))
