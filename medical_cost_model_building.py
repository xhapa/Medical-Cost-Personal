import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import pickle

medical_cost_df = pd.read_csv('insurance_ext.csv')

X_features = ['age2', 'owbysmok', 'smoker_yes', 'children']
Y_features = ['charges']

print(X_features)

X = medical_cost_df[X_features].values
Y = medical_cost_df[Y_features].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

sc_X = StandardScaler().fit(X)
sc_Y = StandardScaler().fit(Y)

X_train_std = sc_X.transform(X_train)
X_test_std = sc_X.transform(X_test)
Y_train_std = sc_Y.transform(Y_train)
Y_test_std = sc_Y.transform(Y_test)

model = LinearRegression()
model.fit(X_train_std, Y_train_std)

Y_pred = model.predict(X_test_std)
print(Y_pred.shape)

mse = metrics.mean_squared_error(Y_test_std, Y_pred)
r2 = metrics.r2_score(Y_test_std, Y_pred)
print(f'MSE: {mse.round(4)}\t R2: {r2.round(4)}')

pickle.dump(model, open('./models/medical_cost_mod.pkl', 'wb'))
pickle.dump(sc_X, open('./models/scalerX.pkl', 'wb'))
pickle.dump(sc_Y, open('./models/scalerY.pkl', 'wb'))