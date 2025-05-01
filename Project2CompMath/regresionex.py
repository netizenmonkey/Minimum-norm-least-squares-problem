import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from minnormls import min_norm_LSQR

df = pd.read_csv('df_test.csv')
data = df.to_numpy()

m = 6700
r1 = np.random.randint(0, m, 5000)
I=list(range(0,m))
J = np.unique(r1)
K = np.setdiff1d(I, J)

A_train = data[J, 2:].astype(float)
b_train = data[J, 1:2].astype(float)

reg = LinearRegression()
reg.fit(A_train, b_train)

A_test = data[K, 2:].astype(float)
b_test = data[K, 1:2].astype(float)

x_custom = min_norm_LSQR(A_train, b_train)
pred_builtin = reg.predict(A_test)
pred_custom = A_test @ x_custom

print(np.column_stack((np.round(pred_builtin), np.round(pred_custom), b_test)))
print(np.round(np.max(np.abs(pred_builtin - b_test))))
print(np.round(np.max(np.abs(pred_custom - b_test))))
