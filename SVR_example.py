import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data
X = np.sort(5 * np.random.rand(50, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 2 * (0.5 - np.random.rand(10))

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# Plot results
lw = 2
plt.scatter(X, y, color='red', label='data')
plt.plot(X, y_rbf, color='blue', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='orange', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='purple', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()