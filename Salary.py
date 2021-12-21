import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Data Preprocessing
# =============================================================================
# read dataset with pandas
df = pd.read_csv('Salary_Data.csv')

# visualization
df.plot(x='YearsExperience', y='Salary', title="Salary evolution per years of experience")

# i/o split values
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Train/Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)


import statsmodels.api as sm
# x_train = sm.add_constant(x_train) # b0
# x_test = sm.add_constant(x_test) # b0
model = sm.OLS(y_train, x_train).fit()
model.summary()

# test linear model
import statsmodels.stats.api as sms
sms.linear_harvey_collier(model)

# calculate residuals
from statsmodels.compat import lzip
residuals = model.resid
plt.hist(residuals, range=(-45000, 45000))

# Q-Q plot
import scipy as sp
fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residuals, plot=ax, fit=True)
print(r**2)

# Test D'Agostino
from scipy.stats import normaltest
normaltest(residuals)

# Check Homoscedasticity
# Goldfeld-Quandt test 
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)

# checking homoscedasticity with a plot
fig, ax = plt.subplots(figsize=(6,2.5))
_ = ax.scatter(x_train, residuals)

#residuals autocorrelation with Durbin-Watson
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(residuals))

#leverage
from statsmodels.stats.outliers_influence import OLSInfluence
test_class = OLSInfluence(model)
test_class.dfbetas[:5,:]

from statsmodels.graphics.regressionplots import plot_leverage_resid2,influence_plot
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(model, ax = ax)
influence_plot(model)

#model's prediction
y_pred = model.predict(x_test)

#Metrics evaluation
from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_test, y_pred)
mae = mean_squared_error(y_test, y_pred)
print("r2: ", r2, "mae: ", mae)

# Visualizing the Training results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Salary vs Esperience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Visualizing the Test results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, model.predict(x_train), color = 'blue')
plt.title('Salary vs Esperience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
