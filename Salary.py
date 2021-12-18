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
x_train = sm.add_constant(x_train) # b0
x_test = sm.add_constant(x_test) # b0
# model = sm.OLS(y_train, X_train).fit()
# model.summary()