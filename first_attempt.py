import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor

model_frame = pd.read_csv('data/train.csv')

target_columns = ['Year of Record','Age', 'Size of City', 'Yearly Income in addition to Salary (e.g. Rental Income)']

for column in target_columns:
  model_frame[column] = model_frame[column].fillna(method='ffill')

independent_vars = model_frame[target_columns]
dependent_var = model_frame['Total Yearly Income [EUR]'].values

gcsv = GridSearchCV(estimator = CatBoostRegressor(random_state=15000),
                    param_grid = { 'n_estimators': (100, 200, 250), 'max_depth': (2, 4, 8) }, 
                    n_jobs = -1, cv = 5, verbose=1, scoring='neg_mean_squared_error')

regr = Pipeline(steps=[('enc', TargetEncoder()),
                       ('grid', gcsv)])

X_train, X_test, Y_train, Y_test = train_test_split(independent_vars, dependent_var, train_size = 0.8, test_size = 0.2)

regr.fit(X_train, Y_train)

target_frame = pd.read_csv('data/test.csv')[target_columns].fillna(method='ffill')

y_predict = regr.predict(target_frame[target_columns])
print(metrics.mean_absolute_error(Y_test, regr.predict(X_test)))

# Instances saved to separate file for ease of access
instances = pd.read_csv('data/instances.csv')['Instance'].values
f = open("data/submission.csv", "w")

# Write to File
f.write("Instance,Total Yearly Income [EUR]\n")

for i in range(len(y_predict)):
  f.write(str(instances[i]) + "," + str(y_predict[i]) + "\n")
