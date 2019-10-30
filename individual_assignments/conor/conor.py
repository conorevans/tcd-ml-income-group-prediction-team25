import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor

# load data frame for data from which we will build our model
model_frame = pd.read_csv('withlabels.csv')

# fill NaN values in categorical fields with not_given
for column in model_frame.select_dtypes(include=['object']):
  # we want to mark categorical values as not_given (missing, non_present, whatever preferred term)
  # as there may be a connection between values not being given and other target variables (in our case, Income)
  # people from country X may be more reticent to share salary, height, whatever.
  # best not to ffill 
  model_frame[column] = model_frame[column].fillna('not_given')

for column in model_frame.select_dtypes(exclude=['object']).drop('Instance', axis=1):
  # cut bottom and top 10 percent of data to have a more balanced mean used to fill N/A values
  lower = np.percentile(model_frame[column].dropna(),10)
  upper = np.percentile(model_frame[column].dropna(),90)
  mean = model_frame[model_frame[column].between(lower,upper)][column].values.mean()
  model_frame[column] = model_frame[column].fillna(mean)

# import scipy.stats as stats
# this code was used to find correlation between columns and Income, to refine which columns used
"""
correlation_columns = []
for column in data_frame.columns:
  if(column != 'Instance' and column != 'Income in EUR'):
    correlation_columns.append(column)

correlations = []
for column in correlation_columns:
  correlations.append([column,stats.pearsonr(data_frame[column], data_frame['Income in EUR'])])

for tuple in correlations:
  print(tuple)
  print("\n")
"""

# these were the resulting low correlation columns, drop them
low_correlation = ['Hair Color', 'Wears Glasses', 'Size of City']
model_frame = model_frame.drop(columns=low_correlation)

columns = []
for column in model_frame.columns:
  if(column != 'Income in EUR'):
    columns.append(column)

# get our X and Y
dependent_var = model_frame['Income in EUR'].values
independent_vars = model_frame[columns]

# get features we will transform
# hard-coded but there aren't too many columns and no need to build
# something complicated if the scope does not require it
numeric_features = ['Year of Record','Age','Body Height [cm]']
categorical_features = ['Country','Gender','Profession']

# build GridSearchCV object
# using cat boost regressors
# increasing n_estimators can improve score but increases computation time
# ditto for max_depth, though capped at 16 for CBRegressor
# n_jobs to use all available CPU
# cross validate 5 times - seems to be accepted as common standard,
# I tried higher but it seemed to overfit
gcsv = GridSearchCV(estimator = CatBoostRegressor(random_state=15000),
                    param_grid = { 'n_estimators': (100, 200, 250), 'max_depth': (8, 12, 16) }, 
                    n_jobs = -1, cv = 5, verbose=1, scoring='neg_mean_squared_error')


# build pipeline
# impute using column transformers (passed directly rather than assigning to two diff objects - as done previously - to allow
# us to build only one pipeline and not two.

# again, I have already filled missing data at start of file but SimpleImputer seems to improve it so whether it is tacking on
# to my work or overwriting it is unknown
regr = Pipeline(steps=[('Imputer', (ColumnTransformer(transformers=[('num', SimpleImputer(strategy='median'), numeric_features),
                                               ('cat', SimpleImputer(strategy='most_frequent', fill_value='not_present'), categorical_features)]))),
                      # cols are hard-coded again
                       ('enc', TargetEncoder(cols=[3,4,5])),
                       ('grid', gcsv)])

# get training and test values and fit data
X_train, X_test, Y_train, Y_test = train_test_split(independent_vars, dependent_var, train_size = 0.8, test_size = 0.2)

regr.fit(X_train, Y_train)

# build data frame for our target dataset
target_frame = pd.read_csv('nolabels.csv').drop(columns=low_correlation).fillna(method='ffill')

# get predict values and print out RMSE
y_predict = regr.predict(target_frame[columns])
print(np.sqrt(metrics.mean_squared_error(Y_test, regr.predict(X_test))))

# Instances saved to separate file for ease of access
instances = pd.read_csv('instances.csv')['Instance'].values
f = open("predictions.csv", "w")

# Write to File
f.write("Instance,Income\n")

for i in range(len(y_predict)):
  f.write(str(instances[i]) + "," + str(y_predict[i]) + "\n")
