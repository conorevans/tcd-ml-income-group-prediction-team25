import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor

model_frame = pd.read_csv('data/train.csv')
target_frame = pd.read_csv('data/test.csv')

def treat_gender(frame):
  frame['Gender'] = frame['Gender'].replace({'unknown': 'not_given'})
  frame['Gender'] = frame['Gender'].replace({'f': 'female'})
  return frame

def treat_university_degree(frame):
  frame['University Degree'] = frame['University Degree'].replace({'No': 'not_given'})
  return frame

def treat_work_experience(frame):
    frame['Work Experience in Current Job [years]'] = pd.to_numeric(frame['Work Experience in Current Job [years]'], errors='coerce')
    return frame

def treat_additional_income(frame):
  frame['Yearly Income in addition to Salary (e.g. Rental Income)'] = frame['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: float(x.rstrip('EUR')))
  return frame

def treat_year_of_record(frame):
  frame['Year of Record'] = frame['Year of Record'].fillna(method='bfill')
  return frame

def preprocess(frame):
  frame = treat_gender(frame)
  frame = treat_university_degree(frame)
  frame = treat_work_experience(frame)
  frame = treat_additional_income(frame)
  frame = treat_year_of_record(frame)
  frame = frame.replace('#NUM!', np.NaN)
  frame = frame.fillna(method = 'bfill')
  return frame

model_frame = preprocess(model_frame)
target_frame = preprocess(target_frame)

model_frame['Small City'] = model_frame['Size of City'] <= 3000
target_frame['Small City'] = target_frame['Size of City'] <= 3000
    
target_columns = ['Work Experience in Current Job [years]','Year of Record', 'Gender', 'Crime Level in the City of Employement', 'Country', 'Age', 'Profession', 'University Degree', 'Small City', 'Size of City', 'Yearly Income in addition to Salary (e.g. Rental Income)']

independent_vars = model_frame[target_columns]
dependent_var = model_frame['Total Yearly Income [EUR]'].apply(np.log).values

gcsv = GridSearchCV(estimator = CatBoostRegressor(random_state=15000),
                    param_grid = { 'n_estimators': (400, 800), 'max_depth': (4, 8, 12) }, 
                    n_jobs = -1, cv = 5, verbose=1, scoring='neg_mean_absolute_error')

regr = Pipeline(steps=[('enc', TargetEncoder()),
                       ('grid', gcsv)])

X_train, X_test, Y_train, Y_test = train_test_split(independent_vars, dependent_var, train_size = 0.8, test_size = 0.2)

regr.fit(X_train, Y_train)

y_predict = np.exp(regr.predict(target_frame[target_columns]))
print(metrics.mean_absolute_error(np.exp(Y_test), np.exp(regr.predict(X_test))))

# Instances saved to separate file for ease of access
instances = pd.read_csv('data/instances.csv')['Instance'].values
f = open("data/submission.csv", "w")

# Write to File
f.write("Instance,Total Yearly Income [EUR]\n")

for i in range(len(y_predict)):
  f.write(str(instances[i]) + "," + str(y_predict[i]) + "\n")

print('submission complete')