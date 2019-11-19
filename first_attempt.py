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
target_frame = pd.read_csv('data/test.csv')

model_frame['Small City'] = model_frame['Size of City'] <= 3000
target_frame['Small City'] = target_frame['Size of City'] <= 3000

def preprocess(frame):
    frame = drop_duplicates(frame)
    frame = drop_unwanted_columns(frame)
    frame = treat_year_of_record(frame)
    frame = treat_housing_situation(frame)
    frame = treat_work_experience(frame)
    frame = treat_university_degree(frame)
    frame = treat_gender(frame)
    frame = treat_additional_income(frame)
    frame = treat_NaN_values(frame)
    return frame

def drop_duplicates(frame):
    frame.sort_values('Instance', inplace = True)
    frame.drop_duplicates('Instance', keep = 'first', inplace = True)
    return frame

def drop_unwanted_columns(frame):
    frame = frame.drop(columns = ['Instance','Wears Glasses','Hair Color'])
    return frame

def treat_year_of_record(frame):
    frame['Year of Record'] = frame['Year of Record'].fillna(method='bfill')
    return frame

def treat_gender(frame):
    frame['Gender'] = frame['Gender'].replace({'f': 'female'})
    return frame

def treat_housing_situation(frame):
    frame['Housing Situation'] = frame['Housing Situation'].replace({'0': 'none', 0: 'none', 'nA': 'none'})
    frame = frame.astype({'Housing Situation': str})
    return frame

def treat_work_experience(frame):
    frame['Work Experience in Current Job [years]'] = pd.to_numeric(frame['Work Experience in Current Job [years]'], errors='coerce')
    return frame

def treat_university_degree(frame):
    frame.loc[frame['University Degree'] == '0', 'University Degree'] = 'No'
    return frame

def treat_additional_income(frame):
    frame['Yearly Income in addition to Salary (e.g. Rental Income)'] = frame['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: float(x.rstrip('EUR')))
    return frame

def treat_NaN_values(frame):
    frame = frame.replace('#NUM!', np.NaN)
    frame = frame.fillna(method = 'bfill')
    return frame

def scale_income(frame):
    frame['Total Yearly Income [EUR]'] = np.log(frame['Total Yearly Income [EUR]'])
    return frame['Total Yearly Income [EUR]'].values

model_frame = preprocess(model_frame)
target_frame = preprocess(target_frame)

target_columns = ['Year of Record', 'Gender', 'Crime Level in the City of Employement','Satisfation with employer', 'Country', 'Age', 'Profession', 'University Degree', 'Small City', 'Size of City']
numerical_features = ['Year of Record', 'Crime Level in the City of Employement', 'Age', 'Size of City']
categorical_features = ['Gender', 'Satisfation with employer', 'Country', 'Profession', 'University Degree']

independent_vars = model_frame[target_columns]
dependent_var = scale_income(model_frame['Total Yearly Income [EUR]'])

gcsv = GridSearchCV(estimator = CatBoostRegressor(random_state=15000, od_type='Iter',od_wait=100),
                    param_grid = { 'n_estimators': (200, 400), 'max_depth': (4, 8, 12) }, 
                    n_jobs = -1, verbose=10, cv = 5, scoring='neg_mean_absolute_error')

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
