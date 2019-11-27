# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from scipy.stats import stats
from lightgbm import LGBMRegressor

# read in data
model_frame = pd.read_csv('data/train.csv')
target_frame = pd.read_csv('data/test.csv')

def treat_gender(frame):
  frame['Gender'] = frame['Gender'].replace({'f': 'female'})
  return frame

def treat_work_experience(frame):
  frame['Work Experience in Current Job [years]'] = pd.to_numeric(frame['Work Experience in Current Job [years]'], errors='coerce')
  frame['Work Experience in Current Job [years]'] = frame['Work Experience in Current Job [years]'].fillna(method='bfill')
  frame['significant-work-experience'] = frame['Work Experience in Current Job [years]'].between(15,30)
  return frame

# remove EUR from additional income input i.e. change '2555.12 EUR' to 2555.12
def treat_additional_income(frame):
  frame['Yearly Income in addition to Salary (e.g. Rental Income)'] = frame['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: float(x.rstrip('EUR')))
  frame = frame.astype({'Yearly Income in addition to Salary (e.g. Rental Income)': float})
  frame['high-additional-income'] = frame['Yearly Income in addition to Salary (e.g. Rental Income)'] >= 20000.00
  return frame

# 'Year of Record' appears to be ordered
# give all the n/a values are at the end of the file, we backwards fill
def treat_year_of_record(frame):
  frame['Year of Record'] = frame['Year of Record'].fillna(method='bfill')
  frame['post-2010'] = frame['Year of Record'] >= 2010.0
  return frame

# instead of encoding ~200 countries, build two columns using
# pearson correlation coefficient for correlation and p_value for
# non-correlation, then drop original country column
def treat_country(model_frame, target_frame):
  model_frame['country-pcc'] = model_frame['Country']
  model_frame['country-p_value'] = model_frame['Country']

  target_frame['country-pcc'] = target_frame['Country']
  target_frame['country-p_value'] = target_frame['Country']
  pcc_dict, pval_dict = {}, {}

  for country in model_frame['Country'].unique():
    # scipy.stats.pearsonr(x, y)
    # Calculates a Pearson correlation coefficient ('pcc') and the p-value for testing non-correlation.
    pcc, p_value = stats.pearsonr(model_frame['Country'] == country, model_frame['Total Yearly Income [EUR]'])
    unrelated = p_value > 0.05
    pcc_dict[country] = pcc
    # encode as string because we will have three values: {'True', 'False', 'unknown'}
    pval_dict[country] = str(unrelated)

    model_frame.loc[model_frame['Country'] == country, 'country-pcc'] = float(pcc)
    model_frame.loc[model_frame['Country'] == country, 'country-p_value'] = str(unrelated)

  for country in target_frame['Country'].unique():
    if country in pcc_dict:
      target_frame.loc[target_frame['Country'] == country, 'country-pcc'] = pcc_dict[country]
      target_frame.loc[target_frame['Country'] == country, 'country-p_value'] = pval_dict[country]
    else:
      target_frame.loc[target_frame['Country'] == country, 'country-pcc'] = 'nA'
      target_frame.loc[target_frame['Country'] == country, 'country-p_value'] = 'unknown'

  model_frame = model_frame.drop('Country', axis=1)
  target_frame = target_frame.drop('Country', axis=1)
  target_frame = target_frame.fillna(method = 'bfill')

  return [model_frame, target_frame]

# same as country
def treat_profession(model_frame, target_frame):
  model_frame['profession-pcc'] = model_frame['Profession']
  model_frame['profession-p_value'] = model_frame['Profession']

  target_frame['profession-pcc'] = target_frame['Profession']
  target_frame['profession-p_value'] = target_frame['Profession']
  pcc_dict, pval_dict = {}, {}

  for profession in model_frame['Profession'].unique():
    pcc, p_value = stats.pearsonr(model_frame['Profession'] == profession, model_frame['Total Yearly Income [EUR]'])
    unrelated = p_value > 0.05
    pcc_dict[profession] = pcc
    pval_dict[profession] = str(unrelated)

    model_frame.loc[model_frame['Profession'] == profession, 'profession-pcc'] = float(pcc)
    model_frame.loc[model_frame['Profession'] == profession, 'profession-p_value'] = str(unrelated)

  for profession in target_frame['Profession'].unique():
    if profession in pcc_dict:
      target_frame.loc[target_frame['Profession'] == profession, 'profession-pcc'] = pcc_dict[profession]
      target_frame.loc[target_frame['Profession'] == profession, 'profession-p_value'] = pval_dict[profession]
    else:
      target_frame.loc[target_frame['Profession'] == profession, 'profession-pcc'] = 'nA'
      target_frame.loc[target_frame['Profession'] == profession, 'profession-p_value'] = 'unknown'

  model_frame = model_frame.drop('Profession', axis=1)
  target_frame = target_frame.drop('Profession', axis=1)
  target_frame = target_frame.fillna(method = 'bfill')

  return [model_frame, target_frame]

# unify the way categorical n/a values are encoded
# e.g. gender has 'unknown'
def fillna_categorical(frame):
  for column in frame.select_dtypes(include='object'):
    frame[column] = frame[column].fillna('unknown')
  return frame

def preprocess(frame):
  frame = treat_gender(frame)
  frame = treat_work_experience(frame)
  frame = treat_additional_income(frame)
  frame = treat_year_of_record(frame)
  # Excel error we need to deal with
  frame = frame.replace('#NUM!', np.NaN)
  frame = fillna_categorical(frame)
  # see data/correlations/satisfation.txt
  frame['unhappy'] = frame['Satisfation with employer'] == 'Unhappy'
  frame['average'] = frame['Satisfation with employer'] == 'Average'
  # ensure no null values were introduced
  frame = frame.fillna(method = 'bfill')
  return frame

# process data
model_frame = preprocess(model_frame)
target_frame = preprocess(target_frame)

model_frame, target_frame = treat_country(model_frame, target_frame)
model_frame, target_frame = treat_profession(model_frame, target_frame)

# feature select
target_columns = ['Work Experience in Current Job [years]','Year of Record', 'Gender', 'Crime Level in the City of Employement', 
                  'country-pcc', 'country-p_value', 'Age', 'University Degree', 'Size of City', 
                  'Yearly Income in addition to Salary (e.g. Rental Income)', 'profession-pcc', 'profession-p_value',
                  # see data/correlations/ wexp.txt, additionalincome.txt, satisfation.txt
                  'significant-work-experience', 'high-additional-income', 'unhappy', 'average']

selected_features = model_frame[target_columns]
# scale our target variable
income = model_frame['Total Yearly Income [EUR]'].apply(np.log).values

# build GridSearch Object and regression pipeline
gscv = GridSearchCV(estimator = LGBMRegressor(random_state=15000, num_leaves=4200),
                    param_grid = { 'n_estimators': (400, 800), 'max_depth': (4, 8, 12) }, 
                    n_jobs = -1, cv = 5, verbose=1, scoring='neg_mean_absolute_error')

regr = Pipeline(steps=[('enc', TargetEncoder()),
                       ('grid', gscv)])

X_train, X_test, Y_train, Y_test = train_test_split(selected_features, income, train_size = 0.8)

regr.fit(X_train, Y_train)

y_predict = np.exp(regr.predict(target_frame[target_columns]))
# print local score
print(metrics.mean_absolute_error(np.exp(Y_test), np.exp(regr.predict(X_test))))

# Instances saved to separate file for ease of access
instances = pd.read_csv('data/instances.csv')['Instance'].values
f = open("data/submission.csv", "w")

# Write to File
f.write("Instance,Total Yearly Income [EUR]\n")

for i in range(len(y_predict)):
  f.write(str(instances[i]) + "," + str(y_predict[i]) + "\n")
