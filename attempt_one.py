import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def fill_null(df):
    for i in df:
        if df[i].dtype.kind in 'biufc':
            df[i] = df[i].replace(np.nan,0)
        else:
            df[i] = df[i].replace(np.nan,'0')
    return df
def filter_zero(df):
    df['isValuable'] = 'Yes'
    for i in df:
        if df[i].dtype.kind in 'biufc':
            df['isValuable'] = np.where(df[i] == 0, 'No' ,df['isValuable'])
        if df[i].dtype.kind not in 'biufc':
            df['isValuable'] = np.where(df[i] == '0', 'No' ,df['isValuable'])
    return df

# Importing the dataset
df = pd.read_csv('tcd-ml-comp-201920-income-pred-group/train.csv')
df = fill_null(df)

# Remove possible noises
df['Housing Situation'] = np.where(df['Housing Situation'] == 'nA','0', np.where(df['Housing Situation'] == 0, '0', df['Housing Situation']))
df['Hair Color'] = np.where(df['Hair Color'] == 'unknown', '0',df['Hair Color'])
df['Gender'] = np.where(df['Gender'] == 'unknown', '0', np.where(df['Gender'] == 'other', '0', df['Gender']))
df['University Degree'] = np.where(df['University Degree'] == '0', 'No', df['University Degree'])
# impute data 0s so that they won't be filtered out
df['Wears Glasses'] = df['Wears Glasses'].apply(str)
df['Wears Glasses'] = np.where(df['Wears Glasses'] == '0', 'No',np.where(df['Wears Glasses'] == '1', 'Yes',df['Wears Glasses']))
df = filter_zero(df)

df2 = df.copy()
cat_cols = df.select_dtypes(include = 'object').columns[:-1]

for i in cat_cols:
    mean_val = df2.groupby(i)['Total Yearly Income [EUR]'].mean()
    df2.loc[:, i] = df2[i].map(mean_val)

df2 = df2[df2['isValuable'] == 'Yes']
df2 = df2.drop(columns = ['isValuable','Instance'])

#Prepare trainning data
Y = df2['Total Yearly Income [EUR]']
X = df2.drop('Total Yearly Income [EUR]', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
rfs = RandomForestRegressor(n_estimators=100)
rfs = rfs.fit(X_train, Y_train)
rfs.score(X_test,Y_test) # get a score of 0.8412

print(metrics.mean_absolute_error(Y_test, rfs.predict(X_test)))

'''Use the model to make predictions'''
# Process process the test data exactly the same as the test data
dfSubmission = pd.read_csv('tcd-ml-comp-201920-income-pred-group/test.csv')
dfSubmission = fill_null(dfSubmission)

dfSubmission['Housing Situation'] = np.where(dfSubmission['Housing Situation'] == 'nA','0', np.where(dfSubmission['Housing Situation'] == 0, '0', dfSubmission['Housing Situation']))
dfSubmission['Hair Color'] = np.where(dfSubmission['Hair Color'] == 'unknown', '0',dfSubmission['Hair Color'])
dfSubmission['Gender'] = np.where(dfSubmission['Gender'] == 'unknown', '0',np.where(dfSubmission['Gender'] == 'other', '0', dfSubmission['Gender']))
df['University Degree'] = np.where(df['University Degree'] == '0', 'No', df['University Degree'])
dfSubmission['Wears Glasses'] = dfSubmission['Wears Glasses'].apply(str)
dfSubmission['Wears Glasses'] = np.where(dfSubmission['Wears Glasses'] == '0', 'No', np.where(dfSubmission['Wears Glasses'] == '1', 'Yes',dfSubmission['Wears Glasses']))

dfSubmission = filter_zero(dfSubmission)
dfSubmission2 = dfSubmission.copy()

cat_cols = dfSubmission2.select_dtypes(include = 'object').columns[:-1]
for i in cat_cols:
    mean_val = df.groupby(i)['Total Yearly Income [EUR]'].mean()
    dfSubmission2.loc[:, i] = dfSubmission2[i].map(mean_val)

dfSubmission2 = dfSubmission2.drop(['isValuable','Instance'], axis = 1)
dfSubmission2.head()

x = dfSubmission2.columns[:-1]
x_submission = dfSubmission2[x]
x_submission = x_submission.fillna(0)
x_submission['Total Yearly Income [EUR]'] = rfs.predict(x_submission)


# Save the result to the submission file
submissionFile = 'tcd-ml-comp-201920-income-pred-group/submission.csv'
submission = pd.read_csv(submissionFile)
tmp = x_submission['Total Yearly Income [EUR]']
submission['Income'] = tmp
submission.to_csv(submissionFile, index=False)