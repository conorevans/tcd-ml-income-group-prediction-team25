from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor, LinearRegression
import category_encoders as ce

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge

import statistics
import pandas as p
import numpy as np

trainData= p.read_csv('C:\\Users\\Ryan BARRON\\Desktop\\CS_Fourth_Year\\CS4061_Machine_Learning\\Kaggle_Comptetion_Part1\\tcd ml 2019-20 income prediction training (with labels).csv')
trainData = trainData.drop("Instance", axis=1)

target= "Income in EUR"
cols = trainData.columns

# Setting my labels and target variables
x = trainData.drop(target, axis=1)
y = trainData[target]





# Impute missing year of record with median
yorImputer = SimpleImputer(strategy="mean")
yorScale = PowerTransformer("box-cox")
x[cols[0]] = yorScale.fit_transform(yorImputer.fit_transform(x[cols[0]] .to_frame()))


# Make gender consistent, replace 0, unknown and nan with same label, unknown
x[cols[1]] = x[cols[1]].replace(to_replace="0", value="unknown").fillna("unknown")
genderEncoder = LabelEncoder()
x[cols[1]] = genderEncoder.fit_transform(x[cols[1]].to_frame())

# Age
ageImputer = SimpleImputer(strategy="mean")
ageScaler = PowerTransformer("box-cox")
x[cols[2]] = ageScaler.fit_transform(ageImputer.fit_transform(x[cols[2]].to_frame()))

# Country
countryEncoder =LabelEncoder()
x[cols[3]] = countryEncoder.fit_transform(x[cols[3]].fillna("unknown").to_frame())

# Size of city
cityImputer = SimpleImputer(strategy="mean")
cityScaler = PowerTransformer("box-cox")
x[cols[4]] = cityScaler.fit_transform(cityImputer.fit_transform(x[cols[4]].to_frame()))

# Profession
professionEncoder = LabelEncoder()
x[cols[5]] = professionEncoder.fit_transform(x[cols[5]].fillna("unknown").to_frame())

# University
x[cols[6]] = x[cols[6]].replace(to_replace="0", value="No").fillna("unknown")
universityEncoder = LabelEncoder()
x[cols[6]] = universityEncoder.fit_transform(x[cols[6]].to_frame())

# Glasses
# donothing

# Hair color
x[cols[8]] = x[cols[8]].fillna("Unknown")
hairEncoder = LabelEncoder()
x[cols[8]] = hairEncoder.fit_transform(x[cols[8]].to_frame())

# Body height
heightImputer = SimpleImputer(strategy="mean")
heightScaler = PowerTransformer("box-cox")
x[cols[9]] = heightScaler.fit_transform(heightImputer.fit_transform(x[cols[9]].to_frame()))

# Spliting my data into training data and validation data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3)



# numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant',fill_value='missing')),
#                                                 ('encoder', ce.TargetEncoder())])

# numeric_features = trainData.select_dtypes(include=['int64', 'float64']).drop([target], axis=1).columns
# categorical_features = trainData.select_dtypes(include=['object']).columns


# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])

regressions = [ 
                AdaBoostRegressor(RandomForestRegressor(random_state=0))
                 ]

# for regressor in regressions:
#     pip = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('Regression', regressor)])
#     pip.fit(xT, yT)
#     print(regressor)
#     print("model score: %.3f" % pip.score(xTest, yTest))                  

for regressor in regressions:
    regressor.fit(xTrain, yTrain)
    print(regressor)
    print("model score: %.3f" % regressor.score(xTest, yTest))                  


# sgd = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('Regression', SGDRegressor())])
# sgd.fit(xT, yT)
# score = sgd.score(xTest, yTest)
# print(score)

