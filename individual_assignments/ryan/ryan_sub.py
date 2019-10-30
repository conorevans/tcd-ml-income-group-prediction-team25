from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor, LinearRegression
import category_encoders as ce

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge

import statistics
import pandas as p
import numpy as np

# Open train data, test data and the submission csv
trainData= p.read_csv('C:\\Users\\Ryan BARRON\\Desktop\\CS_Fourth_Year\\CS4061_Machine_Learning\\Kaggle_Comptetion_Part1\\tcd ml 2019-20 income prediction training (with labels).csv')
trainData = trainData.drop("Instance", axis=1)
testData= p.read_csv('C:\\Users\\Ryan BARRON\\Desktop\\CS_Fourth_Year\\CS4061_Machine_Learning\\Kaggle_Comptetion_Part1\\tcd ml 2019-20 income prediction test (without labels).csv')
testData = testData.drop("Instance", axis=1)
submitData= p.read_csv('C:\\Users\\Ryan BARRON\\Desktop\\CS_Fourth_Year\\CS4061_Machine_Learning\\Kaggle_Comptetion_Part1\\tcd ml 2019-20 income prediction submission file.csv')

cols = trainData.columns
trainTarget= "Income in EUR"
testTarget= "Income"

# Setting my labels and target variables
x = trainData.drop(trainTarget, axis=1)
y = trainData[trainTarget]

xTest = testData.drop(testTarget, axis=1)

# Impute missing year of record with mean and apply a power transform to the train data and test data
yorImputer = SimpleImputer(strategy="mean")
yorScale = PowerTransformer(method="box-cox")
x[cols[0]] = yorScale.fit_transform(yorImputer.fit_transform(x[cols[0]].to_frame()))
xTest[cols[0]] = yorScale.transform(yorImputer.transform(xTest[cols[0]].to_frame()))

# Make gender consistent, replace 0, unknown and nan with same label, unknown. Then use a target encoder for both train and test 
x[cols[1]] = x[cols[1]].replace(to_replace="0", value="unknown").fillna("unknown")
genderEncoder = ce.TargetEncoder()
x[cols[1]] = genderEncoder.fit_transform(x[cols[1]].to_frame(), y)
xTest[cols[1]] = genderEncoder.transform(xTest[cols[1]].to_frame())

# Age: Impute missing data with the median and apply a power transform
ageImputer = SimpleImputer(strategy="median")
ageScaler = PowerTransformer(method="box-cox")
x[cols[2]] = ageScaler.fit_transform(ageImputer.fit_transform(x[cols[2]].to_frame()))
xTest[cols[2]] = ageScaler.transform(ageImputer.transform(xTest[cols[2]].to_frame()))

# Country: Replace nan with unknown, then use a target encoder
countryEncoder = ce.TargetEncoder()
x[cols[3]] = countryEncoder.fit_transform(x[cols[3]].fillna("unknown").to_frame(), y)
xTest[cols[3]] = countryEncoder.transform(xTest[cols[3]].to_frame())

# Size of city: Imputing missing data with the median, then scale it with a power transform
cityImputer = SimpleImputer(strategy="median")
cityScaler = PowerTransformer(method="box-cox")
x[cols[4]] = cityScaler.fit_transform(cityImputer.fit_transform(x[cols[4]].to_frame()))
xTest[cols[4]] = cityScaler.transform(cityImputer.transform(xTest[cols[4]].to_frame()))

# Profession: Replace nan with unknown and use target encoder
professionEncoder = ce.TargetEncoder()
x[cols[5]] = professionEncoder.fit_transform(x[cols[5]].fillna("unknown").to_frame(), y)
xTest[cols[5]] = professionEncoder.transform(xTest[cols[5]].fillna("unknown").to_frame())

# University: Make 0s and No consistent as the same. Replace nans with unknown then use target encoder
x[cols[6]] = x[cols[6]].replace(to_replace="0", value="No").fillna("unknown")
universityEncoder = ce.TargetEncoder()
x[cols[6]] = universityEncoder.fit_transform(x[cols[6]].to_frame(), y)
xTest[cols[6]] = universityEncoder.transform(xTest[cols[6]].to_frame())

# Glasses: Already made of 1s and 0s, no need to alter
# :donothing: https://cdn.frankerfacez.com/emoticon/twitter_image/364795.png

# Hair color: Replace nans with unknown, then use target encoder
x[cols[8]] = x[cols[8]].fillna("Unknown")
hairEncoder = ce.TargetEncoder()
x[cols[8]] = hairEncoder.fit_transform(x[cols[8]].to_frame(), y)
xTest[cols[8]] = hairEncoder.transform(xTest[cols[8]].to_frame())

# Body height: Replace nans with the median, then use a power transform
heightImputer = SimpleImputer(strategy="median")
heightScaler = PowerTransformer(method="box-cox")
x[cols[9]] = heightScaler.fit_transform(heightImputer.fit_transform(x[cols[9]].to_frame()))
xTest[cols[9]] = heightScaler.transform(heightImputer.transform(xTest[cols[9]].to_frame()))

# Use RandomForest regressor with 100 estimators, fit on the training data and then predict using test features.
regressor = RandomForestRegressor(n_estimators=100)                     
regressor.fit(x, y)
yResult= regressor.predict(xTest)

# Write back to the submission csv
submitData["Income"] = yResult
submitData.to_csv("./tcd ml 2019-20 income prediction submission file.csv")