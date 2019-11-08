import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer

model_frame = pd.read_csv('train.csv').head(250000)

fig, ax = plt.subplots()
target = model_frame['Total Yearly Income [EUR]']
for column in model_frame.drop(columns = ['Instance', 'Total Yearly Income [EUR]']).select_dtypes(exclude=['object']):
  print(column)
  print(stats.pearsonr(model_frame[column].fillna(method='ffill'), target))

  ax.plot(model_frame[column], target)

  ax.set(xlabel = column, ylabel = 'Income')
  ax.grid()

  fig.savefig(f'{column}.png')