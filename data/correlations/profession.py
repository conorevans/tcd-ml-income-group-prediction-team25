import pandas as pd
import numpy as np
import scipy.stats as stats

frame = pd.read_csv('data/train.csv')

f = open("data/correlations/profession.txt", "w")
frame['Profession'] = frame['Profession'].fillna(method='bfill')
for profession in frame['Profession'].unique():
  p_score, z_score = stats.pearsonr(frame['Profession'] == profession, frame['Total Yearly Income [EUR]'])
  z_score = z_score > 0.05
  f.write(str(profession) + "\n")
  f.write(str(p_score) + "\n")
  f.write(str(z_score) + "\n")


