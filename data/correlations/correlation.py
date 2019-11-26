import pandas as pd
import numpy as np
import scipy.stats as stats

frame = pd.read_csv('data/train.csv')

f = open("data/correlations/soc2.txt", "w")

# get column X
frame = frame.fillna(method='bfill')

# set arbitrary boundaries
# for floats, use logical boundaries then tweak based on observed trends
# for categorical vars, use .unique() and run for each value
boundaries = [0, 3000, 5000, 10000, 50000, 100000, 1000000]

# for each boundary
for boundary in boundaries:
#for boundary in frame['x'].unique():
  # get correlation for column X and income
  # == / <= / >=
  pcc, p_value = stats.pearsonr(frame['Size of City'] <= boundary, frame['Total Yearly Income [EUR]'])
  p_value = p_value > 0.05
  f.write(str(boundary) + "\n")
  f.write('-=--==--==-=-=-\n')
  f.write(str(pcc) + "\n")
  f.write(str(p_value) + "\n")


