'''
Based on data exploration (e.g. a gap between city size ), 
this lightgbm model used same data processing methods as one of the best models from competition 1
E.g., impute numeric columns with its mean value, and categorical values with mode.
'''
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv("/Data/train.csv")

test = pd.read_csv("/Data/test.csv")

rename_cols = {"Total Yearly Income [EUR]":'Income'}

train = train.rename(columns=rename_cols)

data = pd.concat([train,test],ignore_index=True)

fill_col_dict = {'Year of Record': 1999.0,
 'Gender':'female',
 'Age': 15,
 'Profession': 'principal administrative associate',
 'University Degree': 0,
 'Hair Color': 'Black'}
for col in fill_col_dict.keys():
    data[col] = data[col].fillna(fill_col_dict[col])

def create_cat_con(df,cats,cons,normalize=True):
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict()
        nm = cat + '_FE_FULL'
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
            new_col = cat +'_'+ con
            print('timeblock frequency encoding:', new_col)
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col]/df[cat+'_FE_FULL']
    return df

cats = ['Year of Record', 'Gender', 'Country',
        'Profession', 'University Degree','Wears Glasses',
        'Hair Color','Age', 
        'Satisfation with employer', 'Housing Situation', 'Crime Level in the City of Employement',
        'Work Experience in Current Job [years]']
cons = ['Size of City','Body Height [cm]']

data = create_cat_con(data,cats,cons)

# %% [code]
for col in train.dtypes[train.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))

del_col = set(['Income','Instance'])
features_col =  list(set(data) - del_col)

# %% [code]
X_train,X_test = data[features_col].iloc[:1048574],data[features_col].iloc[1048574:]
Y_train = data['Income'].iloc[:1048574]
X_test_id = data['Instance'].iloc[1048574:]
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)

# %% [code]
params = {
          'max_depth': 20,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": 100,
          "n_jobs": 4,
         }
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
# test_data = lgb.Dataset(X_test)
clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
pre_test_lgb = clf.predict(X_test)

# %% [code]
from sklearn.metrics import mean_absolute_error
pre_val_lgb = clf.predict(x_val)
val_mae = mean_absolute_error(y_val,pre_val_lgb)


sub_df = pd.DataFrame({'Instance':X_test_id,
                       'Total Yearly Income [EUR]':pre_test_lgb})
sub_df.head()

sub_df.to_csv("/Data/subLGB_1.csv",index=False)

