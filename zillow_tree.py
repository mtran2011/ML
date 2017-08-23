# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import xgboost as xgb
from sklearn.model_selection import train_test_split

print('Loading data...')
prop_df = pd.read_csv('../input/properties_2016.csv')
print('Finished loading properties...')
train_df = pd.read_csv('../input/train_2016_v2.csv')
samplesub_df = pd.read_csv('../input/sample_submission.csv')
print('Finished loading data')

print('Binding to float32')
for col, dtype in zip(prop_df.columns, prop_df.dtypes):
    if dtype == np.float64:
        prop_df[col] = prop_df[col].astype(np.float32)
print('Finished binding to float32')

train_df = train_df.merge(prop_df, how='inner', on='parcelid')

missing_cnt = train_df.isnull().sum(axis=0).sort_values(ascending=False)
print('Count of missing values')
print(missing_cnt)

# delete all features that have NA
train_df = train_df[missing_cnt[missing_cnt==0].index]

x_train = train_df.drop(['logerror', 'roomcnt', 'parcelid'], axis=1, inplace=False)
y_train = train_df['logerror']

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)

print('Building DMatrix...')
