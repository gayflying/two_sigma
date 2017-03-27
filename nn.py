# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:47:04 2017

@author: Michael Hartman

This is a simple Keras NN
"""

# Original Address: https://www.kaggle.com/zeroblue/two-sigma-connect-rental-listing-inquiries/simple-starter-keras-nn/code

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.layers.advanced_activations import PReLU, ELU, ThresholdedReLU

label_column = 'interest_level'
num_classes = 3

start_time = time.time()

data_path =  "./"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train = pd.read_json(train_file)
test = pd.read_json(test_file)

# Make the label numeric
label_map = pd.Series({'low': 2, 'medium': 1, 'high': 0})
train[label_column] = label_map[train[label_column]].values

all_data = train.append(test)
all_data.set_index('listing_id', inplace=True)

print('Identify bad geographic coordinates')
all_data['bad_addr'] = 0
mask = ~all_data['latitude'].between(40.5, 40.9)
mask = mask | ~all_data['longitude'].between(-74.05, -73.7)
bad_rows = all_data[mask]
all_data.loc[mask, 'bad_addr'] = 1

print('Create neighborhoods')
# Replace bad values with mean
mean_lat = all_data.loc[all_data['bad_addr']==0, 'latitude'].mean()
all_data.loc[all_data['bad_addr']==1, 'latitude'] = mean_lat
mean_long = all_data.loc[all_data['bad_addr']==0, 'longitude'].mean()
all_data.loc[all_data['bad_addr']==1, 'longitude'] = mean_long
# From: https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/unsupervised-and-supervised-neighborhood-encoding
kmean_model = KMeans(42)
loc_df = all_data[['longitude', 'latitude']].copy()
standardize = lambda x: (x - x.mean()) / x.std()
loc_df['longitude'] = standardize(loc_df['longitude'])
loc_df['latitude'] = standardize(loc_df['latitude'])
kmean_model.fit(loc_df)
all_data['neighborhoods'] = kmean_model.labels_

#print('Distance from center')
lat = np.square(all_data['latitude'] - mean_lat)
lng = np.square(all_data['longitude'] - mean_long)
#all_data['dist_from_center'] = np.sqrt(lat + lng)

print('Fix Bathrooms')
mask = all_data['bathrooms'] > 9
all_data.loc[mask, 'bathrooms'] = 1

print('Break up the date data')
all_data['created'] = pd.to_datetime(all_data['created'])
#all_data['year'] = all_data['created'].dt.year
all_data['month'] = all_data['created'].dt.month
all_data['day_of_month'] = all_data['created'].dt.day
all_data['weekday'] = all_data['created'].dt.dayofweek
#all_data['day_of_year'] = all_data['created'].dt.dayofyear
all_data['hour'] = all_data['created'].dt.hour

all_data['count_feat'] = all_data['features'].apply(len)
all_data['count_desc'] = all_data['description'].str.split().apply(len)

all_data['addr_has_number'] = all_data['display_address'].str.split().str.get(0)
is_digit = lambda x: str(x).isdigit()
all_data['addr_has_number'] = all_data['addr_has_number'].apply(is_digit)

print('Bed and bath features')
all_data['bedrooms'] += 1
all_data['bed_to_bath'] = all_data['bathrooms'] 
all_data['bed_to_bath'] /= all_data['bedrooms']
all_data['price_per_bed'] = all_data['price'] / all_data['bedrooms']
bath = all_data['bathrooms'].copy()
bath.loc[all_data['bathrooms']==0] = 1
all_data['price_per_bath'] = all_data['price'] / bath
# Half baths are not interesting
# See https://www.kaggle.com/arnaldcat/two-sigma-connect-rental-listing-inquiries/a-proxy-for-sqft-and-the-interest-on-1-2-baths/notebook
all_data['half_bath'] = all_data['bathrooms'] == all_data['bathrooms'] // 1

all_data['rooms'] = all_data['bathrooms'] * 0.5 + all_data['bedrooms']
all_data['price_per_room'] = all_data['price'] / all_data['rooms']

print('Create ratios')
median_list = ['bedrooms', 'bathrooms', 'rooms']
median_list = ['bedrooms', 'bathrooms', 'building_id', 'rooms', 'neighborhoods']
for col in median_list:
    median_price = all_data[[col, 'price']].groupby(col)['price'].median()
    median_price = median_price[all_data[col]].values.astype(float)
    all_data['median_' + col] = median_price
    all_data['ratio_' + col] = all_data['price'] / median_price
    all_data['median_' + col] = np.log(all_data['median_' + col].values)

#print('Additional medians and ratios')
median_list = [c for c in all_data.columns if c.startswith('median_')]
all_data['median_mean'] = all_data[median_list].mean(axis=1)
ratio_list = [c for c in all_data.columns if c.startswith('ratio_')]
all_data['ratio_mean'] = all_data[ratio_list].mean(axis=1)
    
print('Normalize the price')
all_data['price'] = np.log(all_data['price'].values)

print('Building counts')
bldg_count = all_data['building_id'].value_counts()
bldg_count['0'] = 0
all_data['bldg_count'] = np.log1p(bldg_count[all_data['building_id']].values)
all_data['zero_bldg'] = all_data['building_id']=='0'

print('Manager counts')
mgr_count = all_data['manager_id'].value_counts()
all_data['mgr_count'] = np.log1p(mgr_count[all_data['manager_id']].values)

#Scale features
scaler = StandardScaler()
cols = [c for c in all_data.columns]
scale_keywords = ['price', 'count', 'ratio', '_to_', 
                  'day_', 'hour', 'median', 'longitude', 'latitude']
scale_list = [c for c in cols if any(w in c for w in scale_keywords)]
print('Scaling features:', scale_list)
all_data[scale_list] = scaler.fit_transform(all_data[scale_list].astype(float))

print('Create dummies')
mask = all_data['bathrooms'] > 3
all_data.loc[mask, 'bathrooms'] = 4
mask = all_data['bedrooms'] >= 5
all_data.loc[mask, 'bedrooms'] = 5
mask = all_data['rooms'] >= 6
all_data.loc[mask, 'rooms'] = 6
cat_cols = ['bathrooms', 'bedrooms', 'month', 'weekday', 'rooms', 
            'neighborhoods']
#cat_cols = ['bathrooms', 'bedrooms', 'month', 'weekday', 'rooms']
for col in cat_cols:
    dummy = pd.get_dummies(all_data[col], prefix=col)
    dummy = dummy.astype(bool) 
    all_data = all_data.join(dummy)
all_data.drop(cat_cols, axis=1, inplace=True)

print('Drop columns')
drop_cols = ['description', 'photos', 'display_address', 'street_address', 
             'features', 'created', 'building_id', 'manager_id', 
             'longitude', 'latitude'
             ]
             
all_data.drop(drop_cols, axis=1, inplace=True)

data_columns = all_data.columns.tolist()
data_columns.remove(label_column)

mask = all_data[label_column].isnull()
train = all_data[~mask].copy()
test = all_data[mask].copy()

elapsed = (time.time() - start_time)
print('Data loaded and prepared in:', timedelta(seconds=elapsed))

def nn_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(len(data_columns),)))
    model.add(Dense(128,  
                    #activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
                    kernel_constraint=max_norm(2.0),
                    input_shape = (len(data_columns),),))
#    model.add(Dropout(0.25))
#    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(64,  
                    #activation='softplus',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
                    kernel_constraint=max_norm(2.0),
                    ))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.5))
    
    
    model.add(Dense(16,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.000025),
                    kernel_constraint=max_norm(2.0)
                    ))
#    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dense(32,
                    #activation='softplus', 
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(0.00005),
                    kernel_constraint=max_norm(2.0)
                    ))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(0.25))
    

    model.add(Dense(units=num_classes, 
                    activation='softmax', 
                    kernel_initializer='he_normal',
                    ))
    
#    opt = optimizers.Adadelta(lr=1)
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='nadam',
                  metrics=['accuracy']
                  )
    return(model)
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
model = nn_model()
X = train[data_columns].values
y = train[label_column].values
model.fit(X, y, epochs = 40, batch_size=1024, verbose = 2, callbacks=[earlyStopping], validation_split=0.2)

train_pred = model.predict_proba(X)
#Normalize the predictions
pred_sum = train_pred.sum(axis=1)
train_pred = train_pred / pred_sum[:, None]
score = log_loss(y, train_pred)
print('Score:', score)

test_pred = model.predict_proba(test[data_columns].values)
#Normalize the predictions
pred_sum = test_pred.sum(axis=1)
test_pred = test_pred / pred_sum[:, None]
test_out = pd.DataFrame(test_pred, columns = ['high', 'medium', 'low'], index=test.index)
test_out.to_csv('output.csv')

elapsed = (time.time() - start_time)
print('Completed in:', timedelta(seconds=elapsed))