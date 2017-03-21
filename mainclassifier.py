# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn import model_selection

class XGBoost_Classifier:
    '''
    The main classifier
    '''
    def __init__(self,_num_rounds,_eta=0.1,_gamma=0,_max_depth=6,_min_child_weight=1,_max_delt_step=0,
                 _subsample=0.7,_colsample_bytree=0.7,_scale_position_weight=0,
                 _objective='multi:softprob',_num_class = 3,_eval_metric='mlogloss',
                 _seed=0,_silent=1):
        self.param = {}
        self.param['eta'] = _eta
        self.param['gamma'] = _gamma
        self.param['max_depth'] = _max_depth
        self.param['min_child_weight'] = _min_child_weight
        self.param['max_delt_step'] = _max_delt_step
        self.param['subsample']= _subsample
        self.param['colsample_bytree'] = _colsample_bytree
        self.param['scale_position_weght'] = _scale_position_weight
        self.param['objective'] = _objective
        self.param['num_class'] = _num_class
        self.param['eval_metric'] = _eval_metric
        self.param['seed'] = _seed
        self.param['silent'] = _silent
        self.num_rounds = _num_rounds
        
    def train(self,x_train,y_train):
        xgmat_train = xgb.DMatrix(x_train,label=y_train)
        param_list = list(self.param.items())
        watch_list = [(xgmat_train,'train')]
        self.model = xgb.train(param_list,xgmat_train,self.num_rounds)
        return self.model
    
    def predict(self,x_test):
        xgmat_test = xgb.DMatrix(x_test)
        y_predict = self.model.predict(xgmat_test)
        return y_predict
    
    def predict_logloss(self,x_test,y_test):
        xgmat_test = xgb.DMatrix(x_test)
        y_predict = self.model.predict(xgmat_test)
        logloss = log_loss(y_test,y_predict)
        return y_predict,logloss
    
    
    
    
def target_trans(df):
    target_map = {'high':0,'medium':1,'low':2}
    df['target'] = df['interest_level'].apply(lambda x: target_map[x])
    return df


def feature_trans_create_time(df,used_columns):
    '''
    df is DateFrame
    used_columns is the column list which will be used to train model
    '''
    create_date = pd.to_datetime(df['created'])
    df['year'] = create_date.dt.year
    df['month'] = create_date.dt.month
    df['day'] = create_date.dt.day
    if 'created' in used_columns:
        used_columns.remove('created')
    used_columns += ['year','month','day']
    return df,used_columns

def feature_trans_features(df,used_columns):
    df['num_features'] = df['features'].apply(lambda x: len(x))
    if 'features' in used_columns:
        used_columns.remove('features')
    used_columns.append('num_features')
    return df,used_columns



    


def run_model(path_train,path_test,path_save = None):
    df_train = pd.read_json(path_train)
    df_test = pd.read_json(path_test)
    used_columns=['price','bathrooms','bedrooms','created','features','latitude'
                  ,'longitude','price']
    df_train = target_trans(df_train)
    df_list = []
    for df in [df_train,df_test]:
        df,used_columns = feature_trans_create_time(df,used_columns)
        df,used_columns = feature_trans_features(df,used_columns)
        df_list.append(df)
        
    df_train,df_test = df_list
    print(df_test.shape)
    kf = model_selection.KFold(3)
    classifier = XGBoost_Classifier(100)
    used_columns = list(set(used_columns))
    for train,test in kf.split(df_train):
        x_train = df_train[used_columns].iloc[train]
        y_train = df_train['target'].iloc[train]
        x_test = df_train[used_columns].iloc[train]
        y_test = df_train['target'].iloc[train]
        print(x_train.shape,y_train.shape)
        classifier.train(x_train,y_train)
        y_predict,loss = classifier.predict_logloss(x_test,y_test)
        print('curloss = %.9f' % loss)
        
    if path_save:
        final_predict = classifier.predict(df_test[used_columns])
        print(df_test.count())
        df_result = pd.DataFrame(final_predict)
        df_result.columns = ["high", "medium", "low"]
        print(df_test.count())
        df_result['listing_id'] = df_test.listing_id.values
        print(df_result.count())
        print(df_test.count())
        df_result.to_csv(path_save)



if __name__ == '__main__':
    path_train = 'data/train.json'
    path_test = 'data/test.json'
    path_save = 'data/submission.csv'
    run_model(path_train,path_test,path_save)
    
        
    

        
        
        
        
        
        
        
        