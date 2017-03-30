# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

current train loss is 0.547081834   rank=1047/1546
current train loss is 0.455862868   rank=更差了！！！

"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn import model_selection
from sklearn.cluster import KMeans
import kneighbors as kn



class XGBoost_Classifier:
    '''
    The main classifier
    '''
    def __init__(self,_num_rounds,_eta=0.05,_gamma=0,_max_depth=10,_min_child_weight=1,_max_delta_step=0,
                 _subsample=0.5,_colsample_bytree=0.5,_scale_position_weight=0,
                 _objective='multi:softprob',_num_class = 3,_eval_metric='mlogloss',
                 _seed=0,_silent=1):
        self.param = {}
        self.param['eta'] = _eta
        self.param['gamma'] = _gamma
        self.param['max_depth'] = _max_depth
        self.param['min_child_weight'] = _min_child_weight
        self.param['max_delta_step'] = _max_delta_step
        self.param['subsample']= _subsample
        self.param['colsample_bytree'] = _colsample_bytree
        self.param['lambda'] = 2
        self.param['scale_position_weght'] = _scale_position_weight
        self.param['objective'] = _objective
        self.param['num_class'] = _num_class
        self.param['eval_metric'] = _eval_metric
        self.param['seed'] = _seed
        self.param['silent'] = _silent
        self.param['updater'] = 'grow_gpu'
        self.num_rounds = _num_rounds
        
    def train(self,x_train,y_train, eval_list=(), early_stopping=None):
        # x_train = preprocessing.scale(x_train)
        xgmat_train = xgb.DMatrix(x_train,label=y_train)
        param_list = list(self.param.items())
        watch_list = [(xgmat_train,'train')]
        self.model = xgb.train(param_list,xgmat_train,self.num_rounds, evals=eval_list, early_stopping_rounds=early_stopping)
        return self.model
    
    def predict(self,x_test):
        # x_test = preprocessing.scale(x_test)
        xgmat_test = xgb.DMatrix(x_test)
        y_predict = self.model.predict(xgmat_test)
        return y_predict
    
    def predict_logloss(self,x_test,y_test):
        # x_test = preprocessing.scale(x_test)
        xgmat_test = xgb.DMatrix(x_test)
        y_predict = self.model.predict(xgmat_test)
        logloss = log_loss(y_test,y_predict)
        return y_predict,logloss
    
    
#对label的处理
def target_trans(df):
    target_map = {'high':0,'medium':1,'low':2}
    df['target'] = df['interest_level'].apply(lambda x: target_map[x])
    return df

#取了一下平均价格
def feature_trans_bed(df,used_columns):
    df["pricePerBed"] = df['price'] / (df['bedrooms']+0.1)
    df["pricePerBath"] = df['price'] / (df['bathrooms']+0.1)
    df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms']+0.1)
    # removed = ['bedrooms','bathrooms']
    # for s in removed:
    #     if s in used_columns:used_columns.remove(s)
    used_columns.extend(['bedrooms','bathrooms','pricePerBed','pricePerBath','pricePerRoom'])
    return df,used_columns

#处理坐标，使用KNN将地理坐标变为该处预期价格
def feature_trans_location(neigh, df,used_columns):
    print("calculating locationValue")
    for key in df['latitude'].keys():
        df.at[key,'locationValue'] = neigh.predict([[df['latitude'][key],df['longitude'][key]]])
    df['netLocationValue'] = df['locationValue'] - df['price'] #溢价
#    if 'latitude' in used_columns:
#        used_columns.remove('latitude')
#    if 'longitude' in used_columns:
#        used_columns.remove('longitude')
    used_columns += ['locationValue']
    used_columns += ['netLocationValue']
    print("locationValue done!")
    return df,used_columns

#对建立时间的简单分解
def feature_trans_create_time(df,used_columns):
    '''
    df is DateFrame
    used_columns is the column list which will be used to train model
    '''
    create_date = pd.to_datetime(df['created'])
    df['year'] = create_date.dt.year
    df['month'] = create_date.dt.month
    df['day'] = create_date.dt.day
    df['dayofweek'] = create_date.dt.dayofweek
    if 'created' in used_columns:
        used_columns.remove('created')
    used_columns += ['year','month','day', 'dayofweek']
    return df,used_columns

#没有用
def feature_trans_features(df,used_columns):
    df['num_features'] = df['features'].apply(lambda x: len(x))
    if 'features' in used_columns:
        used_columns.remove('features')
    used_columns.append('num_features')
    return df,used_columns

#对manager_id的处理，但是测试效果不好 不用
def feature_trans_manager_id(df_train,df_test,used_columns):
    df1 = df_train.groupby(['manager_id', 'interest_level']).count().reset_index()
    df2 = df1.pivot(index='manager_id', columns='interest_level', values='listing_id').fillna(0)
    df2['count'] = df2['high'] + df2['low'] + df2['medium']
    df2['high_ratio'] = df2['high'] / df2['count']
    df2['medium_ratio'] = df2['medium'] / df2['count']
    # df2['low_ratio'] = df2['low'] / df2['count']
    del df2['high']
    del df2['medium']
    del df2['low']
    df_train = pd.merge(df_train, df2, how='left',left_on='manager_id', right_index=True)
    df_train['high_ratio'] = df_train['high_ratio'].fillna(0.3333333333)
    df_train['medium_ratio'] = df_train['medium_ratio'].fillna(0.3333333333)
    df_test = pd.merge(df_test,df2,how='left',left_on='manager_id',right_index=True)
    df_test['high_ratio'] = df_test['high_ratio'].fillna(0.3333333333)
    df_test['medium_ratio'] = df_test['medium_ratio'].fillna(0.3333333333)

    if 'manager_id' in used_columns:
        used_columns.remove('manager_id')
    used_columns.extend(['high_ratio','medium_ratio'])
    return df_train,df_test,used_columns

def feature_trans_manager_id_2(df_train,df_test,used_clumns):
    from managerhandler import manager_skill
    trans = manager_skill()
    # First, fit it to the training data:
    trans.fit(df_train, df_train['target'])
    # Now transform the training data
    X_train_transformed = trans.transform(df_train)
    # You can also do fit and transform in one step:
    X_val_transformed = trans.transform(df_test)
    used_clumns.append('manager_skill')
    return X_train_transformed,X_val_transformed,used_clumns

def feature_kmeans(df_train, df_test, used_columns):
    all_data = df_train.append(df_test)
    
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
    used_columns.extend(['bad_addr', 'neighborhoods'])
    mask = all_data['interest_level'].isnull()
    train = all_data[~mask].copy()
    test = all_data[mask].copy()
    return train, test, used_columns


#阿东提取的features特征
def feature_trans_features_again(df,used_columns):

    def clean(s):
        k=4
        x = s.replace("-", "")
        x = x.replace(" ", "")
        x = x.replace("twenty four hour", "24")
        x = x.replace("24/7", "24")
        x = x.replace("24hr", "24")
        x = x.replace("24-hour", "24")
        x = x.replace("24hour", "24")
        x = x.replace("24 hour", "24")
        x = x.replace("common", "cm")
        x = x.replace("concierge", "doorman")
        x = x.replace("bicycle", "bike")
        x = x.replace("private", "pv")
        x = x.replace("deco", "dc")
        x = x.replace("decorative", "dc")
        x = x.replace("onsite", "os")
        x = x.replace("outdoor", "od")
        x = x.replace("ss appliances", "stainless")
        x = x[:k].strip()
        return x
    # train_df = pd.read_json("data/train.json")
    # test_df = pd.read_json("data/test.json")
    check=pd.read_csv("data/check.txt",encoding='gbk')
    used_columns += list(check['key'])
    fd= df
    fd["features"] = fd[["features"]].apply(lambda _: [list(map(str.strip, map(str.lower, x))) for x in _])
    fd["features"]=fd["features"].apply(lambda  p: [clean(x) for x in p ])

    cooo=check["key"]
    def think(x,cooo):
        kof=[]
        for i in cooo:
            if i in x:
                kof.append(1)
            else:
                kof.append(0)
        return kof

    fd["ol"]=fd["features"].apply(lambda  x:[ think(x,cooo)])
    long1=len(fd["features"])
    long2=len(cooo)

    a=[]
    for j in fd["ol"]:
            a.append(j[0])
    b=np.array(a)
    for i in range(long2):
       
         fd[str(cooo[i])]=b[:,i]

    return df,used_columns
    


def run_model(path_train,path_test,path_save = None,kfold = 0):
    df_train = pd.read_json(path_train)
    df_test = pd.read_json(path_test)
    print('load data success!')
    used_columns=['latitude'
                  ,'longitude','price']
    df_train = target_trans(df_train)
    df_list = []
    neigh = kn.PositionValueProphet(20)
    for df in [df_train,df_test]:
        df,used_columns = feature_trans_create_time(df,used_columns)
        df,used_columns = feature_trans_features_again(df,used_columns)
        df,used_columns = feature_trans_bed(df,used_columns)
#        df,used_columns = feature_trans_location(neigh, df, used_columns)
        # Temp feature, Description Length
        df['length'] = df['description'].str.split().apply(len)
        used_columns.extend(['length'])
        
        df_list.append(df)
        
    df_train,df_test = df_list

    df_train,df_test,used_columns = feature_trans_manager_id_2(df_train,df_test,used_columns)
    df_train, df_test, used_columns = feature_kmeans(df_train, df_test, used_columns)
    #used_columns.remove('longitude')
    #used_columns.remove('latitude')

    classifier = XGBoost_Classifier(2000,_eta=0.02,_gamma=0,_max_depth=10,_min_child_weight=2,_max_delta_step=1,
                 _subsample=0.5,_colsample_bytree=0.5,_scale_position_weight=0,
                 _objective='multi:softprob',_num_class = 3,_eval_metric='mlogloss',
                 _seed=0,_silent=1)
    used_columns = list(set(used_columns))
    print('features engeerning done!..')
    print('used features is ', used_columns)

    def train_predict(train,test):
        print('start to train...')
        x_train = df_train[used_columns].iloc[train]
        y_train = df_train['target'].iloc[train]
        # print(x_train.shape, y_train.shape)
        x_test = df_train[used_columns].iloc[test]
        y_test = df_train['target'].iloc[test]
        xgmat_test = xgb.DMatrix(x_test,label=y_test)
        classifier.train(x_train, y_train, [(xgmat_test, 'valid')], 10)
        print('train done!\nstart to predict and calculate the trainning loss..')
        y_train_predict,train_loss = classifier.predict_logloss(x_train,y_train)
        print('train shape is %s \ntrain loss is %.9f' % ((str(x_train.shape)),train_loss))

        if test is not None:
            print('start to test...\nstart to calculate the test loss...')
            x_test = df_train[used_columns].iloc[test]
            y_test = df_train['target'].iloc[test]
            y_test_predict, test_loss = classifier.predict_logloss(x_test, y_test)
            print('test shape is %s \ntest loss is %.9f' % ((str(x_test.shape)), test_loss))

    if kfold > 0:
        kf = model_selection.StratifiedKFold(n_splits=kfold)
        for train,test in kf.split(df_train,y=df_train['target']):
            train_predict(train,test)
            break
    else:
        train = list(range(df_train.index.size))
        test = None
        train_predict(train,test)

    if path_save:
        final_predict = classifier.predict(df_test[used_columns])
        df_result = pd.DataFrame(final_predict,columns=["high", "medium", "low"],index=df_test.listing_id)
        df_result.to_csv(path_save)



if __name__ == '__main__':

    path_train = 'data/train.json'
    path_test = 'data/test.json'
    path_save = 'data/submission.csv'
    run_model(path_train,path_test,path_save,kfold=10)

    

    

        
        
        
        
        
        
        
        