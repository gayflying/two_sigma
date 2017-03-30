#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

"""
Use 'data/price.txt' to build a KNN regressor based on 'distance', which means the near instance will contribute more on the predict value instead of the mean value.

Parameters
----------
neighbor_num : int
    The KNN regressor will estimate based on the nearst "neighbor_num"

Returns
-------
out : sklearn.neighbors.KNeighborsRegressor
    A Regressor that
"""
def PositionValueProphet(neighbor_num=20):
    priceData = np.array([line.replace("\n","").split(' ') for line in open('data/price.txt').readlines()]).astype(np.float64)
    neigh = KNeighborsRegressor(n_neighbors=neighbor_num, weights='distance')
    neigh.fit(priceData[:,:2], priceData[:,2])
    return neigh
