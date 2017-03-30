#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

"""
Use 'data/locationInterestLevel.txt' to build a KNN regressor based on 'distance', which means the near instance will contribute more on the predict interestlevel instead of the mean level.

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
    interestlevelData = np.array([line.replace("\n","").split(' ') for line in open('data/locationInterestLevel.txt').readlines()]).astype(np.float64)
    neigh = KNeighborsRegressor(n_neighbors=neighbor_num, weights='uniform')
    neigh.fit(interestlevelData[:,:2], interestlevelData[:,2:])
    return neigh
