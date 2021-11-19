import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


##functions


def predict_score(client, model):
    
    with open('test_domain.bin', 'rb') as f_in:
        test_domain = pickle.load(f_in)
        f_in.close()

    print ('client=', client)
    print (type(client))
    item = test_domain[int(client)]
    #print (type(test_domain))
    #print (type(item))
    #print (test_domain.shape)
    item = item.reshape(1,243)
    #print (item.shape)
    #test_sample = test_domain[num] # pour Lime

    xgb_pred = model.predict_proba(item)
    print (xgb_pred)
    return (xgb_pred)
