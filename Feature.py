

import os,sys,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import featureFunction



class FeatureSpace:
    
    def __init__(self, category=None, featureList=None, **kwargs):
        self.featureFunc = []

        if category is not None:
            self.category = category
            self.featureList=[]
            if self.category == 'all':
                for name, obj in inspect.getmembers(featureFunction):
                    if inspect.isclass(obj) and name!='Base':
                        self.featureList.append(name)

            else:
                for name, obj in inspect.getmembers(featureFunction):
                    if inspect.isclass(obj) and name!='Base':
                        if obj().category in self.category:
                            self.featureList.append(name)
                            
        else:
            self.featureList= featureList
            
        self.featureFunc = []
        m = featureFunction
        
        for item in self.featureList:
            if item in kwargs.keys():
                a = getattr(m, item)( kwargs[item])
            else:
                a = getattr(m, item)()
            self.featureFunc.append(a.fit)


            

        
    def calculateFeature(self, data):
        self._X = np.asarray(data)
        self.__result = []
        for f in self.featureFunc:
            self.__result.append(f(self._X))
        return self
    
    def result(self, method = 'array'):
        if method == 'array':
            return np.asarray(self.__result)
        elif method == 'dict':
            return dict(zip(self.featureList,self.__result))
        else:
            return self.__result

