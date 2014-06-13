import os,sys,time
import numpy as np
import pandas as pd

from Base import Base

class Rcs(Base):
    def __init__(self):
        self.category='timeSeries'
    def fit(self, data):
	sigma = np.std(data)
	N = len(data)
	m = np.mean(data)
	s = (np.cumsum(data)-m)*1.0/(N*sigma)
	R = np.max(s) - np.min(s)
        return R
   
class StestonK(Base):
    def __init__(self):
        self.category='timeSeries'
    def fit(self, data):
        N = len(data)
	sigmap = np.sqrt(N*1.0/(N-1)) * (data-np.mean(data))/np.std(data)
	
	K = 1/np.sqrt(N*1.0) * np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap**2))

        
        return K


class automean(Base):
    def __init__(self, length=10):	
        self.category='basic'
        self.length = length[0]
        self.length2 = length[1]
    def fit(self, data):
        return np.mean(data)+self.length+self.length2

    
class autocor(Base):
    def __init__(self):
        self.category='timeSeries'
    def fit(self, data):
        return np.correlate(data, data)[0]
