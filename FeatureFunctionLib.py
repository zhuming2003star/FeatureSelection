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
    '''
    This is just a prototype, not a real feature
    '''
    def __init__(self, length): 
        self.category='basic'
        if len(length)!=2:
            print "need 2 parameters for feature automean"
            sys.exit(1)
        self.length = length[0]
        self.length2 = length[1]
    def fit(self, data):
        return np.mean(data)+self.length+self.length2

class meanvariance(Base):
    def __init__(self): 
        self.category='basic'
      
    def fit(self, data):
        return np.std(data)/np.mean(data)



    
class autocor(Base):
    def __init__(self, lag=1):
        self.category='timeSeries'
    def fit(self, data):
        return np.correlate(data, data)[0]


class StetsonL(Base):
    def __init__(self, second_data):
        self.category='timeSeries'
        if second_data is None:
            print "please provide another data series to compute StetsonL"
            sys.exit(1)
        self.data2 = second_data

    def fit(self, data):
        if len(data) != len(self.data2) :
            print " the lengh of 2 data series are not the same"
            sys.exit(1)

        N = len(data)
        sigmap = np.sqrt(N*1.0/(N-1)) * (data-np.mean(data))/np.std(data)
        sigmaq = np.sqrt(N*1.0/(N-1)) * (self.data2-np.mean(self.data2))/np.std(self.data2)
        sigma_i = sigmap * sigmaq
        J= 1.0/len(sigma_i) * np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i)))
        K = 1/np.sqrt(N*1.0) * np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i**2))
        return J*K/0.798        

class Con(Base):
    '''
    Index introduced for selection of variable starts from OGLE database. 
    To calculate Con, we counted the number of three consecutive starts that are out of 2sigma range, and normalized by N-2
    '''
    def __init__(self, consecutiveStar=3):
        self.category='timeSeries'
        self.consecutiveStar = consecutiveStar

    def fit(self, data):

        N = len(data)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(data)
        m = np.mean(data)
        count=0
        
        for i in xrange(N-self.consecutiveStar+1):
            flag = 0
            for j in xrange(self.consecutiveStar):
                if (data[i+j] > m+2*sigma or data[i+j] < m-2*sigma) :
                    flag = 1
                else:
                    flag=0
                    break
            if flag:
                count = count+1
        return count*1.0/(N-self.consecutiveStar+1)


class VariabilityIndex(Base):
    '''
    The index is the ratio of mean of the square of successive difference to the variance of data points
    '''
    def __init__(self):
        self.category='timeSeries'
        

    def fit(self, data):

        N = len(data)
        sigma2 = np.var(data)
        
        return 1.0/((N-1)*sigma2) * np.sum(np.power(data[1:] - data[:-1] , 2))


class B_R(Base):
    '''
    average color for each MACHO lightcurve 
    mean(B1) - mean(B2)
    '''
    def __init__(self, second_data):
        self.category='basic'
        if second_data is None:
            print "please provide another data series to compute B_R"
            sys.exit(1)
        self.data2 = second_data
      
    def fit(self, data):
        return np.mean(data) - np.mean(self.data2)