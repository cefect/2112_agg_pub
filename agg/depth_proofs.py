'''
Created on Apr. 9, 2022

@author: cefect

simple math checks on some rsutls
'''
import pandas as pd
import numpy as np


def depthFunc(x):
    
    df = pd.Series(x, dtype=float, name='raw').to_frame()
    df['zeros'] = 0
    
    result = df.max(axis=1).values
    
    if len(result)==1:
        return result[0]
    else:
        return result
    
 

def compare_means(
        n=50, #iterations
        shape=10, #length of test arrays
        #delta = 1.0, 
        #dry_frac=0.1,
        ):
    """explorin the bias for raster downsampling stage"""
    
    
    res_d = dict()
    for i in range(n):
        
 
        
        ar_bottom = np.random.rand(shape) * 3 #between zero and 3
              
        ar_top = np.random.rand(shape) + 1  #between 1 and 2
        
        
        
        
        res_d[i] = {
            'pre':depthFunc(ar_top.mean() - ar_bottom.mean()), #averaging before subtraction
            'post':depthFunc(ar_top - ar_bottom).mean(), #averaging after subtraction
            'bot_mean':ar_bottom.mean(),
            'top_mean':ar_top.mean(),
 
            'shape':shape,
            }
        
    #===========================================================================
    # wrap
    #===========================================================================
    df = pd.DataFrame.from_dict(res_d).T
    
    df['bias'] = df['post']/df['pre']
    
    print(df)
    
    
        
        
        


if __name__ == "__main__": 
    compare_means()