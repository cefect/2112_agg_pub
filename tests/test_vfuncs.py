'''
Created on Feb. 21, 2022

@author: cefect

pytests for vcunfs
'''
#===============================================================================
# imports-------
#===============================================================================
import os, pickle
import pytest
from numpy.testing import assert_equal

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import numpy as np

import warnings

from agg.coms.scripts import QSession as CalcSession

#===============================================================================
# prep---------
#===============================================================================
def load_pick(fp):
    assert os.path.exists(fp), fp
        
    with open(fp, 'rb') as f:
        data = pickle.load(f)
        
    return data
    
 
    
"""
beacuse we use these to specify the tests, loading them at the head
   we call a new session for the actual tests... but keep this data alive to speed things up
"""
vid_df = load_pick(r'C:\LS\10_OUT\2112_Agg\ins\vfunc\vid_df_hyd5_dev_0414.pickle')
df_d = load_pick(r'C:\LS\10_OUT\2112_Agg\ins\vfunc\df_d_hyd5_dev_0414.pickle')
    
 
vid_l = vid_df.index.tolist() #list of vfuncs to test

#add some others
vid_l =  [0, 1001, 1002] + vid_l

#vid_l = [1002]
#===============================================================================
# fixture-----
#===============================================================================



@pytest.fixture(scope='module') 
def session(
            #tmpdir_factory,
            #wrk_base_dir=None,
            #base_dir, 
            write, #see conftest.py
            logger, feedback,#see conftest.py (scope=session)
                    ):
    """
    using module scope so each vfunc doesnt have to re-spawn the session
        this reduces independence.. but vfuncs are quite independent from the session anyway
    speeds things up alot
    
    """
    
    #get working directory
    wrk_dir = None
    #===========================================================================
    # if write:
    #     wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    #===========================================================================
    
    with CalcSession(out_dir = None, wrk_dir=wrk_dir, overwrite=write,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     logger=logger, feedback=feedback,
                     ) as ses:
        yield ses
        
@pytest.fixture      
def vfunc(session, request, ):
    
    vid = request.param
    
    #add the preloads
    session.data_d.update({'vid_df':vid_df.copy(), 'df_d':df_d}) 
    
    return session.build_vfunc(vid=vid)

        
@pytest.mark.dev
@pytest.mark.parametrize('vfunc',vid_l, indirect=True)  
def test_vfunc(vfunc):
    
    #===========================================================================
    # test against raw water depths
    #===========================================================================
    #feeding the original water depths back in
    #vfunc.ddf[vfunc.xcn]
    wd_ar = vfunc.dd_ar[0]
    #assert wd_ar.min()>=0 #we allow this
    rl_ar = vfunc.get_rloss(wd_ar)
    
    assert_equal(rl_ar, vfunc.dd_ar[1]), 'training yvals dont match output yvals'

    #===========================================================================
    # test extremes
    #===========================================================================
    #left (negatives)
    rl_ar = vfunc.get_rloss(np.linspace(-100,wd_ar.min()-.001,num=5))
    assert_equal(np.full(len(rl_ar), 0), rl_ar)
    
    #right (maximums)
    rl_ar = vfunc.get_rloss(np.linspace(wd_ar.max(),100,num=5))
    assert_equal(np.full(len(rl_ar), vfunc.dd_ar[1].max()), rl_ar)
    
    #===========================================================================
    # test realism
    #===========================================================================
    if vfunc.relative:
        rl_ar = vfunc.get_rloss(np.linspace(-5,10,num=20))
        
        assert pd.Series(rl_ar).notna().all()
        assert rl_ar.min()==0
        
        assert rl_ar.max()>=5.0
        
        #check max (some functions claim this is reasonable)
        if not rl_ar.max()<=100.0:
            warnings.warn(UserWarning("rl_ar.max()>100 (%.2f)"%rl_ar.max()))
 
        
        assert np.all(np.diff(rl_ar)>=0), 'damage values non-monotonic'
        
 
        
    
 
    
    