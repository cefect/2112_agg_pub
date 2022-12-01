'''
Created on May 9, 2022

@author: cefect

old tests for rloss
'''

import os  
import pytest
print('pytest.__version__:' + pytest.__version__)
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

from conftest import retrieve_finv_d, retrieve_data, search_fp, build_compileds, proj_lib, check_layer_d





@pytest.mark.dev
@pytest.mark.parametrize('rsamp_fn', #see test_rsamps
             ['test_rsamps_test_finv_agg_grid0', 'test_rsamps_test_finv_agg_grid1', 'test_rsamps_test_finv_agg_grid2']) 
@pytest.mark.parametrize('vid', [49, 798,811, 0])
def test_rloss(session, rsamp_fn, vid, base_dir, true_dir, df_d):
 
    #===========================================================================
    # load inputs
    #===========================================================================
    dkey = 'rsamps'
    input_fp = search_fp(os.path.join(base_dir, rsamp_fn), '.pickle', dkey) #find the data file.
    dxser = retrieve_data(dkey, input_fp, session)
    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='rloss'
    rdxind = session.build_rloss(dkey=dkey, vid=vid, dxser=dxser, df_d=df_d)
    
    #===========================================================================
    # check
    #===========================================================================
    rserx = rdxind['rl']
    assert rserx.notna().all()
    assert rserx.min()>=0
    assert rserx.max()<=100
    
    #===========================================================================
    # retrieve trues
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_frame_equal(rdxind, true)


rloss_fn_l = ['test_rloss_49_test_rsamps_test0','test_rloss_49_test_rsamps_test1','test_rloss_49_test_rsamps_test2',
              'test_rloss_798_test_rsamps_tes0','test_rloss_798_test_rsamps_tes1','test_rloss_798_test_rsamps_tes2',
              'test_rloss_811_test_rsamps_tes0','test_rloss_811_test_rsamps_tes1','test_rloss_811_test_rsamps_tes2']


@pytest.mark.parametrize('rloss_fn', rloss_fn_l) #see test_rloss
def test_tloss(session, base_dir, rloss_fn):
    #scale_cn = session.scale_cn
    #===========================================================================
    # load inputs
    #===========================================================================
    dkey = 'rloss'
    input_fp = search_fp(os.path.join(base_dir, rloss_fn), '.pickle', dkey) #find the data file.
    rl_dxind = retrieve_data(dkey, input_fp, session)
    
    #build total vals
    """easier (and broader) to use random total vals than to select the matching"""
    
    tv_dx1 = pd.Series(np.random.random(len(rl_dxind)), index=rl_dxind.droplevel(1).index, name='0').to_frame()
    tv_dx2 = pd.concat({'tvals':tv_dx1}, axis=1, names=['dkey', 'iter'])
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tloss'
    tl_dxind = session.build_tloss(dkey=dkey, tv_data=tv_dx2, rl_dxind=rl_dxind)
    
    #===========================================================================
    # check
    #===========================================================================
    assert_frame_equal(tl_dxind.loc[:, idx['tvals', :]].droplevel(1, axis=0), tv_dx2, check_index_type=False)
    
    #check relative loss
    rl_dx_chk = tl_dxind.loc[idx[:, 'tloss']].droplevel(1, axis=0).divide(tv_dx2.loc[idx[:, 'tvals']])
    rl_dx_chk.columns = pd.Index(range(len(rl_dx_chk.columns)))
    
    rl_dx_chk2 = tl_dxind.loc[idx[:, 'rloss']].droplevel(1, axis=0)
    rl_dx_chk2.columns = pd.Index(range(len(rl_dx_chk2.columns)))
    assert_frame_equal(rl_dx_chk, rl_dx_chk2)

        
#===============================================================================
# helpers-----------
#===============================================================================



 
            