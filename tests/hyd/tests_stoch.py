'''
Created on Feb. 22, 2022

@author: cefect

test for stochastic model
'''
import os  
import pytest

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

import numpy as np

from numpy.testing import assert_equal

from agg.hyd.hscripts import ModelStoch
from tests.conftest import retrieve_finv_d, retrieve_data, search_fp, build_compileds

@pytest.fixture
def modelstoch(tmp_path,
            #wrk_base_dir=None, 
            base_dir, write,logger, feedback,#see conftest.py (scope=session)
            iters=3,
            proj_lib =     {
 
                    'testSet1':{
                          'EPSG': 2955, 
                         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0219_poly.geojson', 
                            }, 
                        },
                    ):
    """
    TODO: get module scoped logger
    """
    #set the seed
    """need to do this for each stochastic runs to get consistent results when combining with other modules"""
    np.random.seed(100)
    
    #get working directory
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    
    with ModelStoch(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     iters=iters,
                     ) as ses:
        yield ses
        
#===============================================================================
# tests-------
#===============================================================================

@pytest.mark.parametrize('tval_type',['rand'], indirect=False) #see tests_model for other types
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
def testS_04tvals_raw(modelstoch,true_dir, base_dir, write, 
               finv_agg_fn, tval_type, normed):
    """very similar to tests_model"""
    norm_scale=1.0
    dkey='tvals_raw'
 
    #===========================================================================
    # load inputs   
    #===========================================================================
    #set the compiled references    
    modelstoch.compiled_fp_d = build_compileds({'finv_agg_mindex':finv_agg_fn},
                                            base_dir)
    
    finv_agg_mindex = modelstoch.retrieve('finv_agg_mindex')

    #===========================================================================
    # execute
    #===========================================================================
    
    tvals_raw_dx = modelstoch.build_tvals_raw(dkey=dkey, 
                                            norm_scale=norm_scale,
                                            tval_type=tval_type, 
                                            normed=normed,
                                            mindex =finv_agg_mindex, write=write)
    
 
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, modelstoch)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_frame_equal(tvals_raw_dx, true)
        
 
@pytest.mark.parametrize('finv_agg_fn, dscale_meth, tvals_raw',[ #have to combine finv_agg with correct tvals_raw output
        ['test_finv_agg_gridded_50_0', 'centroid',              'testS_04tvals_raw_test_finv_ag0'], 
        ['test_finv_agg_none_None_0', 'none',                   'testS_04tvals_raw_test_finv_ag2'],
        ['test_finv_agg_gridded_50_0', 'area_split',            'testS_04tvals_raw_test_finv_ag0'], 
                                        ], indirect=False)  #see test_finv_agg
def testS_05tvals(modelstoch,finv_agg_fn, true_dir, base_dir, write, 
               tvals_raw, dscale_meth):
 
    session = modelstoch
    
    #===========================================================================
    # load inputs   
    #===========================================================================    
    #set the compiled references    
    session.compiled_fp_d = build_compileds({'finv_agg_d':finv_agg_fn,   'tvals_raw':tvals_raw},
                                            base_dir)
    
    #retrieve uncompiled
    tvals_raw = session.retrieve('tvals_raw')
    
    if dscale_meth =='area_split':
        finv_agg_d = session.retrieve('finv_agg_d') #only needed by dscale_
    else:
        finv_agg_d=None
    #finv_agg_mindex = session.retrieve('finv_agg_mindex')
    

    
    #===========================================================================
    # execute
    #===========================================================================
    dkey='tvals'
    tv_dx = session.build_tvals(dkey=dkey, write=write,
                                    tvals_raw_serx=tvals_raw,
                                    finv_agg_d=finv_agg_d,                                     
                                    dscale_meth=dscale_meth)
    
 
    #norm checks
    norm_scale=1.0 #for test proofing
    normed=True #should only pass tvals_raw that are normed
    if normed:
        """I'm not sure this needs to hold for all gridded inventories"""
        assert (tv_dx.groupby(level='studyArea').sum().round(3)==norm_scale).all().all()
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_frame_equal(tv_dx, true)
    
    
    
    
    
    
    