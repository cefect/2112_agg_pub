'''
Created on Feb. 21, 2022

@author: cefect
'''
import os, shutil, pickle
import pytest
import numpy as np

from agg.hyd.hscripts import Model as CalcSession

from qgis.core import QgsVectorLayer, QgsRasterLayer
from hp.Q import vlay_get_fdf
from hp.logr import get_new_file_logger, get_new_console_logger, logging

from pandas.testing import assert_frame_equal

from definitions import src_dir

proj_lib =     {
                    #===========================================================
                    # 'point':{
                    #       'EPSG': 2955, 
                    #      'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0218.geojson', 
                    #      'dem': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\dem_obwb_test_0218.tif', 
                    #      'wd_dir': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd',
                    #      #'aoi':r'C:\LS\02_WORK\NRC\2112_Agg\04_CALC\hyd\OBWB\aoi\obwb_aoiT01.gpkg',
                    #         }, 
                    #===========================================================
                    'testSet1':{ #consider making this random?
                          'EPSG': 2955, 
                         'finv_fp': r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\finv_obwb_test_0219_poly.geojson', 
                         #'wd_fp':r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd\wd_rand_test_0304.tif',

                            }, 
                        }

    
#===============================================================================
# fixture-----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write=False
    if write:
        print('WARNING!!! runnig in write mode')
    return write


    
 
    
@pytest.fixture(scope='function')
def session(tmp_path,
            #wrk_base_dir=None, 
            base_dir, write,logger, feedback,#see conftest.py (scope=session)

                    ):
 
    np.random.seed(100)
    
    #get working directory
    wrk_dir = None
    if write:
        wrk_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    
    with CalcSession(out_dir = tmp_path, proj_lib=proj_lib, wrk_dir=wrk_dir, 
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        yield ses



#===============================================================================
# DIRECTOREIES--------
#===============================================================================

@pytest.fixture
def true_dir(write, tmp_path, base_dir):
    true_dir = os.path.join(base_dir, os.path.basename(tmp_path))
    if write:
        if os.path.exists(true_dir): 
            shutil.rmtree(true_dir)
            os.makedirs(true_dir) #add back an empty folder
            
    return true_dir

@pytest.fixture(scope='function')
def out_dir(write, tmp_path, tgen_dir):
    if write:
        return tgen_dir
    else:
        return tmp_path
    
 

def get_abs(relative_fp):
    return os.path.join(src_dir, relative_fp)

#===============================================================================
# MISC-------
#===============================================================================
@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[','_').replace(']', '_')

@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)

@pytest.fixture(scope='session')
def feedback(logger):
    from hp.Q import MyFeedBackQ
    return MyFeedBackQ(logger=logger)

#===============================================================================
# helpers-------
#===============================================================================
def build_compileds(dkey_d, base_dir): 
    d = dict()
    for dkey, folder in dkey_d.items():
        input_fp = search_fp(os.path.join(base_dir, folder), '.pickle', dkey) #find the data file.
        d[dkey] = input_fp
    return d

def retrieve_finv_d(finv_agg_fn, session, base_dir):
    d= dict()
    for dkey in ['finv_agg_d', 'finv_agg_mindex']:
 
        input_fp = search_fp(os.path.join(base_dir, finv_agg_fn), '.pickle', dkey) #find the data file.
        assert os.path.exists(input_fp), 'failed to find match for %s'%finv_agg_fn
        d[dkey] = retrieve_data(dkey, input_fp, session)
        
    return d.values()

def retrieve_data(dkey, fp, ses):
    assert dkey in ses.data_retrieve_hndls
    hndl_d = ses.data_retrieve_hndls[dkey]
    
    return hndl_d['compiled'](fp=fp, dkey=dkey)

def search_fp(dirpath, ext, pattern): #get a matching file with extension and beginning
    assert os.path.exists(dirpath), 'searchpath does not exist: %s'%dirpath
    fns = [e for e in os.listdir(dirpath) if e.endswith(ext)]
    
    result= None
    for fn in fns:
        if fn.startswith(pattern):
            result = os.path.join(dirpath, fn)
            break
        
    if result is None:
        raise IOError('failed to find a match for \'%s\' in %s'%(pattern, dirpath))
    
    assert os.path.exists(result), result
        
        
    return result

#===============================================================================
# VALIDATIOn-------
#===============================================================================
def check_layer_d(d1, d2, #two containers of layers
                   test_data=True, #check vlay attributes
                   ignore_fid=True,  #whether to ignore the native ordering of the vlay
                   msg=''
                   ):
    
    assert d1.keys()==d2.keys()
    
    for k, vtest in d1.items():
        vtrue = d2[k]
        
        dptest, dptrue = vtest.dataProvider(), vtrue.dataProvider()
        
        assert type(vtest)==type(vtrue), msg
        
        #=======================================================================
        # vectorlayer checks
        #=======================================================================
        if isinstance(vtest, QgsVectorLayer):
            assert dptest.featureCount()==dptrue.featureCount(),msg
            assert vtest.wkbType() == dptrue.wkbType(), msg
            
            #data checks
            if test_data:
                true_df = vlay_get_fdf(vtrue).drop('fid', axis=1, errors='ignore')
                test_df = vlay_get_fdf(vtest) #.drop('fid', axis=1, errors='ignore'), 
                
                if ignore_fid:
                    true_df = true_df.sort_values(true_df.columns[0],  ignore_index=True) #sort by first column and reset index
                    test_df = test_df.sort_values(test_df.columns[0],  ignore_index=True)
                
                
                assert_frame_equal(true_df, test_df,check_names=False)
                
