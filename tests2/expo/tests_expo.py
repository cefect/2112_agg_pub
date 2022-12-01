'''
Created on Sep. 6, 2022

@author: cefect
'''

from tests2.conftest import proj_d
import numpy as np
import pandas as pd
import pytest, copy, os, random
from agg2.expo.scripts import ExpoSession as Session
from agg2.expo.run import run_expo
from agg2.haz.coms import cm_int_d
from hp.rio import write_array
import shapely.geometry as sgeo
import rasterio as rio

from tests2.conftest import bbox_base, crs

#===============================================================================
# vars
#===============================================================================

#bbox1 = sgeo.box(25, 25, 70, 70)

#===============================================================================
# HELPERS--------
#===============================================================================
 
#===============================================================================
# FIXTURES-----
#===============================================================================



@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    #np.random.seed(100)
    #random.seed(100)
 
    
    with Session(  
                 #GeoPandas
                 crs=crs,
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 #prec=prec,
                  proj_name='expoTest', #probably a better way to propagate through this key 
                 run_name=test_name[:8].replace('_',''),
                  
                 relative=True, write=write, #avoid writing prep layers
                 
                 logger=logger, overwrite=True,
                   
                   #oop.Session
                   exit_summary=False,logfile_duplicate=False,
                   compiled_fp_d=dict(), #I guess my tests are writing to the class... no tthe instance
 
                   ) as ses:
        
 
        assert len(ses.data_d)==0
        assert len(ses.compiled_fp_d)==0
        assert len(ses.ofp_d)==0
        yield ses

#===============================================================================
# ressampl class mask
#===============================================================================

 


#===============================================================================
# @pytest.fixture(scope='function')
# def cMask_pick_fp(cMask_rlay_fp, tmp_path):
#     """mimic output of run_catMasks"""
#     df = pd.DataFrame.from_dict(
#         {'downscale':[1,2],'catMosaic':[np.nan, cMask_rlay_fp],}
#         )
#     ofp = os.path.join(tmp_path, 'test_cMasks_%i.pkl'%len(df))
#     df.to_pickle(ofp)
#     
#     return ofp
#  
# 
# @pytest.fixture(scope='function')
# def cMask_rlay_fp(cMask_ar, tmp_path):
#     ofp = os.path.join(tmp_path, 'cMask_%i.tif'%cMask_ar.size)
#     
#     width, height = cMask_ar.shape
#     
#     write_array(cMask_ar, ofp, crs=crs,
#                  transform=rio.transform.from_bounds(*bbox_base.bounds,width, height),  
#                  masked=False)
#     
#     return ofp
#  
#     
#  
# @pytest.fixture(scope='function')    
# def cMask_ar(shape):
#     return np.random.choice(np.array(list(cm_int_d.values())), size=shape)
#===============================================================================



    

#===============================================================================
# TESTS-------------
#===============================================================================

@pytest.mark.parametrize('finv_fp', [proj_d['finv_fp']])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
@pytest.mark.parametrize('bbox', [
                                bbox_base, 
                                None
                                  ])
def test_01_assetRsc(wrkr, cMask_pick_fp, finv_fp, bbox): 
    ofp = wrkr.build_assetRsc(cMask_pick_fp, finv_fp, bbox=bbox)
    



@pytest.mark.parametrize('dsc_l', [([1,2,5])])
@pytest.mark.parametrize('layName', [
    #'wd',
    'wse'])
@pytest.mark.parametrize('finv_fp', [proj_d['finv_fp']])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
@pytest.mark.parametrize('bbox', [
                                bbox_base, 
                                None
                                  ])
def test_02_laySamp(wrkr, lay_pick_fp, finv_fp, bbox, layName): 
    ofp = wrkr.build_layerSamps(lay_pick_fp, finv_fp, bbox=bbox, layName=layName,
                                write=True,
                                )
    
@pytest.mark.dev
@pytest.mark.parametrize('proj_d', [proj_d])
@pytest.mark.parametrize('dsc_l', [([1,2,5])])
@pytest.mark.parametrize('shape', [(10,10)], indirect=False)
def test_runExpo(proj_d, tmp_path, complete_pick_fp):
    run_expo( wrk_dir=tmp_path, case_name='tCn', run_name='tRn', proj_d=proj_d,
              fp_d={'catMasks':complete_pick_fp},
              )
    
 
    
    
    
    
    
    
    
    