'''
Created on Aug. 27, 2022

@author: cefect
'''
import os, shutil, pickle
import pytest
import numpy as np
import pandas as pd
from definitions import src_dir
import shapely.geometry as sgeo
from pyproj.crs import CRS
from hp.logr import get_new_file_logger, get_new_console_logger, logging
from hp.rio import write_array
import rasterio as rio
from agg2.haz.coms import cm_int_d

#===============================================================================
# VARS--------
#===============================================================================
"""need something that is cleanly divisible, 
but still manageable at resolution=1
"""
bbox_base = sgeo.box(0, 0, 2**3, 2**4)
shape_base = tuple(map(int, (bbox_base.bounds[2], bbox_base.bounds[3])))
#bbox_base = sgeo.box(0, 0, 100, 100)
crs=CRS.from_user_input(2953)

#saint Jon sub-set test data
SJ_test_dir= r'C:\LS\10_OUT\2112_Agg\ins\hyd\SaintJohn\test'

proj_d = {
    'EPSG':2953,
    'finv_fp':r'C:\LS\09_REPOS\02_JOBS\2112_agg\cef\tests2\expo\data\finv_SJ_test_0906.geojson',
    'wse_fp_d':{'hi':os.path.join(SJ_test_dir,  'GeoNB_LSJ_aoiT01_0829.tif')},
    'dem_fp_d':{1:os.path.join(SJ_test_dir,'NBDNR2015_r01_aoiT01_0829.tif')},
    }
#===============================================================================
# MISC----
#===============================================================================
@pytest.fixture(scope='session')
def write():
    write=False
    if write:
        print('WARNING!!! runnig in write mode')
    return write

@pytest.fixture(scope='function')
def test_name(request):
    return request.node.name.replace('[','_').replace(']', '_')

@pytest.fixture(scope='session')
def logger():
    return get_new_console_logger(level=logging.DEBUG)


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
# data grids--------
#===============================================================================
@pytest.fixture(scope='function')  
def complete_pick_fp(tmp_path, dsc_l):
    """construct teh complete pickle of each layers stack
    equivalent to the expectations for 'catMasks'"""
    
    #build the resolution-filepath stack for each layer
    d = dict()
    for layName in ['wse', 'wd', 'catMosaic']:
        ar_d = get_ar_d(dsc_l, layName)
        d[layName] = get_rlay_fp_d(ar_d, layName, tmp_path)
    
    df = pd.DataFrame.from_dict(d).rename_axis('downscale')
    
    # move downscale to a column
    df = df.reset_index()
 
    ofp = os.path.join(tmp_path, 'test_complete_%i.pkl' % len(df))
    df.to_pickle(ofp)
    
    return ofp

    
@pytest.fixture(scope='function')
@pytest.mark.parametrize('layName', ['wd'])
def lay_pick_fp_wd(lay_pick_fp):
    return lay_pick_fp

@pytest.fixture(scope='function')
def cm_pick_fp(dsc_l, tmp_path):
    """couldnt get this to work nicely with parameterized fixtures"""
    layName = 'catMosaic'
    
    ar_d = get_ar_d(dsc_l, layName)
    rlay_fp_d = get_rlay_fp_d(ar_d, layName, tmp_path)
    
    df = pd.Series(rlay_fp_d).rename('fp').to_frame().rename_axis('scale')
    
    #add thes emetric columns
    coln_l = ['DD', 'WW', 'WP', 'DP']
    confusion_df = pd.DataFrame((np.random.random((len(df),len(coln_l) ))*ar_d[dsc_l[1]].size).astype(int),
        index = df.index, columns = coln_l,dtype=int)
    
    df = df.join(confusion_df)
 
 
    ofp = os.path.join(tmp_path, 't%s_%i.pkl'%(layName, len(df)))
    df.to_pickle(ofp)
    
    return ofp
    

@pytest.fixture(scope='function')
def lay_pick_fp(rlay_fp_d, layName, tmp_path):
    return get_lay_pick_fp(rlay_fp_d, layName, tmp_path)
    
def get_lay_pick_fp(rlay_fp_d, layName, tmp_path):
    
    df = pd.Series(rlay_fp_d).rename(layName).to_frame().rename_axis('scale')
 
    ofp = os.path.join(tmp_path, 't%s_%i.pkl'%(layName, len(df)))
    df.to_pickle(ofp)
    
    return ofp

def get_lay_pick_fp_full(layName,dsc_l, tmp_path):
    ar_d = get_ar_d(dsc_l, layName)
    rlay_fp_d = get_rlay_fp_d(ar_d, layName, tmp_path)
    
    return get_lay_pick_fp(rlay_fp_d, layName, tmp_path)


@pytest.fixture(scope='function')
def rlay_fp_d(ar_d, layName, tmp_path):
    return get_rlay_fp_d(ar_d, layName, tmp_path)
    
def get_rlay_fp_d(ar_d, layName, tmp_path,
                  bbox = bbox_base,
                  ):
    ofp_d = dict()
    for scale, ar_raw in ar_d.items():
        if not isinstance(ar_raw, np.ndarray):
            continue
        ofp = os.path.join(tmp_path, '%s_%03i.tif'%(layName, scale))
        
        width, height = ar_raw.shape
        
        #write
        with rio.open(ofp,'w',driver='GTiff',nodata=-9999,compress=None,
                  height=height,width=width,count=1,dtype=ar_raw.dtype,
                crs=crs,transform=rio.transform.from_bounds(*bbox.bounds,width, height),                
                ) as ds:
            assert ds.res[0]==scale          
            ds.write(ar_raw, indexes=1,masked=False)
            
 
        
        ofp_d[scale]=ofp
    
    return ofp_d
 
    
 
@pytest.fixture(scope='function')    
def ar_d(dsc_l, layName):
    """building some dummy grids for a set of scales"""
    return get_ar_d(dsc_l, layName)
    
def get_ar_d(dsc_l, layName, shape=None):
    """build a dictionary of test arrays
    
    Parameters
    -----------
    shape: tuple
        base fine resolution shape from which to aggregate
    """
    #===========================================================================
    # defaults
    #===========================================================================
    if shape is None:
        #retrieve shape from bounding box
        """gives us a nice base resolution of 1"""
        shape = shape_base
    
    #check division
    assert dsc_l[0]==1    
    
    d1 = shape[0]
    assert d1%dsc_l[-1]==0, 'bad divisor'
    #===========================================================================
    # build base
    #===========================================================================
    samp_ar =get_ar_source(layName)
    
 
    #===========================================================================
    # random sample from these
    #===========================================================================
    res_d = dict()
    for scale in dsc_l:
        
        #skip first catMosaic
        if scale == dsc_l[0] and layName=='catMosaic':
            res_d[scale] = np.nan
            continue 
        
        #get the shape
        assert d1%scale==0, 'bad divisor: %i'%scale
        si = tuple((np.array(shape)//scale).astype(int))        
        
                   
        res_d[scale] = np.random.choice(samp_ar, size=si)
        
    """ for CatMosaic first is null
    for k,v in res_d.items():
        assert isinstance(v, np.ndarray), k"""
    return res_d

def get_ar(layName, shape):
    return np.random.choice(get_ar_source(layName), size=shape)   

def get_ar_source(layName, 
           shape=(10,10),
           ):
    """get a 1D source array to sample from
    
    Note
    ---------
    this needs to be sampled and reshaped
    
    """
    
    if shape is None:
        """generally, this doesnt matter as we set the shape during random sampling"""
        shape = shape_base     
        
    
 
    if layName=='wd':        
        samp_ar = np.concatenate( #50% real 50% ry
            (np.round(np.random.random(shape)*10, 2).ravel(),
            np.full(shape, 0).ravel())
            ).ravel()
            
    elif layName=='wse':
        samp_ar = np.concatenate( #50% real 50% null
            (np.round(np.random.random(shape)*20, 2).ravel(),
            np.full(shape, -9999).ravel())
            ).ravel()
            
    elif layName=='catMosaic': 
        samp_ar = np.array(list(cm_int_d.values()))
        
    elif layName=='dem':
        samp_ar = np.round(np.random.random(shape)*5, 2).ravel()
    
    else:
        raise IOError('not implemented')
    
    return samp_ar
#===============================================================================
# VALIDATIOn-------
#===============================================================================

def compare_dicts(dtest, dtrue, index_l = None, msg=''
                 ):
        df1 = pd.DataFrame.from_dict({'true':dtrue, 'test':dtest})
        
        if index_l is None:
            index_l = df1.index.tolist()
        
        df2 = df1.loc[index_l,: ].round(3)
        
        bx = ~df2['test'].eq(other=df2['true'], axis=0)
        if bx.any():
            raise AssertionError('%i/%i raster stats failed to match\n%s\n'%(bx.sum(), len(bx), df2.loc[bx,:])+msg)
                
def validate_dict(session, valid_dir, test_stats_d, baseName='base'):
    
    true_fp = os.path.join(valid_dir, '%s_true.pkl' % (baseName))
    
    #===========================================================================
    # write trues
    #===========================================================================
    
    if session.write:
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)
        with open(true_fp, 'wb') as f:
            pickle.dump(test_stats_d, f, pickle.HIGHEST_PROTOCOL)
    else:
        assert os.path.exists(true_fp)
        
    #===========================================================================
    # retrieve trues
    #===========================================================================
    with open(true_fp, 'rb') as f:
        true_stats_d = pickle.load(f)
        
    #===========================================================================
    # compare
    #===========================================================================
    compare_dicts(test_stats_d, true_stats_d)

 
