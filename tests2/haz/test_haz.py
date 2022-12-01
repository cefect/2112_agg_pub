'''
unit tests for downsample v2
'''
 
from hp.np import dropna
from hp.rio import RioWrkr, write_array, load_array, get_stats
from numpy import array, dtype
from tests2.conftest import compare_dicts, src_dir, get_abs, crs, proj_d, get_ar_d, \
    get_rlay_fp_d, get_ar_source, shape_base, get_ar
    
import numpy as np
import pandas as pd
import pytest, copy, os, random
import rasterio as rio
from agg2.haz.coms import get_rand_ar, get_wse_filtered, assert_dx_names, coldx_d
from agg2.haz.scripts import UpsampleSession as Session
from agg2.haz.run import run_haz_agg2
 
xfail = pytest.mark.xfail
 

#===============================================================================
# helpers and globals------
#===============================================================================
dsc_l_global = [1,2*1,2*2]
prec=5
 
#for test data
"""better to use a bounding box for integration w/ vectors"""
output_kwargs = dict(crs=crs,transform=rio.transform.from_origin(1,100,1,1)) 

test_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data'

def assert_stat_check(fp, msg=''):
    dx = pd.read_pickle(fp)
    assert_dx_names(dx, msg=msg+' %s'%os.path.basename(fp))
    
#===============================================================================
# FIXTURES-----
#===============================================================================
@pytest.fixture(scope='function')
def dem_ar(shape):
    return get_ar('dem', shape)
    
@pytest.fixture(scope='function')
def dem_fp(dem_ar, tmp_path):
    return write_array(dem_ar, os.path.join(tmp_path, f'demR_{dem_ar.shape[0]}.tif'), **output_kwargs)
 

@pytest.fixture(scope='function')
def wse_ar(shape, dem_ar):
    wd_ar = get_ar('wse', shape)    
    return get_wse_filtered(dem_ar+wd_ar, dem_ar, nodata=-9999)

@pytest.fixture(scope='function')
def wse_fp(wse_ar, tmp_path): 
    return write_array(wse_ar, os.path.join(tmp_path, f'wseR_{wse_ar.shape[0]}.tif'), **output_kwargs)

@pytest.fixture(scope='function')
def wrkr(tmp_path,write,logger, test_name, 
                    ):
    
    """Mock session for tests"""
 
    np.random.seed(100)
    random.seed(100)
 
    
    with Session(
                 crs=crs, nodata=-9999,
                 #oop.Basic
                 out_dir=tmp_path, 
                 tmp_dir=os.path.join(tmp_path, 'tmp_dir'),
                 prec=prec,
 
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
        
@pytest.fixture(scope='function')
def agg_pick_df(dsc_l, tmp_path):
    """simulate run_agg"""
    fp_lib, ar_lib = dict(), dict()
    
    #===========================================================================
    # build arrays
    #===========================================================================
    
    #build terrain and WD
    for layName in ['wd', 'dem']:
        ar_lib[layName] = get_ar_d(dsc_l, layName)
 
        
    #build wse
    d = dict()
 
    for i, wd_ar in ar_lib['wd'].items():
        dem_ar = ar_lib['dem'][i]        
        d[i] = get_wse_filtered(dem_ar+wd_ar, dem_ar, nodata=-9999)
    
    ar_lib['wse'] = d
    #===========================================================================
    # #build pickels
    #===========================================================================
    for layName, ar_d in ar_lib.items():
        fp_lib[layName] = get_rlay_fp_d(ar_d, layName, tmp_path)
    #===========================================================================
    # wrap
    #===========================================================================
    return pd.DataFrame.from_dict(fp_lib).rename_axis('scale')
    
@pytest.fixture(scope='function')
def agg_pick_fp(agg_pick_df, tmp_path):
 
    
    ofp = os.path.join(tmp_path, 'test_agg_%i.pkl' % len(agg_pick_df))
    agg_pick_df.to_pickle(ofp)
    
    return ofp
    

#===============================================================================
# UNIT TESTS--------
#===============================================================================

@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((16,16))
    ]) 
@pytest.mark.parametrize('dsc_l', [([1,2,4])])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter'])
def test_00_runDsmp(wrkr, dsc_l,method,
                    dem_fp,wse_fp):
    
    wrkr.run_agg(dem_fp, wse_fp,  dsc_l=dsc_l,
                  #write=True,
                  #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data', method).
                  )

#===============================================================================
# 
# @pytest.mark.parametrize('reso_iters', [3, 10])
# def test_00_dscList(wrkr, reso_iters):
#     
#     #build with function
#     res_l = wrkr.get_dscList(reso_iters=reso_iters)
#     
#     #===========================================================================
#     # #validate
#     #===========================================================================
#     assert isinstance(res_l, list)
#     assert len(res_l)==reso_iters
#     assert res_l[0]==1
#===============================================================================

 

@pytest.mark.parametrize('dem_ar, wse_ar', [
    get_rand_ar((8,8))
    ])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter',
    ])
@pytest.mark.parametrize('dsc_l', [([1,2])])
def test_01_dset(dem_fp,dem_ar,wse_fp, wse_ar,   wrkr, dsc_l, method):
    wrkr.build_dset(dem_fp, wse_fp, dsc_l=dsc_l, method=method)


@pytest.mark.parametrize('shape', [shape_base])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter',
    ])
@pytest.mark.parametrize('dsc_l', [([1,2])])
def test_01_runAgg(dem_fp,wse_fp, wrkr, dsc_l, method, agg_pick_fp):
    """wrapper for build_dset"""
    pick_fp = wrkr.run_agg(dem_fp, wse_fp, method=method, dsc_l=dsc_l, write=True,
                 #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                 )
    
    #validate
    df = pd.read_pickle(pick_fp)
    
    #loop and make sure each layer has matching properties
    first=True
    
    for layName, col in df.items():
        assert layName in coldx_d['layer']
        
        for scale, fp in col.items():
            with rio.open(fp, mode='r') as ds:
                stats_d  = get_stats(ds, att_l=['crs', 'nodata', 'bounds'])
                if first:
                    stats_d1 = stats_d
                    first = False
                    continue
                
                compare_dicts(stats_d, stats_d1, msg='%s.%s'%(layName, scale))
                
                if layName=='wse':
                    ds.read(1, masked=True)
                    
    #check against test fixture
    df_fix = pd.read_pickle(agg_pick_fp)
    
    assert set(df_fix.columns).symmetric_difference(df.columns)==set()
    assert set(df_fix.index).symmetric_difference(df.index)==set()
                
 
@pytest.mark.parametrize('dsc_l', [(dsc_l_global)]) 
def test_01b_dsc_agg_x(wrkr, agg_pick_df):
    """this is redundant w/ test_07_pTP"""
    wrkr.build_downscaled_aggXR(agg_pick_df)
          
          
          
@pytest.mark.dev 
@pytest.mark.parametrize('shape', [shape_base])
@pytest.mark.parametrize('dsc_l', [(dsc_l_global)]) 
@pytest.mark.parametrize('write', [True, False]) 
def test_02_catMasks(wrkr,dem_fp,wse_fp, dsc_l, cm_pick_fp, write):
    cf_fp, meta_fp = wrkr.run_catMasks(dem_fp, wse_fp, dsc_l=dsc_l, write=write,
                               #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                               )
    
    #===========================================================================
    # validate
    #===========================================================================
    df = pd.read_pickle(meta_fp)
    """
    df.columns
    """
    #check against test fixture
    df_fix = pd.read_pickle(cm_pick_fp)
    
    assert set(df_fix.columns).symmetric_difference(df.columns)==set()
    assert set(df_fix.index).symmetric_difference(df.index)==set()

@pytest.mark.parametrize('dsc_l', [(dsc_l_global)]) 
def test_03_stats(wrkr, agg_pick_fp, cm_pick_fp):
    res_fp = wrkr.run_stats(agg_pick_fp, cm_pick_fp, write=True)
    assert_stat_check(res_fp)



@pytest.mark.parametrize('dsc_l', [(dsc_l_global)]) 
def test_04_statsFine(wrkr,agg_pick_fp, cm_pick_fp):
    res_fp = wrkr.run_stats_fine(agg_pick_fp, cm_pick_fp, write=True)
    assert_stat_check(res_fp)

 

@pytest.mark.parametrize('dsc_l', [(dsc_l_global)]) 
def test_05_diffs(wrkr, agg_pick_fp):
    res_fp = wrkr.run_diffs(agg_pick_fp,write=True,
                           #out_dir=os.path.join(r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests2\haz\data')
                           )


diffs_pick_fp = os.path.join(src_dir, r'tests2\haz\data\diffs\SJ_test05_direct_0922_diffs.pkl') 
    


@pytest.mark.parametrize('dsc_l', [(dsc_l_global)], indirect=False) 
@pytest.mark.parametrize('pick_fp', [diffs_pick_fp]) 
def test_06_diff_stats(wrkr, pick_fp, cm_pick_fp):
    res_fp = wrkr.run_diff_stats(pick_fp, cm_pick_fp, write=True,
 
                           )
    assert_stat_check(res_fp)
    



@pytest.mark.parametrize('dsc_l', [(dsc_l_global)])
def test_07_pTP(wrkr, agg_pick_df, cm_pick_fp):
    """too complicated to build the netCDF from scratch
    and we need to be careful with memory handling"""
    nc_fp = wrkr.build_downscaled_agg_xarray(agg_pick_df)
    
    wrkr.run_pTP(nc_fp, cm_pick_fp)
    
#===============================================================================
# INTEGRATIOn tests ------------
#===============================================================================


@pytest.mark.parametrize('dsc_l', [(dsc_l_global)])
@pytest.mark.parametrize('method', [
    'direct', 
    'filter',
    ])
@pytest.mark.parametrize('proj_d', [proj_d])
def test_runHaz(method, proj_d, dsc_l, tmp_path):
    """use the function runner"""
    fp_d, stat_d = run_haz_agg2(proj_d=proj_d, method=method, dsc_l=dsc_l, case_name='tCn', run_name='tRn',
                 wrk_dir=tmp_path)
    
    for k, fp in stat_d.items():
        assert_stat_check(fp)
    
    
    
    
 
 
