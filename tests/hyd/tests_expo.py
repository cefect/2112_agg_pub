'''
Created on Feb. 20, 2022

@author: cefect

tests for hyd model exposure calcs

TODO: clean out old pickles
'''




import os  
import pytest
print('pytest.__version__:' + pytest.__version__)
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

import numpy as np
np.random.seed(100)
from numpy.testing import assert_equal

from qgis.core import QgsVectorLayer, QgsWkbTypes
import hp.gdal


from agg.hyd.hscripts import StudyArea as CalcStudyArea
from agg.hyd.hscripts import vlay_get_fdf, RasterCalc
from agg.hyd.hscripts import Model as CalcSession

from conftest import retrieve_finv_d, retrieve_data, search_fp, build_compileds, proj_lib, check_layer_d

from definitions import base_resolution


#===============================================================================
# fixtures-----
#===============================================================================
@pytest.fixture(scope='session')
def base_dir():
    base_dir = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\compiled'
    assert os.path.exists(base_dir)
    return base_dir

@pytest.fixture(scope='module')
def df_d():
    """this is an expensive collecting of csvs (database dump) used to build vfuncs
    keeping this alive for the whole module"""
    with CalcSession() as ses:
        df_d = ses.build_df_d(dkey='df_d')
    return df_d





 
@pytest.fixture
def studyAreaWrkr(session, request):
    
    name = request.param
        
    kwargs = {k:getattr(session, k) for k in ['tag', 'prec', 'trim', 'out_dir', 'overwrite']}
    with CalcStudyArea(session=session, name=name, **session.proj_lib[name]) as sa:
        yield sa

@pytest.fixture  #(scope='module')  needs to match
def extent(studyAreaWrkr):
    extent_layer = studyAreaWrkr.finv_vlay
    return studyAreaWrkr.layerextent(extent_layer, precision=5)

@pytest.fixture   
def dem_fp(studyAreaWrkr, tmp_path, extent):
    np.random.seed(100)
    return studyAreaWrkr.randomuniformraster(base_resolution, bounds=(0,5), extent=extent,
                                             output=os.path.join(tmp_path, 'dem_random.tif'))

@pytest.fixture
# were setup to filter out ground water... but tests are much simpler if we ignore this   
def wse_fp(studyAreaWrkr, tmp_path, extent):
    np.random.seed(100)
    return studyAreaWrkr.randomuniformraster(base_resolution, bounds=(5,7), extent=extent,
                                             output=os.path.join(tmp_path, 'wse_random.tif'))
    
@pytest.fixture   
def wd_rlay(session):
    """todo: convert this to not depend on a file"""
    wd_fp = r'C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\tests\hyd\data\wd\wd_rand_test_0304.tif'
    rlay = session.rlay_load(wd_fp)
    session.mstore.addMapLayer(rlay)
    return rlay
    

#===============================================================================
# TESTS STUDYAREA------
#===============================================================================
 
@pytest.mark.parametrize('aggLevel',[10, 50], indirect=False)  
@pytest.mark.parametrize('studyAreaWrkr',['testSet1'], indirect=True)     
def test_finv_gridPoly(studyAreaWrkr, aggLevel):
    """"this function is also tested in test_finv_agg"""
    finv_vlay = studyAreaWrkr.get_finv_clean()
    df, finv_agg_vlay = studyAreaWrkr.get_finv_gridPoly(aggLevel=aggLevel, finv_vlay=finv_vlay)
     
    assert isinstance(finv_agg_vlay, QgsVectorLayer)
    assert isinstance(df, pd.DataFrame)
     
     
    assert finv_vlay.dataProvider().featureCount() == len(df)
    assert finv_agg_vlay.dataProvider().featureCount() <= len(df)
     
    assert 'Polygon' in QgsWkbTypes().displayString(finv_agg_vlay.wkbType())


@pytest.mark.parametrize('studyAreaWrkr',['testSet1'], indirect=True) 
@pytest.mark.parametrize('dsampStage, resolution, downSampling',[
    ['none',base_resolution, 'none'], #raw... no rexampling
    ['post',30,'Average'],
    ['pre',30,'Average'],
    ['preGW',30,'Average'],
    ['post',30,'Maximum'],
    ['post',30,'Nearest neighbour'],
    ])  
def test_get_drlay(studyAreaWrkr, dsampStage, resolution, downSampling, 
                   dem_fp, wse_fp, #randomly generated rasters 
                   tmp_path):
    
    #===========================================================================
    # get calc result
    #===========================================================================
    d = studyAreaWrkr.get_drlay(
        wse_fp_d = {'hi':wse_fp},
        dem_fp_d = {base_resolution:dem_fp},
        resolution=resolution, downSampling=downSampling, dsampStage=dsampStage, trim=False)
    
    #{'rlay':rlay, 'noData_cnt':null_cnt}
    rlay = d.pop('rlay')
    #===========================================================================
    # check result----
    #===========================================================================
    #resulting stats
    stats_d = studyAreaWrkr.rlay_getstats(rlay)
    assert stats_d['resolution']==resolution
        
    #check nodata values
    assert hp.gdal.getNoDataCount(rlay.source())==0, 'should be true for our random rasters'
    assert rlay.crs() == studyAreaWrkr.qproj.crs()
    #stats_d = studyAreaWrkr.rlay_getstats(rlay)
    
    #===========================================================================
    # get the true depths
    #===========================================================================
    """what is the resolution of the test data??"""
    with RasterCalc(wse_fp, session=studyAreaWrkr, out_dir=tmp_path, logger=studyAreaWrkr.logger) as wrkr:
        #dep_rlay = wrkr.ref_lay
        wse_rlay = wrkr.ref_lay #loaded during init
        dtm_rlay = wrkr.load(dem_fp)
 
 
        entries_d = {k:wrkr._rCalcEntry(v) for k,v in {'wse':wse_rlay, 'dtm':dtm_rlay}.items()}
        
        formula = '{wse} - {dtm}'.format(**{k:v.ref for k,v in entries_d.items()})
        
        chk_rlay_fp = wrkr.rcalc(formula, report=False)
        
        assert hp.gdal.getNoDataCount(chk_rlay_fp)==0
        
        #get true stats
        stats2_d = studyAreaWrkr.rlay_getstats(chk_rlay_fp)
        

    #===========================================================================
    # compare
    #===========================================================================
    
    assert stats2_d['resolution']<=stats_d['resolution']
    
    if downSampling =='Average':
        #check averages (should be about the same)
        assert abs(stats2_d['MEAN'] - stats_d['MEAN']) <1.0
        
        #check extremes (true should always be more exreme)
        assert stats2_d['MAX'] >=stats_d['MAX'] 
        #assert stats2_d['MIN'] <=stats_d['MIN'] #doesnt work as we fill nulls w/ zeros
        assert stats2_d['RANGE']>=stats_d['RANGE']
        
    elif downSampling =='Maximum':
        """not sure why this isnt exact"""
        assert abs(stats2_d['MAX'] - stats_d['MAX']) <0.1, 'maximum values dont match'
        
        
    
    if resolution==0:
        for stat, val in {k:stats_d[k] for k in ['MAX', 'MIN']}.items():
            assert abs(val)<1e-3, stat
            
 
    
    
        
 
    
#===============================================================================
# tests Session-------
#===============================================================================
    

@pytest.mark.parametrize('aggType,aggLevel',[
    ['none',0], 
    ['gridded',20], 
    ['gridded',50],
    ['convexHulls', 5],
    ], indirect=False) 
def test_finv_agg(session, aggType, aggLevel, true_dir, write):
    #===========================================================================
    # #execute the functions to be tested
    #===========================================================================
    test_d = dict()
    dkey1 = 'finv_agg_d'
    test_d[dkey1] = session.build_finv_agg(dkey=dkey1, aggType=aggType, aggLevel=aggLevel, write=write)
    
    dkey2 = 'finv_agg_mindex'
    test_d[dkey2] =session.data_d[dkey2]
    
 
    
    for dkey, test in test_d.items():
        #=======================================================================
        # get the pickle corresponding to this test
        #=======================================================================
        true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
        assert os.path.exists(true_fp), 'failed to find match for %s'%dkey
        true = retrieve_data(dkey, true_fp, session)
        

        #=======================================================================
        # compare
        #=======================================================================
        assert len(test)==len(true)
        assert type(test)==type(true)
        
        if dkey=='finv_agg_d':
            """
            session.vlay_write(test['testSet1'], os.path.join(session.temp_dir, 'finv_agg.geojson'))
            """
            check_layer_d(test, true, test_data=False)
 
                
        elif dkey=='finv_agg_mindex':
            assert_frame_equal(test.to_frame(), true.to_frame())
            
            

@pytest.mark.parametrize('tval_type',[
    'uniform', 
    'footprintArea']) #rand is silly here. see test_stoch also
@pytest.mark.parametrize('normed', [True, False])
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
def test_04tvals_raw(session,true_dir, base_dir, write, 
               finv_agg_fn, tval_type, normed):
    norm_scale=1.0
    dkey='tvals_raw'
 
    #===========================================================================
    # load inputs   
    #===========================================================================
    #set the compiled references    
    session.compiled_fp_d = build_compileds({'finv_agg_mindex':finv_agg_fn},
                                            base_dir)
    
    finv_agg_mindex = session.retrieve('finv_agg_mindex')

    #===========================================================================
    # execute
    #===========================================================================
    
    finv_true_serx = session.build_tvals_raw(dkey=dkey, 
                                            norm_scale=norm_scale,
                                            tval_type=tval_type, 
                                            normed=normed,
                                            mindex =finv_agg_mindex, write=write)
    
 
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(finv_true_serx, true)


@pytest.mark.parametrize('finv_agg_fn, dscale_meth, tvals_raw',[ #have to combine finv_agg with correct tvals_raw output
        ['test_finv_agg_gridded_50_0', 'centroid', 'test_04tvals_raw_test_finv_agg0'], 
        ['test_finv_agg_none_None_0', 'none', 'test_04tvals_raw_test_finv_agg4'],
        ['test_finv_agg_gridded_50_0', 'area_split', 'test_04tvals_raw_test_finv_agg0'], 
                                        ], indirect=False)  #see test_finv_agg
def test_05tvals(session,finv_agg_fn, true_dir, base_dir, write, 
               tvals_raw, dscale_meth):
 
    
    
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
    finv_agg_serx = session.build_tvals(dkey=dkey, write=write,
                                    tvals_raw_serx=tvals_raw,
                                    finv_agg_d=finv_agg_d,
                                     
                                    dscale_meth=dscale_meth)
    
    #data checks
    #assert_index_equal(finv_agg_mindex.droplevel('id').drop_duplicates(), finv_agg_serx.index)
    
 
    #norm checks
    norm_scale=1.0 #for test proofing
    normed=True #should only pass tvals_raw that are normed
    if normed:
        """I'm not sure this needs to hold for all gridded inventories"""
        assert (finv_agg_serx.groupby(level='studyArea').sum().round(3)==norm_scale).all()
    #===========================================================================
    # retrieve true
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(finv_agg_serx, true)

 
@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
def test_sampGeo(session, sgType, finv_agg_fn, true_dir, write, base_dir):
    #===========================================================================
    # load inputs   
    #===========================================================================
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
        
    #===========================================================================
    # execute
    #===========================================================================
    dkey='finv_sg_d'
    vlay_d = session.build_sampGeo(dkey = dkey, sgType=sgType, finv_agg_d=finv_agg_d, write=write)
    
    #===========================================================================
    # retrieve trues    
    #===========================================================================
    
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # check
    #===========================================================================
    check_layer_d(vlay_d, true)




#===============================================================================
# Rsamp tests
#===============================================================================
#rsamps methods are only applicable for certain geometry types  
@pytest.mark.dev
@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_poly_test_finv_ag0','test_sampGeo_poly_test_finv_ag1',])
@pytest.mark.parametrize('samp_method',['zonal'], indirect=False)
@pytest.mark.parametrize('zonal_stat',['Mean','Minimum', 'Maximum'])  

def test_rsamps_poly(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir, zonal_stat,wd_rlay,
                     ):
 
    rsamps_runr(base_dir, true_dir, session, zonal_stat=zonal_stat,
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn,wd_rlay=wd_rlay,
                )
    
 
@pytest.mark.parametrize('finv_sg_d_fn',[ #see test_sampGeo
    'test_sampGeo_centroids_test_fi1','test_sampGeo_centroids_test_fi0'])
@pytest.mark.parametrize('samp_method',['points'], indirect=False) 
def test_rsamps_point(session, finv_sg_d_fn,samp_method, true_dir, write, base_dir, wd_rlay):
 
    rsamps_runr(base_dir, true_dir,session, wd_rlay=wd_rlay,
                samp_method=samp_method, write=write, finv_sg_d_fn=finv_sg_d_fn)
    
    
    



@pytest.mark.parametrize('finv_agg_fn',['test_finv_agg_gridded_50_0', 'test_finv_agg_none_None_0'], indirect=False)  #see test_finv_agg
@pytest.mark.parametrize('sgType',['centroids', 'poly'], indirect=False)  
@pytest.mark.parametrize('samp_method',['true_mean'], indirect=False) 
def test_rsamps_trueMean(session, finv_agg_fn, samp_method, true_dir, write, base_dir, sgType, wd_rlay):
    #===========================================================================
    # build the sample geometry
    #===========================================================================
    """because true_mean requires the raw inventory.. 
        for this test we perform the previous calc (sample geometry) as well
        to simplify the inputs"""
    finv_agg_d, finv_agg_mindex = retrieve_finv_d(finv_agg_fn, session, base_dir)
    
    finv_sg_d = session.build_sampGeo(dkey = 'finv_sg_d', sgType=sgType, finv_agg_d=finv_agg_d, write=False)
 
    #===========================================================================
    # execute
    #===========================================================================        
 
    rsamps_runr(base_dir, true_dir,session, samp_method=samp_method, write=write, 
                finv_sg_d=finv_sg_d, mindex=finv_agg_mindex, wd_rlay=wd_rlay)

def rsamps_runr(base_dir, true_dir,session,finv_sg_d=None,finv_sg_d_fn=None, wd_rlay=wd_rlay, **kwargs):
    """because kwarg combinations are complex for rsamps... its easier to split out the tests"""
    #===========================================================================
    # load inputs   
    #===========================================================================
    if finv_sg_d is None:
        dkey = 'finv_sg_d'
        input_fp = search_fp(os.path.join(base_dir, finv_sg_d_fn), '.pickle', dkey) #find the data file.
        finv_sg_d = retrieve_data(dkey, input_fp, session)

 
    #===========================================================================
    # execute
    #===========================================================================
    saName = list(session.proj_lib.keys())[0]
    
    dkey='rsamps'
    rsamps_serx = session.build_rsamps(dkey=dkey, finv_sg_d=finv_sg_d, 
                                       drlay_d={saName:wd_rlay},
                                        **kwargs)
    
 
    #===========================================================================
    # retrieve trues    
    #===========================================================================
    true_fp = search_fp(true_dir, '.pickle', dkey) #find the data file.
    true = retrieve_data(dkey, true_fp, session)
    
    #===========================================================================
    # compare
    #===========================================================================
    assert_series_equal(rsamps_serx, true)


            
            
    
