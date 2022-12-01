'''
Created on May 12, 2022

@author: cefect

testing hrast specific callers
    see hyd.tests_expo for related tests
'''
import os, math
import pytest
print('pytest.__version__:' + pytest.__version__)
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal, assert_index_equal
idx = pd.IndexSlice

from hp.gdal import array_to_rlay
from agg.hydR.hydR_scripts import RastRun

prec=3

class Session(RastRun):
    
    def ar2lib(self, #get a container of a raster layer in RastRun style
               ar,
               resolution=10.0,
               studyArea='testSA'):
        
        log = self.logger.getChild('ar2lib')
        rlay_fp = array_to_rlay(ar, resolution=resolution)
        rlay =  self.get_layer(rlay_fp, mstore=self.mstore)
        log.info('built %s to \n    %s'%(ar.shape, rlay_fp))
        
        return {resolution:{studyArea:rlay}}
        

@pytest.fixture
def ses(tmp_path,write,logger, feedback, #session for depth runs
        array,
        resolution):
 
    np.random.seed(100)
    with Session(out_dir = tmp_path,  wrk_dir=None, prec=prec,
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        ses.data_d['drlay_lib'] = ses.ar2lib(array, resolution=resolution)
 
        yield ses
        


        
@pytest.fixture
def resolution(request):
    return request.param

@pytest.fixture
def array(request):
    return request.param
 

#===============================================================================
# DEPTH RASERTS------
#===============================================================================

@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5)),
     np.array([(1,np.nan),(1,2)]), 
     ], indirect=True)  
def test_rstats(ses, array):
    ar = array.copy()
    #===========================================================================
    # raster calc
    #===========================================================================
    dkey='rstats'
    
    rserx = ses.build_stats(dkey=dkey).iloc[0,:]
    
 
    #===========================================================================
    # check
    #===========================================================================
    #methods
    df = pd.DataFrame(ar)
 
    for rsColn, npMeth in {
        'MAX':'max',  'MIN':'min', 'SUM':'sum'
        }.items():
        """numpy methods dont have nan support
        assert round(rserx[rsColn], prec) ==  round(getattr(ar, npMethod)(), prec), rsColn"""
        ser = getattr(df, npMeth)(axis=1)
        npVal = round(getattr(ser, npMeth)(), prec)
        assert round(rserx[rsColn], prec) ==  npVal, rsColn
        
    #average
    """sequence matters here... cant use builtin"""
    npVal=df.sum().sum()/df.notna().sum().sum()
    assert round(rserx['MEAN'], prec) ==  round(npVal, prec), rsColn
 
    #nodata
    assert np.isnan(ar).sum()==rserx['noData_cnt']


@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_wetStats(ses, array, resolution):
    dkey='wetStats'
    rserx=ses.retrieve(dkey).iloc[0,:]
    
    wetArray = array[array>0.0]
    
    #wetcount
    wet_cnt = (array>0.0).sum()
    assert wet_cnt==rserx['wetCnt']
    
    #wetArea
    assert rserx['wetArea']==wet_cnt*(resolution**2)
    
    
    #volumes
 
    npVol = wetArray.sum()*(resolution**2)
    assert round(rserx['wetVolume'], prec)==round(npVol, prec)
    
    #mean
    npMean = wetArray.mean()
    assert round(rserx['wetMean'], prec)==round(npMean, prec)
    
    
 
@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_gwArea(ses, array, resolution):
    dkey='gwArea'
    rserx=ses.retrieve(dkey).iloc[0,:]
    
    #wetcount
    wet_cnt = (array<0.0).sum()
 
    
    #wetArea
    assert rserx['gwArea']==wet_cnt*(resolution**2)

#===============================================================================
# SUBTRACTION CALCS ----------
#===============================================================================
@pytest.fixture
def wse_ar(request):
    return request.param
@pytest.fixture
def dem_ar(request):
    return request.param

@pytest.fixture
def base_resolution(request):
    return request.param

@pytest.fixture
def ses_sub(tmp_path,write,logger, feedback, #session for depth runs
        wse_ar, dem_ar,
        base_resolution):
    
    assert np.array_equal(np.array(wse_ar.shape),np.array(dem_ar.shape))
 
    np.random.seed(100)
    with Session(out_dir = tmp_path,  wrk_dir=None, prec=prec,trim=False,
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        
        #build a proj_lib
        epsg = int(ses.qproj.crs().authid().replace('EPSG:', ''))
        
        def glay(ar, **kwargs):
            return array_to_rlay(ar, resolution=base_resolution, epsg=epsg,out_dir=tmp_path, **kwargs)
        
        ses.proj_lib = {'testSA':{
            'wse_fp_d':{'hi':glay(wse_ar, layname='wse')},
            'dem_fp_d':{base_resolution:glay(dem_ar, layname='dem')},
            'EPSG':epsg
            }}
 
 
        yield ses
        
bsize=4
def get_sparse():
    ar = np.random.random((bsize,bsize)).reshape(bsize**2)
    
    rand_e = np.random.choice(ar, size=1)[0]
    
    itemIndex = np.nonzero(np.isclose(ar, rand_e))[0][0]
 
    
    ar[itemIndex] = np.nan
    
    return ar.reshape(bsize, bsize)
    
@pytest.mark.dev
@pytest.mark.parametrize('base_resolution', [10.0], indirect=True)
@pytest.mark.parametrize('resolution_scale', [2])
@pytest.mark.parametrize('sequenceType', ['none', 'inputs', 'outputs'])
@pytest.mark.parametrize('dsampStage', ['pre', 'post', 
                                        #'preGW', 'postFN',
                                        ])

@pytest.mark.parametrize('dem_ar',[
      np.random.random((bsize,bsize))-0.5,
     
     ], indirect=True)  

@pytest.mark.parametrize('wse_ar',[
     #np.random.random((2,2))-0.1,
     get_sparse()
     ], indirect=True)  
def test_sequence(ses_sub, dem_ar,wse_ar, resolution_scale,sequenceType,
                  dsampStage,base_resolution):
    dkey='drlay_lib'
    res_lib=ses_sub.retrieve(dkey, resolution_scale=resolution_scale,
                       sequenceType=sequenceType, dsampStage=dsampStage,
                       iters=2)
    
    
    #===========================================================================
    # build from array-------
    #===========================================================================
    #===========================================================================
    # helpers
    #===========================================================================
    def subtract(ar_top, ar_bot):
        wdf = pd.DataFrame(ar_top)
        dem_df = pd.DataFrame(ar_bot)
        dep_df = wdf.subtract(dem_df)
        
        assert dep_df.isna().sum().sum()==wdf.isna().sum().sum()
        
        return dep_df.values
    
    def downsamp(ar_l):
        return [pd.DataFrame(ar).mean(skipna=True).values for ar in ar_l]
    
    #===========================================================================
    # loop each resolution
    #===========================================================================
    #start initials
    res_ar_d=dict()
    wse_ar_k, dem_ar_k=wse_ar, dem_ar #store for first
    dep_ar_k = np.nan_to_num(subtract(wse_ar, dem_ar))
    
    
    
    for res in res_lib.keys():
        #first
        if res==base_resolution:
            res_ar_d[res] = dep_ar_k
            continue
        print("resolution=%i"%res)
        
        # base dim * scaleVal = agg dim
        scaleVal = base_resolution/res
        newDim = int(len(wse_ar)*scaleVal)
        #=======================================================================
        # #select inputs from sequence
        #=======================================================================
        if sequenceType=='none':
            wse_ar_i, dem_ar_i = wse_ar, dem_ar
        elif sequenceType=='inputs':
            wse_ar_i, dem_ar_i =  wse_ar_k, dem_ar_k
        elif sequenceType=='outputs':
            dep_ar_l = np.split(dep_ar_k, chk_size, axis=0)
            res_ar_d[res]=np.concatenate(downsamp(dep_ar_l))
            dep_ar_k = res_ar_d[res]
            continue
            
        
        #=======================================================================
        # chunk out the arrays
        #=======================================================================
        def chunk(ar):
            ar.reshape(newDim, int(ar.size/float(newDim)))
            raise IOError('stopped here')
            ar_flat = ar.reshape(ar.size)
            d = dict()
            row=-1
            dcol=0
            for i, val in enumerate(ar_flat):
                col = i%len(ar)
                if i%len(ar)==0:row+=1
                
                print('%i: row=%i, col=%i'%(i, row, col))
                
                print(col%newDim)
                if col%newDim==0:
                    dcol+=1
                print('dcol=%i'%dcol)
                
                
                if not row in d: d[row] = dict()
                if not col in d[row]: d[row][col]=list()
                k = i%(1/scaleVal)**2 #block indexer
                
                if not k in d: d[k] = list()
                
                d[k].append(ari)
                
            np.stack(l, axis=0)
            
            flat_l = np.split(ar.reshape(ar.size), (1/scaleVal)**2, axis=0)
            return 
 
        #divide the data into chunks (for raster-like ops)
        wsei_ar_l = chunk(wse_ar_i)
        demi_ar_l = np.split(dem_ar_i, chk_size, axis=0)
        
        #=======================================================================
        # pre-subtraction downsampling
        #=======================================================================

        
        if dsampStage=='pre':
            
            wsei_ar_l1 = downsamp(wsei_ar_l)
            demi_ar_l1 = downsamp(demi_ar_l)  
            
        else:
            wsei_ar_l1 = wsei_ar_l
            demi_ar_l1 = demi_ar_l
            
            
        #=======================================================================
        # subtraction
        #=======================================================================

            
        
        depth_ar_l=[]
        for wse_ar_ii, dem_ar_ii in zip(wsei_ar_l1, demi_ar_l1):

            depth_ar_l.append(subtract(wse_ar_ii, dem_ar_ii))
            
        #=======================================================================
        # post subtraction downsample
        #=======================================================================
        if dsampStage=='post':
            depth_ar_l1 = downsamp(depth_ar_l)
        else:
            depth_ar_l1 = depth_ar_l
        
        #=======================================================================
        # set for next loop
        #=======================================================================
        dep_ar = np.concatenate(depth_ar_l1)
        
        wse_ar_k = np.concatenate(wsei_ar_l1)
        dem_ar_k = np.concatenate(demi_ar_l1)
        dep_ar_k = dep_ar
        res_ar_d[res]=dep_ar
        
    #===========================================================================
    # compare
    #===========================================================================
    pd.concat({k: pd.DataFrame(ar) for k,ar in res_ar_d.items()})
    raise IOError('stopped here')
    
    res_lib
        
                
    
 
 
    
    #wetArea
    assert rserx['gwArea']==wet_cnt*(resolution**2)
    

#===============================================================================
# DIFFERENCE RASTERS---------
#===============================================================================
@pytest.fixture
def ses_diff(tmp_path,write,logger, feedback, #session for difference rastser runs
        array,
        resolution):
 
    np.random.seed(100)
    with Session(out_dir = tmp_path,  wrk_dir=None, prec=prec,
                     overwrite=write,write=write, logger=logger,feedback=feedback,
                     driverName='GeoJSON', #nicer for writing small test datasets
                     ) as ses:
        
        assert len(ses.data_d)==0
        ses.data_d['difrlay_lib'] = ses.ar2lib(array, resolution=resolution)
 
        yield ses
        
        
 
@pytest.mark.parametrize('resolution', [10.0])
@pytest.mark.parametrize('array',[
     np.random.random((5,5))-0.5,
     np.random.random((1,2))*10,
     np.random.random((5,5)),
     #np.array([(1,np.nan),(1,2)]),  #depth raster... no nulls
     ], indirect=True)  
def test_rmseD(ses_diff, array, resolution):
    """
    RastRun.build_rmseD()
    """
    dkey='rmseD'
    rserx=ses_diff.retrieve(dkey).iloc[0,:]
 
    rmse = np.sqrt(np.mean(np.square(array)))
    
 
 
    
    #wetArea
    assert round(rserx[0], prec)==round(rmse, prec)
 
     
 