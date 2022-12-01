'''
Created on May 8, 2022

@author: cefect
'''
import os, datetime, math, pickle, copy, sys
import numpy as np
np.random.seed(100)

start = datetime.datetime.now()
print('start at %s' % start)
from agg.hydR.hydR_scripts import RastRun


def run( #run a basic model configuration
        #=======================================================================
        # #generic
        #=======================================================================
        tag='tag',
        name='hrast0',
        overwrite=True,
        trim=False,

        #=======================================================================
        # write control
        #=======================================================================
        write=True,
        exit_summary=True,
        write_lib=True, #enter the results into the library
 
        compression='med',
        #=======================================================================
        # #data
        #=======================================================================
        studyArea_l = None, #convenience filtering of proj_lib
        proj_lib = None,
        
        #optional loading data from the catalog
        catalog_fp=None,
        
        #=======================================================================
        # session pars
        #=======================================================================
        prec=3,        

        #=======================================================================
        # #parameters
        #=======================================================================
        iters=3, #resolution iterations
        #raster downSampling and selection  (StudyArea.get_raster())
        dsampStage='pre', downSampling='Average', severity = 'hi', 
        #resolution=5, this is what we iterate on
        sequenceType='none', #how to construct layers consecutively

        #=======================================================================
        # debug
        #=======================================================================
        debug_max_len=None,phase_l=['depth', 'diff'],

        **kwargs):
    print('START run w/ %s.%s and '%(name, tag))
 
    #===========================================================================
    # study area filtering
    #===========================================================================
    if proj_lib is None:
        from definitions import proj_lib
    
    if not studyArea_l is None:
        print('filtering studyarea to %i: %s'%(len(studyArea_l), studyArea_l))
        miss_l = set(studyArea_l).difference(proj_lib.keys())
        assert len(miss_l)==0, 'passed %i studyAreas not in proj_lib: %s'%(len(miss_l), miss_l)
        proj_lib = {k:v for k,v in proj_lib.items() if k in studyArea_l}
        
    id_params=dict(downSampling=downSampling, dsampStage=dsampStage, severity=severity, sequenceType=sequenceType)
    #===========================================================================
    # execute
    #===========================================================================
    with RastRun(tag=tag,proj_lib=proj_lib,overwrite=overwrite, trim=trim, name=name,
                     write=write,exit_summary=exit_summary,prec=prec, 
                 bk_lib = {
                     'drlay_lib':dict( severity=severity, downSampling=downSampling, dsampStage=dsampStage, iters=iters, sequenceType=sequenceType),
                     'res_dx':dict(phase_l=phase_l), 
                     'dataExport':dict(compression=compression, debug_max_len=debug_max_len, phase_l=phase_l, id_params=id_params), 
          
                     },
                 **kwargs) as ses:
        
        #
        if not catalog_fp is None:
            ses.compileFromCat(catalog_fp=catalog_fp,id_params=id_params)
            
 
        if 'depth' in phase_l:
            ses.runDownsample()
        
        if 'diff' in phase_l:
            ses.runDiffs()
        
        ses.retrieve('res_dx')
 
        
        if write_lib:
            ses.write_lib(id_params=id_params)

    print('\nfinished %s'%tag)
    
    return 




 

 
def r02(**kwargs):
    rkwargs = dict(
        iters=8, downSampling='Average',write_lib=True, 
        #catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv',
        phase_l=['depth', 'diff']
        )    
    return run(name='hydR02', **{**rkwargs, **kwargs})

def r03(**kwargs):
    rkwargs = dict(
        iters=8, downSampling='Average',write_lib=True, 
        #catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv',
        phase_l=['depth', 'diff']
        )    
    return run(name='hydR03', **{**rkwargs, **kwargs})

def postFN():
    return r01(
        dsampStage='postFN',tag='postFN',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\drlay_lib_hydR01_postFN_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\noData_cnt_hydR01_postFN_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\rstats_hydR01_postFN_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\wetStats_hydR01_postFN_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\gwArea_hydR01_postFN_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\res_dx_hydR01_postFN_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\postFN\20220514\working\layxport_hydR01_postFN_0514.pickle',
            }
        )

def post():
    return r01(
        dsampStage='post',tag='post',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\drlay_lib_hydR01_post_0514.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\noData_cnt_hydR01_post_0514.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\rstats_hydR01_post_0514.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\wetStats_hydR01_post_0514.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\gwArea_hydR01_post_0514.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\res_dx_hydR01_post_0514.pickle',
        'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\post\20220514\working\layxport_hydR01_post_0514.pickle',
            }
        )
    
def pre():
    return r01(
        dsampStage='pre',tag='pre',
        #studyArea_l=['obwb'],iters=5,
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\drlay_lib_hydR02_pre_0515.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\noData_cnt_hydR02_pre_0515.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\rstats_hydR02_pre_0515.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\wetStats_hydR02_pre_0515.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\gwArea_hydR02_pre_0515.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\difrlay_lib_hydR02_pre_0515.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\rstatsD_hydR02_pre_0515.pickle',
        'rmseD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\rmseD_hydR02_pre_0515.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre\20220515\working\res_dx_hydR02_pre_0515.pickle',
 
            }
        )
    
def preGW():
    return r01(
        dsampStage='preGW',tag='preGW',
        compiled_fp_d={
        #=======================================================================
        # 'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\drlay_lib_hydR01_preGW_0514.pickle',
        # 'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\noData_cnt_hydR01_preGW_0514.pickle',
        # 'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\rstats_hydR01_preGW_0514.pickle',
        # 'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\wetStats_hydR01_preGW_0514.pickle',
        # 'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\gwArea_hydR01_preGW_0514.pickle',
        # 'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\res_dx_hydR01_preGW_0514.pickle',
        # 'layxport':r'C:\LS\10_OUT\2112_Agg\outs\hydR01\preGW\20220514\working\layxport_hydR01_preGW_0514.pickle',
        #=======================================================================
            }
        )
    
def pre_nn():
    return r01(
        dsampStage='pre',tag='pre_nn', 
        downSampling='nn',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\drlay_lib_hydR02_pre_nn_0515.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\noData_cnt_hydR02_pre_nn_0515.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\rstats_hydR02_pre_nn_0515.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\wetStats_hydR02_pre_nn_0515.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\gwArea_hydR02_pre_nn_0515.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\difrlay_lib_hydR02_pre_nn_0515.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\rstatsD_hydR02_pre_nn_0515.pickle',
        'rmseD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\rmseD_hydR02_pre_nn_0515.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\pre_nn\20220515\working\res_dx_hydR02_pre_nn_0515.pickle',
 
            }
        )
    
def post_nn():
    return r02(
        dsampStage='post',tag='post_nn', 
        downSampling='nn',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\drlay_lib_hydR02_post_nn_0516.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\noData_cnt_hydR02_post_nn_0516.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\rstats_hydR02_post_nn_0516.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\wetStats_hydR02_post_nn_0516.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\gwArea_hydR02_post_nn_0516.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\difrlay_lib_hydR02_post_nn_0516.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\rstatsD_hydR02_post_nn_0516.pickle',
        'rmseD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\rmseD_hydR02_post_nn_0516.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\res_dx_hydR02_post_nn_0516.pickle',
        'dataExport':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\post_nn\20220516\working\dataExport_hydR02_post_nn_0516.pickle',


            }
        )

def postFN_nn():
    return r02(
        dsampStage='postFN',tag='postFN_nn', 
        downSampling='nn',
        compiled_fp_d={
            'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\drlay_lib_hydR02_postFN_nn_0516.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\noData_cnt_hydR02_postFN_nn_0516.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\rstats_hydR02_postFN_nn_0516.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\wetStats_hydR02_postFN_nn_0516.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\gwArea_hydR02_postFN_nn_0516.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\difrlay_lib_hydR02_postFN_nn_0516.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\postFN_nn\20220516\working\rstatsD_hydR02_postFN_nn_0516.pickle',


            }
        )
    
def preGW_nn():
    return r02(
        dsampStage='preGW',tag='preGW_nn', 
        downSampling='nn',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\drlay_lib_hydR02_preGW_nn_0516.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\noData_cnt_hydR02_preGW_nn_0516.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\rstats_hydR02_preGW_nn_0516.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\wetStats_hydR02_preGW_nn_0516.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\gwArea_hydR02_preGW_nn_0516.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\difrlay_lib_hydR02_preGW_nn_0516.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\rstatsD_hydR02_preGW_nn_0516.pickle',
                'rmseD':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\rmseD_hydR02_preGW_nn_0516.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\res_dx_hydR02_preGW_nn_0516.pickle',
        'dataExport':r'C:\LS\10_OUT\2112_Agg\outs\hydR02\preGW_nn\20220516\working\dataExport_hydR02_preGW_nn_0516.pickle',

 
            }
        )

def post_Sins():
    return r03(tag='post_Sins', 
        dsampStage='post',downSampling='Average',sequenceType='inputs',
        compiled_fp_d={
        'drlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\drlay_lib_hydR03_post_Sins_0517.pickle',
        'noData_cnt':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\noData_cnt_hydR03_post_Sins_0517.pickle',
        'rstats':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\rstats_hydR03_post_Sins_0517.pickle',
        'wetStats':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\wetStats_hydR03_post_Sins_0517.pickle',
        'gwArea':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\gwArea_hydR03_post_Sins_0517.pickle',
        'difrlay_lib':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\difrlay_lib_hydR03_post_Sins_0517.pickle',
        'rstatsD':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\rstatsD_hydR03_post_Sins_0517.pickle',
        'rmseD':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\rmseD_hydR03_post_Sins_0517.pickle',
        'res_dx':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\res_dx_hydR03_post_Sins_0517.pickle',
        'dataExport':r'C:\LS\10_OUT\2112_Agg\outs\hydR03\post_Sins\20220517\working\dataExport_hydR03_post_Sins_0517.pickle',

 
            }
        )
def dev():
    return run(
        trim=True, compression='none',name='hydRd',write_lib=True,
        tag='dev',
        iters=2,
        sequenceType='outputs',
        #dsampStage='postFN',
        #downSampling='Nearest neighbour',
        compiled_fp_d={
 

            },
        #catalog_fp=r'C:\LS\10_OUT\2112_Agg\lib\hydRd\hydRd_run_index.csv',
        #studyArea_l=['obwb'],
        phase_l=['depth', 'diff']
        )
    
if __name__ == "__main__": 
    
    #dev()
 
    #post()
    #===========================================================================
    # postFN()
    #pre()
    #===========================================================================
    #preGW()
    #pre_nn()
    #post_nn()
    #postFN_nn()
    #preGW_nn()
    post_Sins()
    

    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))