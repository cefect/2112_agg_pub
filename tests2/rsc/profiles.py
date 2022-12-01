'''
Created on Aug. 21, 2022

@author: cefect
'''
import os, pstats, cProfile
 

from definitions import proj_lib, wrk_dir
from agg.hydR.dsc.scripts import runr
from hp.prof import prof_simp


def SJ_0821():
    proj_name = 'SJ'
    proj_d = proj_lib[proj_name]
    return runr(
        proj_name=proj_name, run_name='r2',
        downscale=50,
        wse_fp=proj_d['wse_fp_d']['hi'],        
        dem_fp=proj_d['dem_fp_d'][1]
        )
    


    
if __name__ == "__main__":
    prof_simp('SJ_0821()')