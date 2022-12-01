'''
Created on May 8, 2022

@author: cefect
'''
import sys, argparse, datetime
start = datetime.datetime.now()
 
from agg.hydE.hydE_runr import run

if __name__ == "__main__":
    print(sys.argv)
 
    #===========================================================================
    # setup argument parser 
    #===========================================================================
    parser = argparse.ArgumentParser(prog='hydR',description='execute hydR.models')
    #add arguments
    parser.add_argument('-tag','-t', help='run label', type=str)
    parser.add_argument("-name",'-n', help='name for the run group', type=str, default='hydR') #this defaults to None if not passed
    parser.add_argument("-write",'-w', help='flag for writing intermediate data', action='store_false') #defaults to True
    parser.add_argument("-write_lib",'-wl', help='flag for storing results to library', action='store_false') #defaults to True
    parser.add_argument("-catalog_fp",help='optional file path of catalog', type=str, default=None)
    
    
    #hydR (depths)
    parser.add_argument("-iters",'-i', help='resolution iterations', type=int, default=7)  
    parser.add_argument("-dsampStage",help='raster downsampling stage', type=str, default='pre')  
    parser.add_argument("-downSampling",help='raster downsampling GDAL.warp method', type=str, default='Average')
    
    #hydE (expo)
    parser.add_argument("-aggType",help='asset aggregation type', type=str, default='convexHulls')  
    parser.add_argument("-aggIters",help='number of aggLevels to execute', type=int, default=5)
    parser.add_argument("-samp_method",help='method for sampling rasters w/ finvs', type=str, default='zonal')
    parser.add_argument("-zonal_stat",help='for samp_method=zonal, statistc to apply', type=str, default='Mean')
    parser.add_argument("-index_col_n", '-idcn',help='index columns on the catalog', type=int, default=None)
     
    
    
    args = parser.parse_args()
    kwargs = vars(args)
    print('parser got these kwargs: \n    %s'%kwargs) #print all the parsed arguments in dictionary form
    
    
    n = kwargs.pop('index_col_n')
    if not n is None:
        kwargs['index_col'] = list(range(n))
    
 
    print('\n\nSTART (tag=%s) \n\n\n\n'%kwargs['tag'])
    run(**kwargs)
    
    tdelta = datetime.datetime.now() - start
    print('finished in %s' % (tdelta))
    
