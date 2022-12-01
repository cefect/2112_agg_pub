'''
Created on Oct. 22, 2022

@author: cefect

2d profile example of wd averaging
'''
import os, pathlib, itertools, logging, sys, datetime
import pandas as pd
import numpy as np
import numpy.ma as ma
import scipy
idx = pd.IndexSlice
from definitions import wrk_dir
#===============================================================================
# setup matplotlib----------
#===============================================================================
output_format='pdf'
usetex=True
if usetex:
    os.environ['PATH'] += R";C:\Users\cefect\AppData\Local\Programs\MiKTeX\miktex\bin\x64"
 
cm = 1/2.54
import matplotlib
#matplotlib.use('Qt5Agg') #sets the backend (case sensitive)
matplotlib.set_loglevel("info") #reduce logging level
import matplotlib.pyplot as plt
  
#set teh styles
plt.style.use('default')
  
#font
matplotlib.rc('font', **{
        'family' : 'serif', 
        'weight' : 'normal',
        'size'   : 8})
  
  
for k,v in {
    'axes.titlesize':10,
    'axes.labelsize':10,
    'xtick.labelsize':8,
    'ytick.labelsize':8,
    'figure.titlesize':12,
    'figure.autolayout':False,
    'figure.figsize':(17*cm,3.5*cm),
    'legend.title_fontsize':'large',
    'text.usetex':usetex,
    }.items():
        matplotlib.rcParams[k] = v
   
print('loaded matplotlib %s'%matplotlib.__version__)
#===============================================================================
# setup logger
#===============================================================================
logging.basicConfig(
                #filename='xCurve.log', #basicConfig can only do file or stream
                force=True, #overwrite root handlers
                stream=sys.stdout, #send to stdout (supports colors)
                level=logging.INFO, #lowest level to display
                )


#===============================================================================
# funcs
#===============================================================================
from agg2.coms import Agg2Session, Agg2DAComs

class AggExample(Agg2Session, Agg2DAComs):
    def get_wsh_vals(self, cnt=4, mask_zeros=False):
        """build set of aggregated grid values
        
        Parameters
        ----------
        cnt: int
            two power. count of set. number of aggregations
        """
        log = self.logger.getChild('wsh')
        #build the scales from this
        scales_l = [2**(i) for i in range(cnt)]
        smax = max(scales_l)
        #assert smax==2**cnt
        log.info(f'on {cnt} scales:\n    {scales_l}')
        
        """implied when we convert to pandas
        #get x domain
        x_ar = np.linspace(0,2**cnt,num=2**cnt, endpoint=False)"""
        
        #===========================================================================
        # get fine y
        #===========================================================================
     
        
        domain = smax*2*2
        perturb = 2**3
        ys1_ar = np.concatenate(
            [np.full(domain//2-perturb//2, 0), #dry
              np.linspace(0,1,num=perturb//2, endpoint=True),
              #np.full(domain//2, 1), #wet
              np.linspace(1,0,num=perturb//2, endpoint=True),
              np.full(domain//2-perturb//2, 0), #dry            
                ]
            )
        
        log.info(ys1_ar.shape)
        
        if mask_zeros:
            ys1_mar = ma.array(ys1_ar, mask=ys1_ar==0)
        else:
            ys1_mar = ma.array(ys1_ar, mask=False)
        #===========================================================================
        # build aggregates
        #===========================================================================
        d=dict()
        for i, s2 in enumerate(scales_l):
            log.info(f'aggregating {s2}')
            #split and aggregagte
            """using half scales because we mirror below"""
            try:
                ys2_ar = ys1_mar.reshape(len(ys1_mar)//s2, -1).mean(axis=1)
            except Exception as e:
                raise IOError(s2)
            
            #disaggregate
            
           
            d[s2] = scipy.ndimage.zoom(np.where(~ys2_ar.mask,ys2_ar.data,  np.nan), s2, order=0, mode='reflect',   grid_mode=True)
            
        d[1] = ys1_ar
        #===========================================================================
        # merge
        #===========================================================================
        log.info(f'built {len(d)}')
        
        df= pd.DataFrame(d)
        
        return df
    
        df.plot()
     
        
    def plot_profile(self, dx,
                     lw_multiplier=2.0,
                     title_d = {'zeros':'(a)', 'nulls':'(b)'},
                     **kwargs):
        
        #===========================================================================
        # defaults
        #===========================================================================
        self.logger
        log, tmp_dir, out_dir, ofp, _, write = self._func_setup('prof',  subdir=False,ext='.'+output_format, **kwargs)
 
        
    
        
        """
        dx.plot(colormap=cmap)
        """
     
        mdex=dx.columns
        map_d = {'col':'method', 'color':'s2'}
        keys_all_d = {k:mdex.unique(v).tolist() for k,v in map_d.items()}
     
        #===========================================================================
        # setup plot
        #===========================================================================
        fig = plt.figure( 
                    constrained_layout=True, )
        
        ax_ar = fig.subplots(nrows=1, ncols=len(keys_all_d['col']), sharex='all', sharey='all')    
        
        ax_d = dict(zip(keys_all_d['col'], ax_ar))
        
        #===========================================================================
        # #colors
        #===========================================================================
        #retrieve the color map
        cmap = plt.cm.get_cmap(name='copper')
        
        #buidl the normalizer for the data
        norm = matplotlib.colors.Normalize(vmin=min(keys_all_d['color']), vmax=max(keys_all_d['color']))
        color_d = {i:cmap(ni) for i, ni in enumerate(np.linspace(0, 1, len(keys_all_d['color'])))}
        
        fillStyles_l = ['left', 'right', 'bottom', 'top']
     
        #===========================================================================
        # loop and plot
        #===========================================================================
        for gcol, gdx in dx.groupby(map_d['col'], axis=1):
            ax = ax_d[gcol]
            
            #gdx.droplevel(map_d['col'], axis=1).plot(ax=ax, colormap=cmap)
            
            for i, (s2, ser) in enumerate(gdx.droplevel(map_d['col'], axis=1).items()):
                
                ni = norm(s2)
                if not i==0:
                    pkwargs = dict(color=color_d[i], 
                                   #linewidth=3.0-lw_multiplier*ni,
                                   #marker=plt.Line2D.filled_markers[i],
                                   #marker=f'${s2}$',
                                   marker='o', fillstyle=fillStyles_l[i], markeredgewidth=0.1,
                                   linewidth=0.5,
                                   alpha=0.8,
                                   label=f'$s2={s2}$')
                else:
                    pkwargs = dict(color='black', 
                                   #linewidth=0.5,
                                   linewidth=1.0, linestyle='dashed',
                                   #marker='o', markersize=10.0, fillstyle='none'
                                   label='$s1$',
                                   )
                ax.plot(ser, **pkwargs)
                
            #===================================================================
            # post
            #===================================================================
            #add an arrow
            #ax.arrow(ser.index.values.mean(), gdx.max().max(), 0, -(gdx.max().max()-gdx.min().min()), width=0.2, color='black')
            
            ax.arrow(ser.index.values.mean(), 1.0, 0, -(1.0 - ser.max()), 
                     width=0.4,alpha=0.5, color='black',
                     head_length=0.10,  head_width=1.0, 
                     head_starts_at_zero=False, length_includes_head=True)
                
            ax.axis('off') #turn off the spines and tick marks
            ax.set_title(title_d[gcol], loc='left', y=0.8)
            ax.text(0.05, 0.1, '$y=0$', transform=ax.transAxes, va='bottom', ha='left',fontsize=8, color='black')
            
        #===========================================================================
        # legend
        #===========================================================================
        handles, labels = ax.get_legend_handles_labels()
        
        fig.legend(handles, labels)
        
        #===========================================================================
        # write
        #===========================================================================
        return self.output_fig(fig, ofp=ofp, logger=log, fmt=output_format, add_stamp=False)
    """
    plt.show()
    """
 
 
def run_exmp_plots(**kwargs):
    
    with AggExample(logger=logging.getLogger('r'),
                    out_dir=os.path.join(wrk_dir, 'outs', 'da'),
                    fancy_name='_'.join(['agg_exmp', datetime.datetime.now().strftime('%m%d')]),
                    
                     **kwargs) as ses:
    
        df1 = ses.get_wsh_vals()
        df2 = ses.get_wsh_vals(mask_zeros=True)
        
        dx=pd.concat({'zeros':df1, 'nulls':df2}, axis=1, names=['method', 's2'])
    
    
        output = ses.plot_profile(dx)
        
    return output
    
 
    
 
 

if __name__ == "__main__":
    run_exmp_plots()
 
    
    print('finished')