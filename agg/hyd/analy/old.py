'''
Created on Apr. 23, 2022

@author: cefect
'''
class Old(object):
    def xxxplot_total_bars(self, #generic total bar charts
                        
                    #data
                    dkey_d = {'rsamps':'mean','tvals':'var','tloss':'sum'}, #{dkey:groupby operation}
                    dx_raw=None,
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_rown='dkey',
                    plot_coln='studyArea',
                    plot_bgrp='modelID',
                    plot_colr=None,
                    sharey='row',
                    
                    #errorbars
                    qhi=0.99, qlo=0.01,
                    
                    #labelling
                    add_label=True, 
                    bar_labels=True,
                    baseline_loc='first_bar', #what to consider the baseline for labelling deltas
                    
 
                    
                    #plot style
                    colorMap=None,title=None,
                    #ylabel=None,
                    
                    ):
        """"
        compressing a range of values
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_total_bars')
        if plot_colr is None: plot_colr=plot_bgrp
        
        if dx_raw is None: dx_raw = self.retrieve('outs')
 
 
        """
        view(dx)
        dx.loc[idx[0, 'LMFRA', 'LMFRA_0500yr', :], idx['tvals', 0]].sum()
        dx_raw.columns.unique('dkey')
        """
        
        
        log.info('on %s'%str(dx_raw.shape))
        
        #=======================================================================
        # data prep
        #=======================================================================
        #add requested indexers
        meta_indexers = set([plot_rown, plot_coln, plot_colr, plot_bgrp]).difference(['dkey']) #all those except dkey
        
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[list(dkey_d.keys()), :]], 
                                meta_indexers = meta_indexers,
                                modelID_l=modelID_l)
        
        mdex = dx.index
        """no... want to report stats per dkey group
        #move dkeys to index for consistency
        dx.stack(level=0)"""
        
        #get label dict
        lserx =  mdex.to_frame().reset_index(drop=True).loc[:, ['modelID', 'tag']
                           ].drop_duplicates().set_index('modelID').iloc[:,0]
                            
        mid_tag_d = {k:'%s (%s)'%(v, k) for k,v in lserx.items()}
        
        if modelID_l is None:
            modelID_l = mdex.unique('modelID').tolist()
        else:
            miss_l = set(modelID_l).difference( mdex.unique('modelID'))
            assert len(miss_l)==0, 'requested %i modelIDs not in teh data \n    %s'%(len(miss_l), miss_l)
            
        #=======================================================================
        # configure dimensions
        #=======================================================================
        if plot_rown=='dkey':
            row_keys = list(dkey_d.keys())
            axis=1
 
        else:
            dkey = list(dkey_d.keys())[0]
            axis=0
            row_keys = mdex.unique(plot_rown).tolist()
            
            if title is None:
                title = '%s %s'%(dkey, dkey_d[dkey])
        
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
        """
        view(dx)
        plt.show()
        """
        col_keys = mdex.unique(plot_coln).tolist()
        
        
        
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, sharex='all',  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        
        if not title is None:
            fig.suptitle(title)
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        if plot_colr == 'dkey_range':
            ckeys = ['hi', 'low', 'mean']
        else:
            ckeys = mdex.unique(plot_colr) 
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #===================================================================
        # loop and plot
        #===================================================================

        
        for col_key, gdx1 in dx.groupby(level=[plot_coln]):
            keys_d = {plot_coln:col_key}
 
            
            for row_key, gdx2 in gdx1.groupby(level=[plot_rown], axis=axis):
                keys_d[plot_rown] = row_key
                ax = ax_d[row_key][col_key]
                
                if plot_rown=='dkey':
                    dkey = row_key
 
                #===============================================================
                # data prep
                #===============================================================
 
                
                f = getattr(gdx2.groupby(level=[plot_bgrp]), dkey_d[dkey])
                
                try:
                    gdx3 = f()#.loc[modelID_l, :] #collapse assets, sort
                except Exception as e:
                    raise Error('failed on %s w/ \n    %s'%(keys_d, e))
                
                gb = gdx3.groupby(level=0, axis=1)   #collapse iters (gb object)
                
                #===============================================================
                #plot bars------
                #===============================================================
                #===============================================================
                # data setup
                #===============================================================
 
                barHeight_ser = gb.mean() #collapse iters(
                ylocs = barHeight_ser.T.values[0]
                
                #===============================================================
                # #formatters.
                #===============================================================
 
                # labels conversion to tag
                if plot_bgrp=='modelID':
                    tick_label = [mid_tag_d[mid] for mid in barHeight_ser.index] #label by tag
                else:
                    tick_label = ['%s=%s'%(plot_bgrp, i) for i in barHeight_ser.index]
                #tick_label = ['m%i' % i for i in range(0, len(barHeight_ser))]
  
                # widths
                bar_cnt = len(barHeight_ser)
                width = 0.9 / float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                xlocs = np.linspace(0, 1, num=len(barHeight_ser))# + width * i
                bars = ax.bar(
                    xlocs,  # xlocation of bars
                    ylocs,  # heights
                    width=width,
                    align='center',
                    color=color_d.values(),
                    #label='%s=%s' % (plot_colr, ckey),
                    #alpha=0.5,
                    tick_label=tick_label,
                    )
                
                #===============================================================
                # add error bars--------
                #===============================================================
                if len(gdx2.columns.get_level_values(1))>1:
                    
                    #get error values
                    err_df = pd.concat({'hi':gb.quantile(q=qhi),'low':gb.quantile(q=qlo)}, axis=1).droplevel(axis=1, level=1)
                    
                    #convert to deltas
                    assert np.array_equal(err_df.index, barHeight_ser.index)
                    errH_df = err_df.subtract(barHeight_ser.values, axis=0).abs().T.loc[['low', 'hi'], :]
                    
                    #add the error bars
                    ax.errorbar(xlocs, ylocs,
                                errH_df.values,  
                                capsize=5, color='black',
                                fmt='none', #no data lines
                                )
                    """
                    plt.show()
                    """
                    
                #===============================================================
                # add labels--------
                #===============================================================
                if add_label:
 
                    meta_d = keys_d
                    
                    meta_d['modelID'] = str(gdx2.index.unique('modelID').tolist())
 
                    ax.text(0.9, 0.1, get_dict_str(meta_d), transform=ax.transAxes, va='bottom', ha='right',fontsize=8, color='black')
                
                if bar_labels:
                    log.debug(keys_d)
 
                    if dkey_d[dkey] == 'var':continue
                    #===========================================================
                    # #calc errors
                    #===========================================================
                    d = {'pred':barHeight_ser.T.values[0]}
                    
                    # get trues
                    if baseline_loc == 'first_bar':
                        d['true'] = np.full(len(barHeight_ser),d['pred'][0])
                    elif baseline_loc == 'first_axis':
                        if col_key == col_keys[0] and row_key == row_keys[0]:
                            base_ar = d['pred'].copy()
 
                        
                        d['true'] = base_ar
                    else:
                        raise Error('bad key')
                        
                    
                    d['delta'] = (d['pred'] - d['true']).round(3)
                    
                    # collect
                    tl_df = pd.DataFrame.from_dict(d)
                    
                    tl_df['relErr'] = (tl_df['delta'] / tl_df['true'])
                
                    tl_df['xloc'] = xlocs
                    #===========================================================
                    # add as labels
                    #===========================================================
                    for event, row in tl_df.iterrows():
                        ax.text(row['xloc'], row['pred'] * 1.01, #shifted locations
                                '%+.1f %%' % (row['relErr'] * 100),
                                ha='center', va='bottom', rotation='vertical',
                                fontsize=10, color='red')
                        
                    log.debug('added error labels \n%s' % tl_df)
                    
                    #expand the limits
                    ylims = ax.get_ylim()
                    
                    ax.set_ylim(tuple([1.1*x for x in ylims]))
                    
                #===============================================================
                # #wrap format subplot
                #===============================================================
                """
                fig.show()
                """
 
                ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
                # first row
                #===============================================================
                # if row_key == mdex.unique(plot_rown)[0]:
                #     pass
                #===============================================================
         
                        
                # first col
                if col_key == mdex.unique(plot_coln)[0]:
                    ylabel = '%s (%s)'%(dkey,  dkey_d[dkey])
                    ax.set_ylabel(ylabel)
                    
        
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        
        return self.output_fig(fig, fname='total_bars_%s' % (self.longname))
    
    


    
    def xxxplot_dkey_mat(self, #flexible plotting of model results in a matrix
                  
                    #data
                    dkey='tvals', #column group w/ values to plot
                    dx_raw=None,
                    modelID_l = None, #optinal sorting list
                    
                    #plot config
                    plot_type='hist',
                    plot_rown='aggLevel',
                    plot_coln='dscale_meth',
                    plot_colr='dkey_range',
                    #plot_bgrp='modelID',
                    
                    #data control
                    xlims = None,
                    qhi=0.99, qlo=0.01,
                    drop_zeros=True,
                    
                    #labelling
                    add_label=True,
 
                    
                    #plot style
                    colorMap=None,
                    #ylabel=None,
                    sharey='all', sharex='col',
                    
                    #histwargs
                    bins=20, rwidth=0.9, 
                    mean_line=True, #plot a vertical line on the mean
                    fmt='svg',
                    ):
        """"
        generally 1 modelId per panel
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_dkey_mat')
        
        
        idn = self.idn
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
        
        
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================

        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = set([plot_rown, plot_coln]),
                                modelID_l=modelID_l)
        
        log.info('on %s'%str(dx_raw.shape))
        mdex = dx.index
        
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, sharex=sharex,  # everything should b euniform
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        fig.suptitle('\'%s\' values'%dkey)
        
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
        if plot_colr == 'dkey_range':
            ckeys = ['hi', 'low', 'mean']
        else:
            ckeys = mdex.unique(plot_colr) 
        
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        for gkeys, gdx0 in dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)
            
            #xlims = (gdx0.min().min(), gdx0.max().max())
            #===================================================================
            # prep data
            #===================================================================
                    
            # #drop empty iters
            bxcol = gdx0.isna().all(axis=0)
            
            if bxcol.any():
     
                log.warning('got %i/%i empty iters....dropping'%(bxcol.sum(), len(bxcol)))
                gdx0 = gdx0.loc[:,~bxcol]
            else:
                gdx0 = gdx0
            
            data_d, bx = self.prep_ranges(qhi, qlo, drop_zeros, gdx0)
            
            """
            view(gdx0)
            gdx0.index.droplevel('gid').to_frame().drop_duplicates().reset_index(drop=True)
            """
                
            #===================================================================
            # #get color
            #===================================================================
            if plot_colr == 'dkey_range':
                color = [color_d[k] for k in data_d.keys()]
            else:
                #all the same color
                color = [color_d[keys_d[plot_colr]] for k in data_d.keys()]
 
            #===================================================================
            # histogram of means
            #===================================================================
            if plot_type == 'hist':
 
                ar, _, patches = ax.hist(
                    data_d.values(),
                        range=xlims,
                        bins=bins,
                        density=False,  color = color, rwidth=rwidth,
                        label=list(data_d.keys()))
                
                bin_max = ar.max()
                #vertical mean line
                if mean_line:
                    ax.axvline(gdx0.mean().mean(), color='black', linestyle='dashed')
            #===================================================================
            # box plots
            #===================================================================
            elif plot_type=='box':
                #===============================================================
                # zero line
                #===============================================================
                #ax.axhline(0, color='black')
 
                #===============================================================
                # #add bars
                #===============================================================
                boxres_d = ax.boxplot(data_d.values(), labels=data_d.keys(), meanline=True,
                           # boxprops={'color':newColor_d[rowVal]}, 
                           # whiskerprops={'color':newColor_d[rowVal]},
                           # flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
                            )
 
                #===============================================================
                # add extra text
                #===============================================================
                
                # counts on median bar
                for gval, line in dict(zip(data_d.keys(), boxres_d['medians'])).items():
                    x_ar, y_ar = line.get_data()
                    ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(data_d[gval]),
                            # transform=ax.transAxes, 
                            va='bottom', ha='center', fontsize=8)
                    
            #===================================================================
            # violin plot-----
            #===================================================================
            elif plot_type=='violin':

                #===============================================================
                # plot
                #===============================================================
                parts_d = ax.violinplot(data_d.values(),  
                                       showmeans=True,
                                       showextrema=True,  
                                       )
 
                #===============================================================
                # color
                #===============================================================
                #===============================================================
                # if len(data_d)>1:
                #     """nasty workaround for labelling"""                    
                #===============================================================
                labels = list(data_d.keys())
                #ckey_d = {i:color_key for i,color_key in enumerate(labels)}
 
 
                
                #style fills
                for i, pc in enumerate(parts_d['bodies']):
                    pc.set_facecolor(color)
                    pc.set_edgecolor(color)
                    pc.set_alpha(0.5)
                    
                #style lines
                for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                    parts_d[partName].set(color='black', alpha=0.5)
            else:
                raise Error(plot_type)
            
            if not xlims is None:
                ax.set_xlim(xlims)
            #===================================================================
            # labels
            #===================================================================
            if add_label:
                # get float labels
                meta_d = {'modelIDs':str(gdx0.index.unique(idn).tolist()),
                           'cnt':len(gdx0), 'zero_cnt':bx.sum(), 'drop_zeros':drop_zeros,
                           'iters':len(gdx0.columns),
                           'min':gdx0.min().min(), 'max':gdx0.max().max(), 'mean':gdx0.mean().mean()}
                
                if plot_type == 'hist':
                    meta_d['bin_max'] = bin_max
 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
            
            #===================================================================
            # wrap format
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
 
                
                # first row
                if row_key == row_keys[0]:
                    #first col
                    if col_key == col_keys[0]:
                        if plot_type=='hist':
                            ax.legend()
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type == 'hist':
                        ax.set_ylabel('count')
                    elif plot_type in ['box', 'violin']:
                        ax.set_ylabel(dkey)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type == 'hist':
                        ax.set_xlabel(dkey)
                    elif plot_type=='violin':
                        
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                    #last col
                    if col_key == col_keys[-1]:
                        pass
 
        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        
        return self.output_fig(fig, fname='%s_%s_%sx%s_%s' % (dkey, plot_type, plot_rown, plot_coln, self.longname), fmt=fmt)
    



    def xxxplot_depths(self,
                    # data control
                    plot_fign='studyArea',
                    plot_rown='grid_size',
                    plot_coln='event',
                    plot_zeros=False,
                    serx=None,
                    
                    # style control
                    xlims=(0, 2),
                    ylims=(0, 2.5),
                    calc_str='points',
                    
                    out_dir=None,
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_depths')
        if out_dir is None: out_dir = self.out_dir
        if serx is None: serx = self.retrieve('rsamps')
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        plt.show()
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i' % len(serx))
        
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(),  # row keys
                                    gsx1.index.unique(plot_coln).tolist(),  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all',  # everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
            for (row_key, col_key), gsx2r in gsx1r.groupby(level=[plot_rown, plot_coln]):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_coln, plot_fign])
                
                if plot_zeros:
                    ar = gsx2.values
                else:
                    bx = gsx2 > 0.0
                    ar = gsx2[bx].values
                
                if not len(ar) > 0:
                    log.warning('no values for %s.%s.%s' % (sName, row_key, col_key))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color='blue', alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                #===============================================================
                # #label
                #===============================================================
                # get float labels
                meta_d = {'calc_method':calc_str, plot_rown:row_key, 'wet':len(ar), 'dry':(gsx2 <= 0.0).sum(),
                           'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
 
                ax.text(0.5, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='blue')

                #===============================================================
                # styling
                #===============================================================                    
                # first columns
                if col_key == mdex.unique(plot_coln)[0]:
                    """not setting for some reason"""
                    ax.set_ylabel('density')
 
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    # ax.set_title('event \'%s\''%(rlayName))
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('depth (m)')
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)' % (sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName] = self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s' % (sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures' % len(res_d))
        
        return res_d

    def xxxplot_tvals(self,
                    plot_fign='studyArea',
                    plot_rown='grid_size',
                    # plot_coln = 'event',
                    
                    plot_zeros=True,
                    xlims=(0, 200),
                    ylims=None,
                    
                    out_dir=None,
                    color='orange',
                    
                    ):
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_tvals')
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
 
        serx = self.retrieve('tvals')
        
        assert serx.notna().all().all(), 'drys should be zeros'

        """
        self.retrieve('tvals')
        view(serx)
        """
        #=======================================================================
        # loop on studyAreas
        #=======================================================================
        
        log.info('on %i' % len(serx))
        
        col_key = ''
        res_d = dict()
        for i, (sName, gsx1r) in enumerate(serx.groupby(level=plot_fign)):
            plt.close('all')
            gsx1 = gsx1r.droplevel(plot_fign)
            mdex = gsx1.index
            
            fig, ax_d = self.get_matrix_fig(
                                    gsx1.index.unique(plot_rown).tolist(),  # row keys
                                    [col_key],  # col keys
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey='all', sharex='all',  # everything should b euniform
                                    fig_id=i,
                                    set_ax_title=True,
                                    )
            
            for row_key, gsx2r in gsx1r.groupby(level=plot_rown):
                #===============================================================
                # #prep data
                #===============================================================
                gsx2 = gsx2r.droplevel([plot_rown, plot_fign])
                bx = gsx2 > 0.0
                if plot_zeros:
                    ar = gsx2.values
                else:
                    
                    ar = gsx2[bx].values
                
                if not len(ar) > 0:
                    log.warning('no values for %s.%s.%s' % (sName, row_key,))
                    continue
                #===============================================================
                # #plot
                #===============================================================
                ax = ax_d[row_key][col_key]
                ax.hist(ar, color=color, alpha=0.3, label=row_key, density=True, bins=30, range=xlims)
                
                # label
                meta_d = {
                    plot_rown:row_key,
                    'cnt':len(ar), 'zeros_cnt':np.invert(bx).sum(), 'min':ar.min(), 'max':ar.max(), 'mean':ar.mean()}
                
                txt = '\n'.join(['%s=%.2f' % (k, v) for k, v in meta_d.items()])
                ax.text(0.5, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='blue')

                #===============================================================
                # styling
                #===============================================================                    
                # first columns
                #===============================================================
                # if col_key == mdex.unique(plot_coln)[0]:
                #     """not setting for some reason"""
                #===============================================================
                ax.set_ylabel('density')
 
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                    ax.set_xlim(xlims)
                    ax.set_ylim(ylims)
                    pass
                    # ax.set_title('event \'%s\''%(rlayName))
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel('total value (scale)')
                    
            fig.suptitle('depths for studyArea \'%s\' (%s)' % (sName, self.tag))
            #===================================================================
            # wrap figure
            #===================================================================
            res_d[sName] = self.output_fig(fig, out_dir=os.path.join(out_dir, sName), fname='depths_%s_%s' % (sName, self.longname))

        #=======================================================================
        # warp
        #=======================================================================
        log.info('finished writing %i figures' % len(res_d))
        
        return res_d      


    def xxxplot_terrs_box(self,  # boxplot of total errors
                    
                    # data control
                    dkey='errs',
                    ycoln=('tl', 'delta'),  # values to plot
                    plot_fign='studyArea',
                    plot_rown='event',
                    plot_coln='vid',
                    plot_colr='grid_size',
                    # plot_bgrp = 'event',
                    
                    # plot style
                    ylabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    out_dir=None,
                   ):
        """
        matrix figure
            figure: studyAreas
                rows: grid_size
                columns: events
                values: total loss sum
                colors: grid_size
        
        """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_terr_box')
        if colorMap is None: colorMap = self.colorMap
        if ylabel is None: ylabel = dkey
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        log.info('on \'%s\' w/ %i' % (dkey, len(dx_raw)))
        
        #=======================================================================
        # setup data
        #=======================================================================

        # make slice
        dxser = dx_raw.loc[:, ycoln]
 
        #=======================================================================
        # loop on figures
        #=======================================================================
        for i, (fig_key, gser0r) in enumerate(dxser.groupby(level=plot_fign)):
            
            mdex = gser0r.index
            plt.close('all')
            
            fig, ax_d = self.get_matrix_fig(
                                        mdex.unique(plot_rown).tolist(),  # row keys
                                        mdex.unique(plot_coln).tolist(),  # col keys
                                        figsize_scaler=4,
                                        constrained_layout=True,
                                        sharey='all',
                                        sharex='none',  # events should all be uniform
                                        fig_id=i,
                                        set_ax_title=True,
                                        )
            
            s = '-'.join(ycoln)
            fig.suptitle('%s for %s:%s (%s)' % (s, plot_fign, fig_key, self.tag))
 
            """
            fig.show()
            """
            
            #===================================================================
            # loop and plot
            #===================================================================
            for (row_key, col_key), gser1r in gser0r.droplevel(plot_fign).groupby(level=[plot_rown, plot_coln]):
                
                # data setup
                gser1 = gser1r.droplevel([plot_rown, plot_coln])
     
                # subplot setup 
                ax = ax_d[row_key][col_key]
                
                # group values
                gd = {k:g.values for k, g in gser1.groupby(level=plot_colr)}
                
                #===============================================================
                # zero line
                #===============================================================
                ax.axhline(0, color='red')
 
                #===============================================================
                # #add bars
                #===============================================================
                boxres_d = ax.boxplot(gd.values(), labels=gd.keys(), meanline=True,
                           # boxprops={'color':newColor_d[rowVal]}, 
                           # whiskerprops={'color':newColor_d[rowVal]},
                           # flierprops={'markeredgecolor':newColor_d[rowVal], 'markersize':3,'alpha':0.5},
                            )
                
                #===============================================================
                # add extra text
                #===============================================================
                
                # counts on median bar
                for gval, line in dict(zip(gd.keys(), boxres_d['medians'])).items():
                    x_ar, y_ar = line.get_data()
                    ax.text(x_ar.mean(), y_ar.mean(), 'n%i' % len(gd[gval]),
                            # transform=ax.transAxes, 
                            va='bottom', ha='center', fontsize=8)
                    
                    #===========================================================
                    # if add_text:
                    #     ax.text(x_ar.mean(), ylims[0]+1, 'mean=%.2f'%gd[gval].mean(), 
                    #         #transform=ax.transAxes, 
                    #         va='bottom',ha='center',fontsize=8, rotation=90)
                    #===========================================================
                    
                #===============================================================
                # #wrap format subplot
                #===============================================================
                ax.grid(alpha=0.8)
                # first row
                if row_key == mdex.unique(plot_rown)[0]:
                     
                    # last col
                    if col_key == mdex.unique(plot_coln)[-1]:
                        # ax.legend()
                        pass
                         
                # first col
                if col_key == mdex.unique(plot_coln)[0]:
                    ax.set_ylabel(ylabel)
                    
                # last row
                if row_key == mdex.unique(plot_rown)[-1]:
                    ax.set_xlabel(plot_colr)
            #===================================================================
            # wrap fig
            #===================================================================
            log.debug('finsihed %s' % fig_key)
            self.output_fig(fig, fname='box_%s_%s' % (s, self.longname),
                            out_dir=os.path.join(out_dir, fig_key))

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
 
    def xxxplot_errs_scatter(self,  # scatter plot of error-like data
                    # data control   
                    dkey='errs',
                    
                    # lossType='rl',
                    ycoln=('rl', 'delta'),
                    xcoln=('depth', 'grid'),
                       
                    # figure config
                    folder_varn='studyArea',
                    plot_fign='event',
                    plot_rown='grid_size',
                    plot_coln='vid',
                    plot_colr=None,
                    # plot_bgrp = 'event',
                    
                    plot_vf=False,  # plot the vf
                    plot_zeros=False,
                    
                    # axconfig
                    ylims=None,
                    xlims=None,
                    
                    # plot style
                    ylabel=None,
                    xlabel=None,
                    colorMap=None,
                    add_text=True,
                    
                    # outputs
                    fmt='png', transparent=False,
                    out_dir=None,
                   ):
        
        # raise Error('lets fit a regression to these results')
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_errs_scatter')
        if colorMap is None: colorMap = self.colorMap
        if ylabel is None: ylabel = '.'.join(ycoln)
        if xlabel is None: xlabel = '.'.join(xcoln)
        
        #=======================================================================
        # if plot_vf:
        #     assert lossType=='rl'
        #=======================================================================
            
        if plot_colr is None: plot_colr = plot_rown
        if out_dir is None: out_dir = self.out_dir
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
 
        if plot_vf:
            vf_d = self.retrieve('vf_d')
        log.info('on \'%s\' for %s vs %s w/ %i' % (dkey, xcoln, ycoln, len(dx_raw)))
        
        #=======================================================================
        # prep data
        #=======================================================================
        # get slice specified by user
        dx1 = pd.concat([dx_raw.loc[:, ycoln], dx_raw.loc[:, xcoln]], axis=1)
        dx1.columns.set_names(dx_raw.columns.names, inplace=True)
 
        #=======================================================================
        # plotting setup
        #=======================================================================
        # get colors
        cvals = dx_raw.index.unique(plot_colr)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        """
        plt.show()
        """
        #=======================================================================
        # loop an study area/folders
        #=======================================================================
        
        for folder_key, gdx0 in dx1.groupby(level=folder_varn, axis=0):
 
            #=======================================================================
            # loop on figures
            #=======================================================================
            od = os.path.join(out_dir, folder_key, xlabel)
            plt.close('all')
        
            for i, (fig_key, gdx1) in enumerate(gdx0.groupby(level=plot_fign, axis=0)):
                keys_d = dict(zip([folder_varn, plot_fign], (folder_key, fig_key)))
                mdex = gdx1.index
                
                fig, ax_d = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(),  # row keys
                                            mdex.unique(plot_coln).tolist(),  # col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='all',
                                            sharex='all',  # events should all be uniform
                                            fig_id=i,
                                            set_ax_title=False,
                                            )
                
                s = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                fig.suptitle('%s vs %s for %s' % (xcoln, ycoln, s))
            
                """
                fig.show()
                """
            
                #===================================================================
                # loop on axis row/column (and colors)----------
                #===================================================================
                for (row_key, col_key, ckey), gdx2 in gdx1.groupby(level=[plot_rown, plot_coln, plot_colr]):
                    keys_d.update(
                        dict(zip([plot_rown, plot_coln, plot_colr], (row_key, col_key, ckey)))
                        )
                    # skip trues
                    #===========================================================
                    # if ckey == 0:
                    #     continue 
                    #===========================================================
                    # subplot setup 
                    ax = ax_d[row_key][col_key]
 
                    #===============================================================
                    # prep data
                    #===============================================================
 
                    dry_bx = gdx2[xcoln] <= 0.0
                    
                    if not plot_zeros:
                        xar, yar = gdx2.loc[~dry_bx, xcoln].values, gdx2.loc[~dry_bx, ycoln].values
                    else:
                        xar, yar = gdx2[xcoln].values, gdx2[ycoln].values

                    #===============================================================
                    # zero line
                    #===============================================================
                    ax.axhline(0, color='black', alpha=0.8, linewidth=0.5)
                    
                    #===============================================================
                    # plot function
                    #===============================================================
                    if plot_vf:
                        vf_d[col_key].plot(ax=ax, logger=log, set_title=False,
                                           lineKwargs=dict(
                            color='black', linestyle='dashed', linewidth=1.0, alpha=0.9)) 
                    
                    #===============================================================
                    # #add scatter plot
                    #===============================================================
                    ax.plot(xar, yar,
                               color=newColor_d[ckey], markersize=4, marker='x', alpha=0.8,
                               linestyle='none',
                                   label='%s=%s' % (plot_colr, ckey))
 
                    #===========================================================
                    # add text
                    #===========================================================

                    if add_text:
                        meta_d = {'ycnt':len(yar),
                                  'dry_cnt':dry_bx.sum(),
                                  'wet_cnt':np.invert(dry_bx).sum(),
                                  'y0_cnt':(yar == 0.0).sum(),
                                  'ymean':yar.mean(), 'ymin':yar.min(), 'ymax':yar.max(),
                                  'xmax':xar.max(),
                              # 'plot_zeros':plot_zeros,
                              }
                        
                        if ycoln[1] == 'delta':
                            meta_d['rmse'] = ((yar ** 2).mean()) ** 0.5
                                            
                        txt = '\n'.join(['%s=%.2f' % (k, v) for k, v in meta_d.items()])
                        ax.text(0.1, 0.9, txt, transform=ax.transAxes, va='top', fontsize=8, color='black')
     
                    #===============================================================
                    # #wrap format subplot
                    #===============================================================
                    ax.set_title('%s=%s and %s=%s' % (
                         plot_rown, row_key, plot_coln, col_key))
                    
                    # first row
                    if row_key == mdex.unique(plot_rown)[0]:
                        
                        # last col
                        if col_key == mdex.unique(plot_coln)[-1]:
                            pass
                             
                    # first col
                    if col_key == mdex.unique(plot_coln)[0]:
                        ax.set_ylabel(ylabel)
                        
                    # last row
                    if row_key == mdex.unique(plot_rown)[-1]:
                        ax.set_xlabel(xlabel)
                        
                    # loast col
                    if col_key == mdex.unique(plot_coln)[-1]:
                        pass
                        # ax.legend()
                        
                #===================================================================
                # post format
                #===================================================================
                for row_key, ax0_d in ax_d.items():
                    for col_key, ax in ax0_d.items():
                        ax.grid()
                        
                        if not ylims is None:
                            ax.set_ylim(ylims)
                        
                        if not xlims is None:
                            ax.set_xlim(xlims)
                #===================================================================
                # wrap fig
                #===================================================================
                log.debug('finsihed %s' % fig_key)
                s = '_'.join(['%s' % (keys_d[k]) for k in [ folder_varn, plot_fign]])
                
                s2 = ''.join(ycoln) + '-' + ''.join(xcoln)
                
                self.output_fig(fig, out_dir=od,
                                fname='scatter_%s_%s_%s' % (s2, s, self.longname.replace('_', '')),
                                fmt=fmt, transparent=transparent, logger=log)
            """
            fig.show()
            """

        #=======================================================================
        # wrap
        #=======================================================================
        log.info('finsihed')
        return
    
    def xxxplot_accuracy_mat(self,  # matrix plot of accuracy
                    # data control   
                    dkey='errs',
                    lossType='tl',
                    
                    folder_varns=['studyArea', 'event'],
                    plot_fign='vid',  # one raster:vid per plot
                    plot_rown='grid_size',
                    plot_zeros=True,

                    # output control
                    out_dir=None,
                    fmt='png',
                    
                    # plot style
                    binWidth=None,
                    colorMap=None,
                    lims_d={'raw':{'x':None, 'y':None}}  # control limits by column
                    # add_text=True,
                   ):
        
        """
        row1: trues
        rowx: grid sizes
        
        col1: hist of raw 'grid' values (for this lossType)
        col2: hist of delta values
        col3: scatter of 'grid' vs. 'true' values 
            """
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_accuracy_mat.%s' % lossType)
        if colorMap is None: colorMap = self.colorMap
 
        if out_dir is None: out_dir = self.out_dir
        col_keys = ['raw', 'delta', 'correlation']
        #=======================================================================
        # #retrieve child data
        #=======================================================================
        dx_raw = self.retrieve(dkey)
        
        # slice by user
        dxind1 = dx_raw.loc[:, idx[lossType, ['grid', 'true', 'delta']]].droplevel(0, axis=1)
        """
        dx_raw.columns
        view(dx_raw)
        """
        # get colors
        cvals = dx_raw.index.unique(plot_rown)
        cmap = plt.cm.get_cmap(name=colorMap) 
        newColor_d = {k:matplotlib.colors.rgb2hex(cmap(ni)) for k, ni in dict(zip(cvals, np.linspace(0, 1, len(cvals)))).items()}
        
        #=======================================================================
        # helpers
        #=======================================================================
        lim_max_d = {'raw':{'x':(0, 0), 'y':(0, 0)}, 'delta':{'x':(0, 0), 'y':(0, 0)}}

        def upd_lims(key, ax):
            # x axis
            lefti, righti = ax.get_xlim()
            leftj, rightj = lim_max_d[key]['x'] 
            
            lim_max_d[key]['x'] = (min(lefti, leftj), max(righti, rightj))
            
            # yaxis
            lefti, righti = ax.get_ylim()
            leftj, rightj = lim_max_d[key]['y'] 
            
            lim_max_d[key]['y'] = (min(lefti, leftj), max(righti, rightj))
        
        def set_lims(key, ax):
            if key in lims_d:
                if 'x' in lims_d[key]:
                    ax.set_xlim(lims_d[key]['x'])
                if 'y' in lims_d[key]:
                    ax.set_ylim(lims_d[key]['y'])
            
            upd_lims(key, ax)
        #=======================================================================
        # loop and plot----------
        #=======================================================================
        
        log.info('for \'%s\' w/ %i' % (lossType, len(dxind1)))
        for fkeys, gdxind1 in dxind1.groupby(level=folder_varns):
            keys_d = dict(zip(folder_varns, fkeys))
            
            for fig_key, gdxind2 in gdxind1.groupby(level=plot_fign):
                keys_d[plot_fign] = fig_key
                
                # setup folder
                od = os.path.join(out_dir, fkeys[0], fkeys[1], str(fig_key))
                """
                view(gdxind2)
                gdxind2.index
                fig.show()
                """
                log.info('on %s' % keys_d)
                #===============================================================
                # figure setup
                #===============================================================
                mdex = gdxind2.index
                plt.close('all')
                fig, ax_lib = self.get_matrix_fig(
                                            mdex.unique(plot_rown).tolist(),  # row keys
                                            col_keys,  # col keys
                                            figsize_scaler=4,
                                            constrained_layout=True,
                                            sharey='none',
                                            sharex='none',  # events should all be uniform
                                            fig_id=0,
                                            set_ax_title=True,
                                            )
                
                s = ' '.join(['%s-%s' % (k, v) for k, v in keys_d.items()])
                fig.suptitle('%s Accruacy for %s' % (lossType.upper(), s))
                
                #===============================================================
                # raws
                #===============================================================
                varn = 'grid'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    ax = ax_lib[ax_key]['raw']
                    self.ax_hist(ax,
                        gser,
                        label=varn,
                        stat_keys=['min', 'max', 'median', 'mean', 'std'],
                        style_d=dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros=plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    # set limits
                    set_lims('raw', ax)
                    
                #===============================================================
                # deltas
                #===============================================================
                varn = 'delta'
                for ax_key, gser in gdxind2[varn].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    
                    self.ax_hist(ax_lib[ax_key][varn],
                        gser,
                        label=varn,
                        stat_keys=['min', 'max', 'median', 'mean', 'std'],
                        style_d=dict(color=newColor_d[ax_key]),
                        binWidth=binWidth,
                        plot_zeros=plot_zeros,
                        logger=log.getChild(s1),
                        )
                    
                    upd_lims(varn, ax)
                #===============================================================
                # scatter
                #===============================================================
                for ax_key, gdxind3 in gdxind2.loc[:, ['grid', 'true']].groupby(level=plot_rown):
                    if ax_key == 0:continue
                    keys_d[plot_rown] = ax_key
                    s1 = ' '.join(['%s:%s' % (k, v) for k, v in keys_d.items()])
                    
                    self.ax_corr_scat(ax_lib[ax_key]['correlation'],
                          
                          gdxind3['grid'].values,  # x (first row is plotting gridded also)
                          gdxind3['true'].values,  # y 
                          style_d=dict(color=newColor_d[ax_key]),
                          label='grid vs true',
                          
                          )
                
                #=======================================================================
                # post formatting
                #=======================================================================
                """
                fig.show()
                """
                for row_key, d0 in ax_lib.items():
                    for col_key, ax in d0.items():
                        
                        # first row
                        if row_key == mdex.unique(plot_rown)[0]:
                            pass
                            
                            # last col
                            if col_key == col_keys[-1]:
                                pass
                            
                        # last row
                        if row_key == mdex.unique(plot_rown)[-1]:
                            # first col
                            if col_key == col_keys[0]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'grid'))
                            elif col_key == col_keys[1]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'delta'))
                            elif col_key == col_keys[-1]:
                                ax.set_xlabel('%s (%s)' % (lossType, 'grid'))
                                 
                        # first col
                        if col_key == col_keys[0]:
                            ax.set_ylabel('count')
                            ax.set_xlim(lim_max_d['raw']['x'])
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        # second col
                        if col_key == col_keys[1]:
                            ax.set_ylim(lim_max_d['raw']['y'])
                            
                        # loast col
                        if col_key == col_keys[-1]:
                            # set limits from first columns
                            col1_xlims = ax_lib[row_key]['raw'].get_xlim()
                            ax.set_xlim(col1_xlims)
                            ax.set_ylim(col1_xlims)
                            
                            if not row_key == 0:
                                
                                ax.set_ylabel('%s (%s)' % (lossType, 'true'))
                                # move to the right
                                ax.yaxis.set_label_position("right")
                                ax.yaxis.tick_right()
 
                #===============================================================
                # wrap fig
                #===============================================================
                s = '_'.join([str(e) for e in keys_d.values()])
                self.output_fig(fig, out_dir=od,
                                fname='accuracy_%s_%s_%s' % (lossType, s, self.longname.replace('_', '')),
                                fmt=fmt, logger=log, transparent=False)
                
            #===================================================================
            # wrap folder
            #===================================================================
                
        #===================================================================
        # wrap
        #===================================================================
        log.info('finished')
        
        return
    def xxxplot_compare_mat(self, #flexible plotting of model results vs. true in a matrix
                  
                    #data
                    dkey='tvals',#column group w/ values to plot
                    aggMethod='mean', #method to use for aggregating the true values (down to the gridded)
                    true_dx_raw=None, #base values (indexed to raws per model)
                    baseID=0,
                    dx_raw=None, #combined model results
                    modelID_l = None, #optinal sorting list
                    slice_d = {}, #special slicing
                    
                    #plot config
                    plot_type='scatter', 
                    plot_rown='aggLevel',
                    plot_coln='resolution',
                    plot_colr=None,
                    
                    #plot config [bars]
                    plot_bgrp=None, #grouping (for plotType==bars)
                    err_type='absolute', #what type of errors to calculate (for plot_type='bars')
                        #absolute: modelled - true
                        #relative: absolute/true
                    confusion_rel=True, #whether to use relative confusion metrics
 
                    
                    #data control
                    xlims = None,
                    qhi=0.99, qlo=0.01,
                    #drop_zeros=True, #must always be false for the matching to work
                    
                    #labelling
                    #baseID=None, 
                    add_label=True,
                    title=None,
 
                    
                    #plot style
                    colorMap=None,
                    sharey=None,sharex=None,
                    
                    #outputs
                    write_meta=False, #write all the meta info to a csv
                    **kwargs):
        """"
        generally 1 modelId per panel
        TODO: 
        pass meta as a lambda
            gives more customization outside of this function (should simplify)
        
        
        """
        
        #=======================================================================
        # defaults
        #=======================================================================
        log = self.logger.getChild('plot_compare_mat')
        
        if plot_colr is None: 
            plot_colr=plot_bgrp
        
        if plot_colr is None: 
            plot_colr=plot_rown
            
        if plot_bgrp is None:

            plot_bgrp = plot_colr
            
        
            
        idn = self.idn
        #if baseID is None: baseID=self.baseID
 
        if dx_raw is None:
            dx_raw = self.retrieve('outs')
            
        if true_dx_raw is None:
            true_d = self.retrieve('trues')
            true_dx_raw = true_d[baseID]
            
        if sharey is None:
            if plot_type=='scatter':
                sharey='none'
            else:
                sharey='all'
                
        if sharex is None:
            if plot_type=='scatter':
                sharex='none'
            else:
                sharex='all'
                
        if plot_type=='bars':
            #assert err_type in ['absolute', 'relative']
            assert isinstance(plot_bgrp, str)
        else:
            plot_bgrp = None
        
        
        log.info('on \'%s\' (%s x %s)'%(dkey, plot_rown, plot_coln))
        #=======================================================================
        # data prep
        #=======================================================================
        assert_func(lambda: self.check_mindex_match(true_dx_raw.index, dx_raw.index), msg='raw vs trues')
        
        meta_indexers = set([plot_rown, plot_coln])
        if not plot_bgrp is None:
            meta_indexers.add(plot_bgrp)
        
        #add requested indexers
        dx = self.join_meta_indexers(dx_raw = dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = meta_indexers,
                                modelID_l=modelID_l)
        
        log.info('on %s'%str(dx.shape))
        mdex = dx.index
        
        #and on the trues
        true_dx = self.join_meta_indexers(dx_raw = true_dx_raw.loc[:, idx[dkey, :]], 
                                meta_indexers = meta_indexers,
                                modelID_l=modelID_l)
        
        
        #=======================================================================
        # subsetting
        #=======================================================================
        for name, val in slice_d.items():
            assert name in dx.index.names
            bx = dx.index.get_level_values(name) == val
            assert bx.any()
            dx = dx.loc[bx, :]
            
            bx = true_dx.index.get_level_values(name) == val
            assert bx.any()
            true_dx = true_dx.loc[bx, :]
            log.info('w/ %s=%s slicing to %i/%i'%(
                name, val, bx.sum(), len(bx)))
 
        #=======================================================================
        # setup the figure
        #=======================================================================
        plt.close('all')
 
        col_keys =mdex.unique(plot_coln).tolist()
        row_keys = mdex.unique(plot_rown).tolist()
 
        fig, ax_d = self.get_matrix_fig(row_keys, col_keys,
                                    figsize_scaler=4,
                                    constrained_layout=True,
                                    sharey=sharey, 
                                    sharex=sharex,  
                                    fig_id=0,
                                    set_ax_title=True,
                                    )
        
        #=======================================================================
        # title
        #=======================================================================
        if title is None:
            if not plot_type=='bars':
                title = '\'%s\' errors'%dkey
            else:
                title = '\'%s\' %s'%(dkey, err_type)
                
        for name, val in slice_d.items(): 
            title = title + ' %s=%s'%(name, val) 
            
        fig.suptitle(title)
        #=======================================================================
        # #get colors
        #=======================================================================
        if colorMap is None: colorMap = self.colorMap_d[plot_colr]
 
        ckeys = mdex.unique(plot_colr) 
        color_d = self.get_color_d(ckeys, colorMap=colorMap)
        
        #=======================================================================
        # loop and plot
        #=======================================================================
        meta_dx=None
        true_gb = true_dx.groupby(level=[plot_coln, plot_rown])
        for gkeys, gdx0 in dx.groupby(level=[plot_coln, plot_rown]): #loop by axis data
            
            #===================================================================
            # setup
            #===================================================================
            keys_d = dict(zip([plot_coln, plot_rown], gkeys))
            ax = ax_d[gkeys[1]][gkeys[0]]
            log.info('on %s'%keys_d)

            #===================================================================
            # data prep----------
            #===================================================================
            gdx1 = gdx0
 
            #get the corresponding true values
            tgdx0 = true_gb.get_group(gkeys)
            
            #===================================================================
            # aggregate the trues
            #===================================================================
            """because trues are mapped from the base model.. here we compress down to the model index"""
            tgdx1 = getattr(tgdx0.groupby(level=[gdx1.index.names]), aggMethod)()
            tgdx2 = tgdx1.reorder_levels(gdx1.index.names).sort_index()
            
            if not np.array_equal(gdx1.index, tgdx2.index):
                
                miss_l = set(gdx1.index.unique(5)).symmetric_difference(tgdx2.index.unique(5))
 
                
                if len(miss_l)>0:
                    raise Error('%i/%i true keys dont match on %s \n    %s'%(len(miss_l), len(gdx1), keys_d, miss_l))
                else:
                    raise Error('bad indexers on %s modelIDs:\n    %s'%(keys_d, tgdx2.index.unique('modelID').tolist()))
            
            
            
            #===================================================================
            # meta
            #===================================================================
            meta_d = { 'modelIDs':str(list(gdx1.index.unique(idn))),
                            'drop_zeros':False,'iters':len(gdx1.columns),
                            }
      
            
            #===================================================================
            # scatter plot-----
            #===================================================================
            """consider hist2d?"""
            if plot_type =='scatter':
                #===================================================================
                # reduce ranges
                #===================================================================
                #model results
                data_d, zeros_bx = self.prep_ranges(qhi, qlo, False, gdx1)
                
                #trues
                true_data_d, _ = self.prep_ranges(qhi, qlo, False, tgdx2)
            
                """only using mean values for now"""
                xar, yar = data_d['mean'], true_data_d['mean'] 
                
                stat_d = self.ax_corr_scat(ax, xar, yar, 
                                           #label='%s=%s'%(plot_colr, keys_d[plot_colr]),
                                           scatter_kwargs = {
                                               'color':color_d[keys_d[plot_colr]]
                                               },
                                           logger=log, add_label=False)
                
                gdata = gdx1 #for stats
                #===============================================================
                # meta
                #===============================================================
                meta_d.update(stat_d)
                
                #error calcs
                """would be nice to move this up for the other plot_types?
                    need to sort out iters..."""
                eW = ErrorCalcs(logger=log,
                                pred_ser=pd.Series(data_d['mean']), 
                                true_ser=pd.Series(true_data_d['mean']))

                confusion_rel
                err_d =  eW.get_all()
                cm_df, cm_dx = err_d.pop('confusion') #pull out conusion
                #add confusion matrix stats
                #cm_df, cm_dx = self.get_confusion(pd.DataFrame({'pred':xar, 'true':yar}), logger=log)
                meta_d.update(err_d)
                meta_d.update(cm_dx.droplevel(['pred', 'true']).iloc[:,0].to_dict())
                
            #===================================================================
            # bar plot---------
            #===================================================================
            elif plot_type=='bars':
                """TODO: consolidate w/ plot_total_bars
                integrate with write_suite_smry errors"""
                #===============================================================
                # data setup
                #===============================================================

                if err_type=='bias':
                    predTotal_ser= gdx1.groupby(level=plot_bgrp).sum()
                    trueTotal_ser= tgdx2.groupby(level=plot_bgrp).sum()
                    barHeight_ser = (predTotal_ser/trueTotal_ser).iloc[:,0]
 
                    
                elif err_type=='absolute': #straight differences
                    gdx2 = gdx1 - tgdx2 #modelled - trues

                    gb = gdx2.groupby(level=plot_bgrp)
                    
                    barHeight_ser = gb.sum().iloc[:,0] #collapse iters(
 
                    
                else:
                    #calc error on each group
                    """probably a much nicer way to do this with apply"""
                    barHeight_d = dict()
                    true_gb2 = tgdx2.groupby(level=plot_bgrp)
                    for groupKey, gPred_ser in gdx1.groupby(level=plot_bgrp):
                        gTrue_ser = true_gb2.get_group(groupKey)
                        
                        barHeight_d[groupKey] = ErrorCalcs(logger=log,
                                  pred_ser=gPred_ser.mean(axis=1), #collapse iters?
                                  true_ser=gTrue_ser.mean(axis=1),
                                  ).retrieve(err_type)
                                  
                    s = gdx1.groupby(level=plot_bgrp).sum()
                    barHeight_ser = pd.Series(barHeight_d, index=s.index)
 

 
                assert isinstance(barHeight_ser, pd.Series)
                """always want totals for the bars"""
                
                ylocs = barHeight_ser.values
                gdata = gdx1 - tgdx2 #modelled - trues (data for stats)
                
                #===============================================================
                # #formatters.
                #===============================================================
 
                # labels conversion to tag
                if plot_bgrp=='modelID':
                    raise Error('not implementd')
                    #tick_label = [mid_tag_d[mid] for mid in barHeight_ser.index] #label by tag
                else:
                    tick_label = ['%s=%s'%(plot_bgrp, i) for i in barHeight_ser.index]
                #tick_label = ['m%i' % i for i in range(0, len(barHeight_ser))]
  
                # widths
                bar_cnt = len(barHeight_ser)
                width = 0.9 / float(bar_cnt)
                
                #===============================================================
                # #add bars
                #===============================================================
                xlocs = np.linspace(0, 1, num=len(barHeight_ser))# + width * i
                bars = ax.bar(
                    xlocs,  # xlocation of bars
                    ylocs,  # heights
                    width=width,
                    align='center',
                    color=color_d.values(),
                    #label='%s=%s' % (plot_colr, ckey),
                    #alpha=0.5,
                    tick_label=tick_label,
                    )
                
                ax.axhline(0, color='black') #draw in the axis
                
                #===============================================================
                # add error bars 
                #===============================================================
                if len(gdx1.columns.get_level_values(1))>1:
                    if not err_type=='absolute':
                        raise Error('not implemented')
                    """untesetd"""
                    #get error values
                    err_df = pd.concat({'hi':gb.quantile(q=qhi),'low':gb.quantile(q=qlo)}, axis=1).droplevel(axis=1, level=1)
                    
                    #convert to deltas
                    assert np.array_equal(err_df.index, barHeight_ser.index)
                    errH_df = err_df.subtract(barHeight_ser.values, axis=0).abs().T.loc[['low', 'hi'], :]
                    
                    #add the error bars
                    ax.errorbar(xlocs, ylocs,
                                errH_df.values,  
                                capsize=5, color='black',
                                fmt='none', #no data lines
                                )
                
                #===============================================================
                # add bar labels
                #===============================================================
                d1 = {k:pd.Series(v, dtype=float) for k,v in {'yloc':ylocs, 'xloc':xlocs}.items()}

                for event, row in pd.concat(d1, axis=1).iterrows():
 
                    txt = '%+.2f' %(row['yloc'])

                    #txt = '%+.1f %%' % (row['yloc'] * 100)
                        
                    ax.text(row['xloc'], row['yloc'] * 1.01, #shifted locations
                                txt,ha='center', va='bottom', rotation='vertical',fontsize=10, color='red')
                    
            #===================================================================
            # violin plot-----
            #===================================================================
            elif plot_type=='violin':
                #===============================================================
                # data setup
                #===============================================================
                gdx2 = gdx1 - tgdx2 #modelled - trues
                
                gb = gdx2.groupby(level=plot_bgrp)
                
                data_d = {k:v.values.T[0] for k,v in gb}
                gdata = gdx2
                #===============================================================
                # plot
                #===============================================================
                parts_d = ax.violinplot(data_d.values(),  
 
                                       showmeans=True,
                                       showextrema=True,  
                                       )
                
                #===============================================================
                # color
                #===============================================================
                labels = list(data_d.keys())
                ckey_d = {i:color_key for i,color_key in enumerate(labels)}
                
                #style fills
                for i, pc in enumerate(parts_d['bodies']):
                    pc.set_facecolor(color_d[ckey_d[i]])
                    pc.set_edgecolor(color_d[ckey_d[i]])
                    pc.set_alpha(0.5)
                    
                #style lines
                for partName in ['cmeans', 'cbars', 'cmins', 'cmaxes']:
                    parts_d[partName].set(color='black', alpha=0.5)
 
            else:
                raise KeyError('unrecognized plot_type: %s'%plot_type)
 
            #===================================================================
            # post-format----
            #===================================================================
            ax.set_title(' & '.join(['%s:%s' % (k, v) for k, v in keys_d.items()]))
            #===================================================================
            # labels
            #===================================================================
            if add_label:
                # get float labels
                meta_d.update({ 
                    'min':gdata.min().min(), 'max':gdata.max().max(), 'mean':gdata.mean().mean(),
                          })
 
                ax.text(0.1, 0.9, get_dict_str(meta_d), transform=ax.transAxes, va='top', fontsize=8, color='black')
                
            #===================================================================
            # post-meta--------
            #===================================================================
            meta_serx = pd.Series(meta_d, name=gkeys)
            if meta_dx is None:
                meta_dx = meta_serx.to_frame().T
                meta_dx.index.set_names(keys_d.keys(), inplace=True)
            else:
                meta_dx = meta_dx.append(meta_serx)
                
        #===============================================================
        # #wrap format subplot
        #===============================================================
        """best to loop on the axis container in case a plot was missed"""
        for row_key, d in ax_d.items():
            for col_key, ax in d.items():
                if plot_type=='scatter':
                    ax.legend(loc=1)
                # first row
                if row_key == row_keys[0]:
                    pass
                #last col
                if col_key == col_keys[-1]:
                    pass
                    
                
                        
                # first col
                if col_key == col_keys[0]:
                    if plot_type in ['bars']:
                        ax.set_ylabel('\'%s\' total errors (%s)'%(dkey, err_type))
                    elif plot_type == 'violin':
                        ax.set_ylabel('\'%s\' errors'%(dkey))
                    elif plot_type=='scatter':
                        ax.set_ylabel('\'%s\' (true)'%dkey)
                
                #last row
                if row_key == row_keys[-1]:
                    if plot_type == 'bars': 
                        pass
                        #ax.set_ylabel('\'%s\' (agg - true)'%dkey)
                    elif plot_type=='violin':
                        ax.set_xticks(np.arange(1, len(labels) + 1))
                        ax.set_xticklabels(labels)
                        
                    else:
                        ax.set_xlabel('\'%s\' (aggregated)'%dkey)
                        
                    
 
 
        #=======================================================================
        # wrap---------
        #=======================================================================
        log.info('finsihed')
        """
        plt.show()
        """
        if plot_type=='bar':
            fname = 'compareMat_%s_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, err_type, plot_rown, plot_coln, self.longname)
        else:
            fname='compareMat_%s_%s_%sX%s_%s' % (
            title.replace(' ','').replace('\'',''),
             plot_type, plot_rown, plot_coln, self.longname)
        
        fname = fname.replace('=', '-')
        if write_meta:
            ofp =  os.path.join(self.out_dir, fname+'_meta.csv')
            meta_dx.to_csv(ofp)
            log.info('wrote meta_dx %s to \n    %s'%(str(meta_dx.shape), ofp))
               
        
        return self.output_fig(fig, fname=fname, **kwargs)
 
 
    
    #===========================================================================
    # HELPERS---------
    #===========================================================================
    
