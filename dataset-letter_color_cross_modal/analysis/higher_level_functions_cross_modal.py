#!/usr/bin/env python
# encoding: utf-8
"""
Letter-color associations formed through cross-modal statistical learning
"Letter-color cross modal 2AFC task" for short
Python code by O.Colizoli 2022
Python 3.6
"""

import os, sys, datetime
import numpy as np
import scipy as sp
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
from scipy.signal import decimate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re

#conda install -c conda-forge/label/gcc7 mne
from copy import deepcopy
import itertools
# import pingouin as pg # stats package
# from pingouin import pairwise_ttests

from IPython import embed as shell


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 6, 
    'ytick.labelsize': 6, 
    'legend.fontsize': 6, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()

############################################
# Define parameters
############################################

class higherLevel(object):
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        self.subjects = subjects
        self.exp = experiment_name
        self.project_directory = project_directory
        self.figure_folder = os.path.join(project_directory, 'figures')
        self.dataframe_folder = os.path.join(project_directory, 'data_frames')
        self.sample_rate = sample_rate
        self.time_locked = time_locked
        self.pupil_step_lim = pupil_step_lim                
        self.baseline_window = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        self.trial_bin_folder = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
            
        ##############################    
        # Pupil time series information:
        ##############################
        self.downsample_rate = 20 # 20 Hz
        self.downsample_factor = self.sample_rate / self.downsample_rate
        
    def lin_regress_betas(self, Y, X, eq='ols'):
        # prepare data:
        d = {'Y' : pd.Series(Y),}
        for i in range(len(X)):        
            d['X{}'.format(i)] = pd.Series(X[i])
        data = pd.DataFrame(d) # columns = Y plus X0 ... XN
    
        # using formulas adds constant in statsmodesl by default, otherwise sm2.add_constant(data)
        formula = 'Y ~ X0' 
        if len(X) > 1:
            for i in range(1,len(X)):
                formula = formula + ' + X{}'.format(i) # 'Y ~ X0 + X1 + X2 + X3'
        # fit:
        if eq == 'ols':
            model = sm.ols(formula=formula, data=data)
        fitted = model.fit()
        return np.array(fitted.params) # return beta coefficients
        
    def lin_regress_residuals(self, Y, X, eq='ols'):
        # prepare data:
        d = {'Y' : pd.Series(Y),}
        for i in range(len(X)):        
            d['X{}'.format(i)] = pd.Series(X[i])
        data = pd.DataFrame(d) # columns = Y plus X0 ... XN
    
        # using formulas adds constant in statsmodesl by default, otherwise sm2.add_constant(data)
        formula = 'Y ~ X0' 
        if len(X) > 1:
            for i in range(1,len(X)):
                formula = formula + ' + X{}'.format(i) # 'Y ~ X0 + X1 + X2 + X3'
        # fit:
        if eq == 'ols':
            model = sm.ols(formula=formula, data=data, missing='none')
        fitted = model.fit()
        return np.array(fitted.resid) # return beta coefficients
    
    def tsplot(self, ax, data, alpha_fill=0.2,alpha_line=1, **kw):
        # replacing seaborn tsplot
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        cis = self.bootstrap(data)
        ax.fill_between(x,cis[0],cis[1],alpha=alpha_fill,**kw) # debug double label!
        ax.plot(x,est,alpha=alpha_line,**kw)
        ax.margins(x=0)
    
    
    def bootstrap(self, data, n_boot=10000, ci=68):
        # bootstrap confidence interval for new tsplot
        boot_dist = []
        for i in range(int(n_boot)):
            resampler = np.random.randint(0, data.shape[0], data.shape[0])
            sample = data.take(resampler, axis=0)
            boot_dist.append(np.mean(sample, axis=0))
        b = np.array(boot_dist)
        s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
        s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
        return (s1,s2)

    # common functions
    def cluster_sig_bar_1samp(self,array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
        # permutation-based cluster correction on time courses, then plots the stats as a bar in yloc
        if yloc == 1:
            yloc = 10
        if yloc == 2:
            yloc = 20
        if yloc == 3:
            yloc = 30
        if yloc == 4:
            yloc = 40
        if yloc == 5:
            yloc = 50

        if cluster_correct:
            whatever, clusters, pvals, bla = mne.stats.permutation_cluster_1samp_test(array, n_permutations=nrand, n_jobs=10)
            for j, cl in enumerate(clusters):
                if len(cl) == 0:
                    pass
                else:
                    if pvals[j] < threshold:
                        for c in cl:
                            sig_bool_indices = np.arange(len(x))[c]
                            xx = np.array(x[sig_bool_indices])
                            try:
                                xx[0] = xx[0] - (np.diff(x)[0] / 2.0)
                                xx[1] = xx[1] + (np.diff(x)[0] / 2.0)
                            except:
                                xx = np.array([xx - (np.diff(x)[0] / 2.0), xx + (np.diff(x)[0] / 2.0),]).ravel()
                            ax.plot(xx, np.ones(len(xx)) * ((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], color, alpha=1, linewidth=2.5)
        else:
            p = np.zeros(array.shape[1])
            for i in range(array.shape[1]):
                p[i] = sp.stats.ttest_rel(array[:,i], np.zeros(array.shape[0]))[1]
            sig_indices = np.array(p < 0.05, dtype=int)
            sig_indices[0] = 0
            sig_indices[-1] = 0
            s_bar = zip(np.where(np.diff(sig_indices)==1)[0]+1, np.where(np.diff(sig_indices)==-1)[0])
            for sig in s_bar:
                ax.hlines(((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0], x[int(sig[0])]-(np.diff(x)[0] / 2.0), x[int(sig[1])]+(np.diff(x)[0] / 2.0), color=color, alpha=1, linewidth=2.5)
    
    
    def timeseries_fwer_correction(self,  xind, color, ax, pvals, yloc=5, alpha=0.1, method='fdr_bh'):
        """Add Family-Wise Error Rate correction bar on time series plot.
        
        Parameters
        ----------
        xind : array
            x indices of plat
        
        color : string
            Color of bar

        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in
        
        pvals : array
            Input for FDR correction
        
        alpha : float
            Alpha value for p-value significance (default 0.05)

        method : 'bh' 
            Method for FDR correction (default 'negcorr')
        
        Notes
        -----
        false_discovery_control(ps, *, axis=0, method='bh')
        
        """
        # UNCORRECTED
        # yloc = yloc + 1
        # sig_indices = np.array(pvals < alpha, dtype=int)
        # yvalues = sig_indices * (((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0])
        # yvalues[yvalues == 0] = np.nan # or use np.nan
        # ax.plot(xind, yvalues, linestyle='None', marker='.', color=color, alpha=0.2)
        
        # FDR CORRECTED
        yloc = yloc + 1
        reject, pvals_adjusted, alphaA, alphaB = statsmodels.stats.multitest.multipletests(pvals, alpha, method='fdr_bh', is_sorted=False, returnsorted=False)
        sig_indices = np.array(pvals_adjusted < alpha, dtype=int)
        yvalues = sig_indices * (((ax.get_ylim()[1] - ax.get_ylim()[0]) / yloc)+ax.get_ylim()[0])
        yvalues[yvalues == 0] = np.nan # or use np.nan
        ax.plot(xind, yvalues, linestyle='None', marker='.', color=color, alpha=1)
        
    
    def fisher_transform(self,r):
        """Compute Fisher transform on correlation coefficient.
        
        Parameters
        ----------
        r : array_like
            The coefficients to normalize
        
        Returns
        -------
        0.5*np.log((1+r)/(1-r)) : ndarray
            Array of shape r with normalized coefficients.
        """
        return 0.5*np.log((1+r)/(1-r))
    
    
    def higherlevel_get_phasics(self,):
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory,subj,'beh','{}_{}_beh.csv'.format(subj,self.exp)) # derivatives folder
            B = pd.read_csv(this_log, float_precision='%.16f') # behavioral file
            ### DROP EXISTING PHASICS COLUMNS TO PREVENT OLD DATA
            try: 
                B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                B = B.loc[:, ~B.columns.str.contains('_locked')] # remove all old phasic pupil columns
            except:
                pass
                
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                
                pupil_step_lim = self.pupil_step_lim[t] # kernel size is always the same for each event type
                
                for twi,pupil_time_of_interest in enumerate(self.pupil_time_of_interest[t]): # multiple time windows to average
                
                    # load evoked pupil file (all trials)
                    P = pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f') 
                    P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    P = np.array(P)

                    SAVE_TRIALS = []
                    for trial in np.arange(len(P)):
                        # in seconds
                        phase_start = -pupil_step_lim[0] + pupil_time_of_interest[0]
                        phase_end = -pupil_step_lim[0] + pupil_time_of_interest[1]
                        # in sample rate units
                        phase_start = int(phase_start*self.sample_rate)
                        phase_end = int(phase_end*self.sample_rate)
                        # mean within phasic time window
                        this_phasic = np.nanmean(P[trial,phase_start:phase_end]) 
                        SAVE_TRIALS.append(this_phasic)
                    # save phasics
                    B['pupil_{}_t{}'.format(time_locked,twi+1)] = np.array(SAVE_TRIALS)

                    #######################
                    B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B.to_csv(this_log, float_format='%.16f')
                    print('subject {}, {} phasic pupil extracted {}'.format(subj,time_locked,pupil_time_of_interest))
        print('success: higherlevel_get_phasics')
        
        
    def create_subjects_dataframe(self,blocks):
        """Combine behavior and phasic pupil dataframes of all subjects into a single large dataframe. 
        
        Notes
        -----
        Flag missing trials from concantenated dataframe.
        Output in dataframe folder: task-experiment_name_subjects.csv
        Merge with actual frequencies
        """
        DF = pd.DataFrame()
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_beh.csv'.format(subj,self.exp)), float_precision='%.16f')
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, 'beh', '{}_{}_recording-eyetracking_physio_{}_baselines.csv'.format(subj, self.exp, 'feed_locked')), float_precision='%.16f')
            this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['pupil_baseline_feed_locked'] = np.array(this_baseline)
            
            ###############################
            # flag missing trials
            this_data['missing'] = this_data['button']=='missing'
            this_data['drop_trial'] = np.array(this_data['missing']) #logical or
                        
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
       
        # count missing
        M = DF[DF['button']!='missing'] 
        missing = M.groupby(['subject','button'])['button'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)), float_format='%.16f')
        
        ### print how many outliers
        print('Missing = {}%'.format(np.true_divide(np.sum(DF['missing']),DF.shape[0])*100))
        print('Dropped trials = {}%'.format(np.true_divide(np.sum(DF['drop_trial']),DF.shape[0])*100))

        #####################
        # # merges the actual_frequencies and bins calculated from the oddball task logfiles into subjects' dataframe
        # FREQ = pd.read_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format('task-letter_color_cross_modal_training')), float_precision='%.16f')
        # FREQ = FREQ.drop(['frequency'],axis=1) # otherwise get double
        # FREQ = FREQ.loc[:, ~FREQ.columns.str.contains('^Unnamed')] # drop all unnamed columns
        # # inner merge on subject, letter, and color (r)
        # M = DF.merge(FREQ,how='inner',on=['subject','letter','r'])
        #
        # # actual frequencies average:
        # shell()
        # AF = M.groupby(['frequency','match'])['actual_frequency'].mean()
        # AF.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies_mean.csv'.format(self.exp)), float_format='%.16f')
        #
        # print('actual frequencies per matching condition')
        # print(AF)
        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_format='%.16f')
        #####################
        print('success: higherlevel_dataframe')
        
        
    def average_conditions(self, ):
        """Average the phasic pupil per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        'frequency' argument determines how the trials were split
        """    
        
        time_locked = 'resp_locked' 
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
                
        '''
        ######## CORRECT x FREQUENCY x TIME WINDOW ########
        '''
        DFOUT = DF.groupby(['subject','correct','frequency']).aggregate({'pupil_{}_t1'.format(time_locked):'mean', 'pupil_{}_t2'.format(time_locked):'mean'})
        # DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-mapping1-timewindow_{}.csv'.format(self.exp,pupil_dv))) # FOR PLOTTING
        # save for RMANOVA format
        DFANOVA =  DFOUT.unstack(['frequency','correct']) 
        print(DFANOVA.columns)
        DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
        DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency-timewindow_rmanova.csv'.format(self.exp)), float_format='%.16f') # for stats
        
        #interaction accuracy and frequency
        for pupil_dv in ['RT', 'pupil_{}_t1'.format(time_locked), 'pupil_{}_t2'.format(time_locked), 'pupil_baseline_{}'.format('feed_locked')]: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency','correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## CORRECT ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject','correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['correct', 'RT', 'pupil_{}_t1'.format(time_locked), 'pupil_{}_t2'.format(time_locked), 'pupil_baseline_{}'.format('feed_locked')]: # mean accuracy
            DFOUT = DF.groupby(['subject','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_conditions')
        

    def plot_phasic_pupil_pe(self,):
        # Phasic pupil feed_locked interaction between frequency and accuracy
        # in each time window of interest
        # GROUP LEVEL DATA
        # separate lines for correct, x-axis is mapping conditions
        
        ylim = [ 
            [-1.5,6.5], # t1
            [-3.25,2.25], # t2
        ]
        tick_spacer = [1.5,1]
        
        time_locked = 'feed_locked'
        dvs = ['pupil_{}_t1'.format(time_locked),'pupil_{}_t2'.format(time_locked)]
        ylabels = ['Pupil response\n(% signal change)','Pupil response\n(% signal change)']
        factor = ['frequency','correct']
        xlabel = 'Letter-color frequency'
        xticklabels = ['40%','70%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax = fig.add_subplot(len(ylabels),1,subplot_counter)
            subplot_counter += 1
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # plot line graph
            for x in[0,1]: # split by error, correct
                D = GROUP[GROUP['correct']==x]
                print(D)
                ax.errorbar(xind,np.array(D['mean']),yerr=np.array(D['sem']),fmt='-',elinewidth=1,label=labels[x],capsize=0, color=colors[x], alpha=1)

            # set figure parameters
            ax.set_title(pupil_dv)
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            # ax.set_ylim(ylim[dvi])
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer[dvi]))
            ax.legend()

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_correct-frequency_lines.pdf'.format(self.exp)))
        print('success: plot_phasic_pupil_pe')
        

    def plot_behavior(self,):
        # plots the group level means of accuracy and RT per mapping condition
        # whole figure, 2 subplots
        
        #######################
        # Mapping1
        #######################
        dvs = ['correct','RT']
        ylabels = ['Accuracy', 'RT (s)']
        factor = 'frequency'
        xlabel = 'Letter-color frequency'
        xticklabels = ['40%','70%'] 
        color = 'black'
        alphas = [0.4,0.7]
        
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,factor,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax = fig.add_subplot(int(len(ylabels)),1,int(subplot_counter)) # 1 subplot per bin window

            subplot_counter += 1
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            for xi,x in enumerate(GROUP[factor]): # frequency
                # ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), color='blue', alpha=alphas[xi], edgecolor='white', ecolor='black')
                ax.bar(xind[xi],np.array(GROUP['mean'][xi]), width=bar_width, yerr=np.array(GROUP['sem'][xi]), capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
            
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-',marker='o', markersize=3,fillstyle='full',color='black',alpha=0.05) # marker, line, black
            
            # set figure parameters
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            if pupil_dv == 'correct':
                ax.set_ylim([0.0,1.])
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.2))
                ax.axhline(0.5, linestyle='--', lw=1, alpha=1, color = 'k') # Add dashed horizontal line at chance level
            else:
                ax.set_ylim([0.2,1.8]) #RT
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(.4))
                
            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_behavior.pdf'.format(self.exp,factor)))
    
        print('success: plot_behav')
    
    def plot_uncertainty_rt(self,):
        # RT interaction between frequency and accuracy (uncertainty?)
        # GROUP LEVEL DATA
        # separate lines for correct, x-axis is mapping conditions
        ylim = [0.6,1.5]
        tick_spacer = .2
        
        dvs = ['RT']
        ylabels = ['RT (s)']
        factor = ['frequency','correct']
        xlabel = 'Letter-color frequency'
        xticklabels = ['40%','70%'] 
        labels = ['Error','Correct']
        colors = ['red','blue'] 
        
        xind = np.arange(len(xticklabels))
        
        fig = plt.figure(figsize=(2,2*len(ylabels)))
        subplot_counter = 1
        
        for dvi,pupil_dv in enumerate(dvs):

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)))
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            ax = fig.add_subplot(len(ylabels),1,subplot_counter)
            subplot_counter += 1
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # plot line graph
            for x in[0,1]: # split by error, correct
                D = GROUP[GROUP['correct']==x]
                print(D)
                ax.errorbar(xind,np.array(D['mean']),yerr=np.array(D['sem']),fmt='-',elinewidth=1,label=labels[x],capsize=0, color=colors[x], alpha=1)

            # set figure parameters
            ax.set_title(pupil_dv)
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
            # ax.legend()

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_correct-frequency_lines_{}.pdf'.format(self.exp,pupil_dv)))
        print('success: plot_uncertainty_rt')
        
        
    def dataframe_evoked_pupil_higher(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP PHASE 2 trials.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns   
        csv_names = deepcopy(['subject','correct','frequency','correct-frequency'])
        factors = [['subject'],['correct'],['frequency'],['correct','frequency']]
        
        for t,time_locked in enumerate(self.time_locked):
            # Loop through conditions                
            for c,cond in enumerate(csv_names):
                # intialize dataframe per condition
                COND = pd.DataFrame()
                g_idx = deepcopy(factors)[c]       # need to add subject idx for groupby()
                
                if not cond == 'subject':
                    g_idx.insert(0, 'subject') # get strings not list element
                
                for s,subj in enumerate(self.subjects):
                    subj_num = re.findall(r'\d+', subj)[0]
                    SBEHAV = DF[DF['subject']==int(subj_num)].reset_index() # not 'sub-' in DF
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj,'beh','{}_{}_recording-eyetracking_physio_{}_evoked_basecorr.csv'.format(subj,self.exp,time_locked)), float_precision='%.16f'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    # merge behavioral and evoked dataframes so we can group by conditions
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)
                    
                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['drop_trial'] == 0] # drop outliers based on RT
                    #############################
                    
                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    df = SDATA.groupby(g_idx)[evoked_cols].mean() # only get kernels out
                    df = pd.DataFrame(df).reset_index()
                    # add to condition dataframe
                    COND = pd.concat([COND,df],join='outer',axis=0) # can also do: this_cond = this_cond.append()  
                # save output file
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp,time_locked,cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher')
        
    
    def plot_evoked_pupil(self):
        # plots evoked pupil 2 subplits
        # plots the group level mean for target_locked
        # plots the group level accuracy x mapping interaction for target_locked

        ylim_feed = [-4,15]
        tick_spacer = 3
        
        t = 0
        time_locked = 'feed_locked'
        
        fig = plt.figure(figsize=(8,2))
        #######################
        # FEEDBACK MEAN RESPONSE
        #######################
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(141)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, factor)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
    
        xticklabels = ['mean response']
        colors = ['black'] # black
        alphas = [1]

        # plot time series
        i=0
        TS = np.array(COND.iloc[:,-kernel:]) # index from back to avoid extra unnamed column pandas
        self.tsplot(ax, TS, color='k', label=xticklabels[i])
        self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)
    
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from {} (s)'.format(self.time_locked))
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(self.time_locked)
                
        # compute peak of mean response to center time window around
        m = np.mean(TS,axis=0)
        argm = np.true_divide(np.argmax(m),self.sample_rate) + self.pupil_step_lim[t][0] # subtract pupil baseline to get timing
        print('mean response = {} peak @ {} seconds'.format(np.max(m),argm))
        # ax.axvline(np.argmax(m), lw=0.25, alpha=0.5, color = 'k')
        
        #######################
        # CORRECT
        #######################
        csv_name = 'correct'
        factor = 'correct'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(142)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked,csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['Error','Correct']
        colorsts = ['r','b',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [1,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from {} (s)'.format(self.time_locked))
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(self.time_locked)
        # ax.legend(loc='best')
        
        #######################
        # FREQUENCY
        #######################
        csv_name = 'frequency'
        factor = 'frequency'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(143)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
        xticklabels = ['40%','70%']
        colorsts = ['indigo','indigo',]
        alpha_fills = [0.2,0.2] # fill
        alpha_lines = [0.4,1]
        save_conds = []
        
        # plot time series
        for i,x in enumerate(np.unique(COND[factor])):
            TS = COND[COND[factor]==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_difference = save_conds[0]-save_conds[1]
        self.cluster_sig_bar_1samp(array=pe_difference, x=pd.Series(range(pe_difference.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from {} (s)'.format(time_locked))
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(time_locked)
        ax.legend(loc='best')
        
        #######################
        # CORRECT x FREQUENCY
        #######################
        csv_name = 'correct-frequency'
        factor = ['correct','frequency']
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)

        ax = fig.add_subplot(144)
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0

        # Compute means, sems across group
        COND = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, csv_name)))
        COND = COND.loc[:, ~COND.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ########
        # make unique labels for each of the 4 conditions
        conditions = [
            (COND['correct'] == 0) & (COND['frequency'] == 70), # Easy Error 1
            (COND['correct'] == 1) & (COND['frequency'] == 70), # Easy Correct 2
            (COND['correct'] == 0) & (COND['frequency'] == 40), # Hard Error 3
            (COND['correct'] == 1) & (COND['frequency'] == 40), # Hard Correct 4
            ]
        values = [1,2,3,4]
        conditions = np.select(conditions, values) # don't add as column to time series otherwise it gets plotted
        ########
                    
        xticklabels = ['Error 70%','Correct 70%','Error 40%','Correct 40%']
        colorsts = ['r','b','r','b']
        alpha_fills = [0.2,0.2,0.05,0.05] # fill
        alpha_lines = [1,1,0.4,0.4]
        save_conds = []
        # plot time series
        
        for i,x in enumerate(values):
            TS = COND[conditions==x] # select current condition data only
            TS = np.array(TS.iloc[:,-kernel:])
            self.tsplot(ax, TS, color=colorsts[i], label=xticklabels[i], alpha_fill=alpha_fills[i], alpha_line=alpha_lines[i])
            save_conds.append(TS) # for stats
        
        # stats        
        ### COMPUTE INTERACTION TERM AND TEST AGAINST 0!
        pe_interaction = (save_conds[0]-save_conds[1]) - (save_conds[2]-save_conds[3])
        self.cluster_sig_bar_1samp(array=pe_interaction, x=pd.Series(range(pe_interaction.shape[-1])), yloc=1, color='black', ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)

        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
        
        xticks = [event_onset,mid_point,end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0,np.true_divide(self.pupil_step_lim[t][1],2),self.pupil_step_lim[t][1]])
        ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from {} (s)'.format(self.time_locked))
        ax.set_ylabel('Pupil response\n(% signal change)')
        ax.set_title(self.time_locked)
        ax.legend(loc='best')
                
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_evoked.pdf'.format(self.exp)))
        print('success: plot_evoked_pupil')

    def individual_differences(self,):
       # correlate interaction term in pupil with frequency effect in accuracy
       
       time_locked = 'feed_locked'
       dvs = ['pupil_{}_t1'.format(time_locked),'pupil_{}_t2'.format(time_locked)]
       
       fig = plt.figure(figsize=(2*len(dvs),2))
       
       for sp,pupil_dv in enumerate(dvs):
           ax = fig.add_subplot(1,1*len(dvs),sp+1) # 1 subplot per bin window
           
           B = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_correct_rmanova.csv'.format(self.exp)))
           P = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp, pupil_dv)))

           # frequency effect
           P['main_effect_freq'] = (P['70']-P['40'])
           B['main_effect_freq'] = (B['70']-B['40']) # fraction correct
           
           x = np.array(B['main_effect_freq'])
           y = np.array(P['main_effect_freq'])           
           # all subjects
           r1,pval1 = stats.spearmanr(x,y)
           print('all subjects')
           print(pupil_dv)
           print('r={}, p-val={}'.format(r1,pval1))
           # shell()
           # all subjects in grey
           ax.plot(x, y, 'o', markersize=3, color='black') # marker, line, black
           m, b = np.polyfit(x, y, 1)
           ax.plot(x, m*x+b, color='black',alpha=.5, label='all participants')
           
           # test with subjects that do better than chance on average
           B['mean'] = (B['70']+B['40'])/2
           B['remove'] = B['mean'] <= .5
           # B['remove'] = B['70'] <= .5 # only select on subjects for which high frequency was learned
           
           print('N subjects removed = {}'.format(np.sum(B['remove'])))
           
           x = x[B['remove']==0]
           y = y[B['remove']==0]
           
           r2,pval2 = stats.spearmanr(x,y)
           print('better than chance subjects')
           print(pupil_dv)
           print('r={}, p-val={}'.format(r2,pval2))
           
           # better than chance subjects in green
           ax.plot(x, y, 'o', markersize=3, color='green') # marker, line, black
           m, b = np.polyfit(x, y, 1)
           ax.plot(x, m*x+b, color='green', alpha=.5, label='> chance level')
           
           # set figure parameters
           ax.set_title('r1={}, p-val1={}\nr2={}, p-val2={}'.format(np.round(r1,2),np.round(pval1,3),np.round(r2,2),np.round(pval2,3)))
           ax.set_ylabel('{} (70%-40%)'.format(pupil_dv))
           ax.set_xlabel('accuracy (70%-40%)')
           ax.legend()
           
       plt.tight_layout()
       fig.savefig(os.path.join(self.figure_folder,'{}_frequency_individual_differences.pdf'.format(self.exp)))
       print('success: individual_differences')
       
       
    def dataframe_evoked_correlation_AQ(self, df):
        """Timeseries individual differences correlation between pupil and AQ score for:
        all trials, correct vs. error, frequency (20-80%), interaction term (20-80%)

        Notes
        -----
        Omissions dropped in dataframe_evoked_pupil_higher()
        Output correlation coefficients per time point in dataframe folder.
        """
        
        AQ = df        
        iv = 'aq_score'
        aq_scores = np.array(AQ[iv])
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        df_out = pd.DataFrame() # timepoints x condition
        
        for t,time_locked in enumerate(self.time_locked):
            
            # for cond in ['subject', 'correct', 'frequency', 'correct-frequency']:
            for cond in ['subject', 'correct', 'frequency']:
            
                DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, cond)), float_precision='%.16f')
                DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
                # merge AQ and pupil dataframes on subject to make sure have only data containing AQ scores
                this_pupil = DF.loc[DF['subject'].isin(AQ['subject'])]
                
                # get columns of pupil sample points only
                kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate)
                evoked_cols = np.char.mod('%d', np.arange(kernel)) 
                
                if cond == 'correct':
                    # error-correct
                    this_pupil = np.subtract(this_pupil[this_pupil['correct']==0].copy(), this_pupil[this_pupil['correct']==1].copy())
                
                elif cond == 'frequency':
                    # 20%-80%
                    this_pupil = np.subtract(this_pupil[this_pupil['frequency']==40].copy(), this_pupil[this_pupil['frequency']==70].copy()) 
                
                elif cond == 'correct-frequency':
                    # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                    term1 = np.subtract(this_pupil[(this_pupil['frequency']==70) & (this_pupil['correct']==0)].copy() , this_pupil[(this_pupil['frequency']==70) & (this_pupil['correct']==1)].copy() )
                    term2 = np.subtract(this_pupil[(this_pupil['frequency']==40) & (this_pupil['correct']==0)].copy() ,  this_pupil[(this_pupil['frequency']==40) & (this_pupil['correct']==1)].copy() )
                    this_pupil = np.subtract(term1, term2)
                    
                # loop timepoints, regress
                save_timepoint_r = []
                save_timepoint_p = []
                for col in evoked_cols:
                    Y = this_pupil[col] # pupil
                    X = aq_scores # iv
                    try:
                        r, pval = sp.stats.spearmanr(np.array(X), np.array(Y))
                    except:
                        shell()
                    save_timepoint_r.append(self.fisher_transform(r))
                    save_timepoint_p.append(pval)
                    
                # add column for each subject with timepoints as rows
                df_out['{}_r'.format(cond)] = np.array(save_timepoint_r)
                df_out['{}_pval'.format(cond)] = np.array(save_timepoint_p)
                # remove scientific notation from df
                df_out['{}_r'.format(cond)] = df_out['{}_r'.format(cond)].apply(lambda x: '%.16f' % x) 
                df_out['{}_pval'.format(cond)] = df_out['{}_pval'.format(cond)].apply(lambda x: '%.16f' % x)
                
            # save output file
            df_out.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}.csv'.format(self.exp, time_locked, iv)), float_format='%.16f')
        print('success: dataframe_evoked_correlation_AQ')
        
        
    def plot_evoked_correlation_AQ(self):
        """Plot partial correlation between pupil response and model estimates.
        
        Notes
        -----
        Always feed_locked pupil response.
        Partial correlations are done for all trials as well as for correct and error trials separately.
        """
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        iv = 'aq_score'
    
        colors = ['black', 'red', 'purple', 'orange'] 
        alphas = [1]
        # labels = ['All Trials' , 'Error-Correct', '40%-70%', 'Accuracy*Frequency']
        labels = ['All Trials' , 'Error-Correct', '40%-70%']
        
        time_locked = 'feed_locked'
        t = 0
        CORR = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}.csv'.format(self.exp, time_locked, iv)), float_precision='%.16f')
        
        #######################
        # FEEDBACK PLOT R FOR EACH PUPIl COND
        #######################
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*self.sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*self.sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # for i,cond in enumerate(['subject', 'correct', 'frequency', 'correct-frequency']):
        for i,cond in enumerate(['subject', 'correct', 'frequency',]):
        
            # Compute means, sems across group
            TS = np.array(CORR['{}_r'.format(cond)])
            pvals = np.array(CORR['{}_pval'.format(cond)])
            
            ax.plot(pd.Series(range(TS.shape[-1])), TS, color=colors[i], label=labels[i])
                
            # stats        
            self.timeseries_fwer_correction(pvals=pvals, xind=pd.Series(range(pvals.shape[-1])), color=colors[i], yloc=i, ax=ax)
            
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        xticks = [event_onset, ((mid_point-event_onset)/2)+event_onset, mid_point, ((end_sample-mid_point)/2)+mid_point, end_sample]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]*.25, self.pupil_step_lim[t][1]*.5, self.pupil_step_lim[t][1]*.75, self.pupil_step_lim[t][1]])
        # ax.set_ylim(ylim_feed)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
        ax.set_xlabel('Time from feedback (s)')
        ax.set_ylabel('rs')
        ax.set_title(time_locked)
        ax.legend()
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        # plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_evoked_correlation_{}.pdf'.format(self.exp, time_locked, iv)))
        # shell()
        print('success: plot_evoked_correlation_AQ')
    
    
    def regression_pupil_AQ(self, df):
        """Multiple regression of AQ components (IVs) onto average pupil response in early time window
        Use 20%-80% pupil response as DV

        Notes
        -----
        print(results.summary())
        Quantities of interest can be extracted directly from the fitted model. Type dir(results) for a full list. Here are some examples:
        print('Parameters: ', results.params)
        print('R2: ', results.rsquared)
        """
        
        AQ = df        
        ivs = ['social', 'attention','communication','fantasy','detail']
        cond = 'frequency'
        dv = 'pupil_feed_locked_t1'
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        df_out = pd.DataFrame() # timepoints x condition
                    
        for cond in ['frequency']:
            
            # task-letter_color_visual_decision_frequency_pupil_feed_locked_t1.csv
            DF = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp, cond, dv)), float_precision='%.16f')
            DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # merge AQ and pupil dataframes on subject to make sure have only data containing AQ scores
            this_pupil = DF.loc[DF['subject'].isin(AQ['subject'])]

            if cond == 'correct':
                # error-correct
                this_pupil = np.subtract(this_pupil[this_pupil['correct']==0].copy(), this_pupil[this_pupil['correct']==1].copy())
            
            elif cond == 'frequency':
                # 20%-80%
                this_pupil = np.subtract(this_pupil[this_pupil['frequency']==20].copy(), this_pupil[this_pupil['frequency']==80].copy()) 
            
            elif cond == 'correct-frequency':
                # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                term1 = np.subtract(this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==0)].copy() , this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==1)].copy() )
                term2 = np.subtract(this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==0)].copy() ,  this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==1)].copy() )
                this_pupil = np.subtract(term1, term2)
            
            # ordinary least squares regression
            Y = this_pupil[dv].reset_index(drop=True)
            X = AQ[ivs].reset_index(drop=True)
            X = sm.add_constant(X)
            model = sm.OLS(Y,X)
            results = model.fit()
            results.params
            print(results.summary())
            
            # save model results in output file
            out_path = os.path.join(self.jasp_folder,'{}_{}_{}_OLS_summary.csv'.format(self.exp, cond, dv))

            text_file = open(out_path, "w")
            text_file.write(results.summary().as_text())
            text_file.close()
            
        print('success: regression_pupil_AQ')
        
       
    def correlation_frequency_AQ(self, df):
        
        AQ = df
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        time_locked = 'resp_locked'
        dvs = ['pupil_{}_t1'.format(time_locked),'pupil_{}_t2'.format(time_locked),'correct']
       
        fig = plt.figure(figsize=(2*len(dvs),2))
       
        for sp,pupil_dv in enumerate(dvs):
            ax = fig.add_subplot(1,1*len(dvs),sp+1) # 1 subplot per bin window
            ax.set_box_aspect(1)
            
            P = pd.read_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp, pupil_dv)))
            P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # make sure subjects are aligned
            M = AQ.merge(P, how='inner', on=['subject'])
            
            # frequency effect
            M['main_effect_freq'] = (M['40']-M['70'])
           
            x = np.array(M['aq_score'])
            y = np.array(M['main_effect_freq'])           
            # all subjects
            r1,pval1 = stats.spearmanr(x,y)
            print('all subjects')
            print(pupil_dv)
            print('r={}, p-val={}'.format(r1,pval1))
            # shell()
            # all subjects in grey
            ax.plot(x, y, 'o', markersize=3, color='green') # marker, line, black
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x+b, color='black',alpha=.5, label='all participants')
           
            if pupil_dv == 'correct':
                # test with subjects that do better than chance on average (only for accuracy)
                M['mean'] = (M['70']+M['40'])/2
                M['remove'] = M['mean'] <= .5
                print('N subjects removed = {}'.format(np.sum(M['remove'])))
           
                x = x[M['remove']==0]
                y = y[M['remove']==0]
           
                r2,pval2 = stats.spearmanr(x,y)
                print('all subjects')
                print(pupil_dv)
                print('r={}, p-val={}'.format(r2,pval2))
           
                # better than chance subjects in green
                ax.plot(x, y, 'o', markersize=3, color='black') # marker, line, black
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m*x+b, color='green', alpha=.5, label='> chance level')
                ax.set_title('rho1={}, p-val1={}\nrho2={}, p-val2={}'.format(np.round(r1,2),np.round(pval1,3),np.round(r2,2),np.round(pval2,3)))
            else:
                ax.set_title('rho1={}, p-val1={}'.format(np.round(r1,2),np.round(pval1,3)))
                
            # set figure parameters
            ax.set_ylabel('{} (40%-70%)'.format(pupil_dv))
            ax.set_xlabel('AQ score')
            ax.legend()
           
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_frequency_AQ.pdf'.format(self.exp)))
        print('success: correlation_frequency_AQ')


    def correlation_accuracy_AQ(self, df):
        
        AQ = df
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        time_locked = 'resp_locked'
        dvs = ['pupil_{}_t1'.format(time_locked),'pupil_{}_t2'.format(time_locked),'RT']
       
        fig = plt.figure(figsize=(2*len(dvs),2))
       
        for sp,pupil_dv in enumerate(dvs):
            ax = fig.add_subplot(1,1*len(dvs),sp+1) # 1 subplot per bin window
            ax.set_box_aspect(1)
            
            P = pd.read_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp, pupil_dv)))
            P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # make sure subjects are aligned
            M = AQ.merge(P, how='inner', on=['subject'])

            # frequency effect
            M['main_effect'] = (M['False']-M['True'])
           
            x = np.array(M['aq_score'])
            y = np.array(M['main_effect'])           
            # all subjects
            r1,pval1 = stats.spearmanr(x,y)
            print('all subjects')
            print(pupil_dv)
            print('r={}, p-val={}'.format(r1,pval1))
            # shell()
            # all subjects in grey
            ax.plot(x, y, 'o', markersize=3, color='green') # marker, line, black
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m*x+b, color='black',alpha=.5, label='all participants')
           
            
            ax.set_title('rho1={}, p-val1={}'.format(np.round(r1,2),np.round(pval1,3)))
                
            # set figure parameters
            ax.set_ylabel('{} (Error-Correct)'.format(pupil_dv))
            ax.set_xlabel('AQ score')
            ax.legend()
           
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_accuracy_AQ.pdf'.format(self.exp)))
        print('success: correlation_accuracy_AQ')       
       
       
       
       
       