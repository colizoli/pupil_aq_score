#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #2 Letter-color 2AFC task - Higher Level Functions
Python code O.Colizoli 2023 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>>> conda install -c conda-forge/label/gcc7 mne
================================================
"""

import os, sys, datetime
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mne
import re
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
from copy import deepcopy
import itertools
from IPython import embed as shell # for debugging only

pd.set_option('display.float_format', lambda x: '%.8f' % x) # suppress scientific notation in pandas

""" Plotting Format
############################################
# PLOT SIZES: (cols,rows)
# a single plot, 1 row, 1 col (2,2)
# 1 row, 2 cols (2*2,2*1)
# 2 rows, 2 cols (2*2,2*2)
# 2 rows, 3 cols (2*3,2*2)
# 1 row, 4 cols (2*4,2*1)
# Nsubjects rows, 2 cols (2*2,Nsubjects*2)

############################################
# Define parameters
############################################
"""
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.linewidth': 1, 
    'axes.labelsize': 7, 
    'axes.titlesize': 7, 
    'xtick.labelsize': 7, 
    'ytick.labelsize': 7, 
    'legend.fontsize': 7, 
    'xtick.major.width': 1, 
    'ytick.major.width': 1,
    'text.color': 'Black',
    'axes.labelcolor':'Black',
    'xtick.color':'Black',
    'ytick.color':'Black',} )
sns.plotting_context()


class higherLevel(object):
    """Define a class for the higher level analysis.

    Parameters
    ----------
    subjects : list
        List of subject numbers
    experiment_name : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])

    Attributes
    ----------
    subjects : list
        List of subject numbers
    exp : string
        Name of the experiment for output files
    project_directory : str
        Path to the derivatives data directory
    figure_folder : str
        Path to the figure directory
    dataframe_folder : str
        Path to the dataframe directory
    trial_bin_folder : str
        Path to the trial bin directory for conditions 
    jasp_folder : str
        Path to the jasp directory for stats
    sample_rate : int
        Sampling rate of pupil measurements in Hertz
    time_locked : list
        List of strings indiciting the events for time locking that should be analyzed (e.g., ['cue_locked','target_locked'])
    pupil_step_lim : list 
        List of arrays indicating the size of pupil trial kernels in seconds with respect to first event, first element should max = 0! (e.g., [[-baseline_window,3],[-baseline_window,3]] )
    baseline_window : float
        Number of seconds before each event in self.time_locked that are averaged for baseline correction
    pupil_time_of_interest : list
        List of arrays indicating the time windows in seconds in which to average evoked responses, per event in self.time_locked, see in higher.plot_evoked_pupil (e.g., [[1.0,2.0],[1.0,2.0]])
    """
    
    def __init__(self, subjects, experiment_name, project_directory, sample_rate, time_locked, pupil_step_lim, baseline_window, pupil_time_of_interest):        
        """Constructor method
        """
        self.subjects           = subjects
        self.exp                = experiment_name
        self.project_directory  = project_directory
        self.figure_folder      = os.path.join(project_directory, 'figures')
        self.dataframe_folder   = os.path.join(project_directory, 'data_frames')
        self.trial_bin_folder   = os.path.join(self.dataframe_folder,'trial_bins_pupil') # for average pupil in different trial bin windows
        self.jasp_folder        = os.path.join(self.dataframe_folder,'jasp') # for dataframes to input into JASP
        ##############################    
        # Pupil time series information:
        ##############################
        self.sample_rate        = sample_rate
        self.time_locked        = time_locked
        self.pupil_step_lim     = pupil_step_lim                
        self.baseline_window    = baseline_window              
        self.pupil_time_of_interest = pupil_time_of_interest
        
        if not os.path.isdir(self.figure_folder):
            os.mkdir(self.figure_folder)
            
        if not os.path.isdir(self.dataframe_folder):
            os.mkdir(self.dataframe_folder)
        
        if not os.path.isdir(self.trial_bin_folder):
            os.mkdir(self.trial_bin_folder)
            
        if not os.path.isdir(self.jasp_folder):
            os.mkdir(self.jasp_folder)
    
    
    def tsplot(self, ax, data, alpha_fill=0.2, alpha_line=1, **kw):
        """Time series plot replacing seaborn tsplot
            
        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in

        data : array
            The data in matrix of format: subject x timepoints

        alpha_line : int
            The thickness of the mean line (default 1)

        kw : list
            Optional keyword arguments for matplotlib.plot().
        """
        x = np.arange(data.shape[1])
        est = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        ## confidence intervals
        # cis = self.bootstrap(data)
        # ax.fill_between(x,cis[0],cis[1],alpha=alpha_fill,**kw) # debug double label!
        ## standard error mean
        sde = np.true_divide(sd, np.sqrt(data.shape[0]))        
        # shell()
        fill_color = kw['color']
        ax.fill_between(x, est-sde, est+sde, alpha=alpha_fill, color=fill_color, linewidth=0.0) # debug double label!
        
        ax.plot(x, est, alpha=alpha_line, **kw)
        ax.margins(x=0)
    

    def timeseries_fwer_correction(self,  xind, color, ax, pvals, alpha, yloc=5, method='fdr_bh'):
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
        
        yloc : int
            Y-axis location for the significance bar.

        method : string
            Method for FDR correction (default 'fdr_bh')
        
        Notes
        -----
        false_discovery_control(ps, *, axis=0, method='bh')
        
        """
        # CORRECTED
        yloc = yloc + 5
        reject, pvals_adjusted, alphaA, alphaB = statsmodels.stats.multitest.multipletests(pvals, alpha, method=method, is_sorted=False, returnsorted=False)
                
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
        # merges the actual_frequencies and bins calculated from the oddball task logfiles into subjects' dataframe
        FREQ = pd.read_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies.csv'.format('task-letter_color_visual_training')), float_precision='%.16f')
        FREQ = FREQ.drop(['frequency'],axis=1) # otherwise get double
        FREQ = FREQ.loc[:, ~FREQ.columns.str.contains('^Unnamed')] # drop all unnamed columns
        # inner merge on subject, letter, and color (r)
        M = DF.merge(FREQ,how='inner',on=['subject','letter','r'])
        
        # actual frequencies average:
        AF = M.groupby(['frequency','match'])['actual_frequency'].mean()
        AF.to_csv(os.path.join(self.dataframe_folder,'{}_actual_frequencies_mean.csv'.format(self.exp)), float_format='%.16f')
        
        print('actual frequencies per matching condition')
        print(AF) 
        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_format='%.16f')
        #####################
        print('success: higherlevel_dataframe')
        

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
    
    
    def dataframe_evoked_correlation_AQ(self, df):
        """Timeseries individual differences (Spearman rank) correlation between pupil and AQ score for:
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
            
            for cond in ['subject', 'correct', 'frequency', 'correct-frequency']:
            
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
                    this_pupil = np.subtract(this_pupil[this_pupil['frequency']==20].copy(), this_pupil[this_pupil['frequency']==80].copy()) 
                
                elif cond == 'correct-frequency':
                    # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                    term1 = np.subtract(this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==0)].copy() , this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==1)].copy() )
                    term2 = np.subtract(this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==0)].copy() ,  this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==1)].copy() )
                    this_pupil = np.subtract(term1, term2)
                                 
                # loop timepoints, regress
                save_timepoint_r = []
                save_timepoint_p = []
                
                for col in evoked_cols:
                    Y = this_pupil[col] # pupil
                    X = aq_scores # iv
                    
                    r, pval = sp.stats.spearmanr(np.array(X), np.array(Y))

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
        labels = ['All Trials' , 'Error-Correct', '20%-80%', 'Accuracy*Frequency']
        
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
        
        for i,cond in enumerate(['subject', 'correct', 'frequency', 'correct-frequency']):
            # Compute means, sems across group
            TS = np.array(CORR['{}_r'.format(cond)])
            pvals = np.array(CORR['{}_pval'.format(cond)])
                        
            ax.plot(pd.Series(range(TS.shape[-1])), TS, color=colors[i], label=labels[i])
                
            # stats        
            self.timeseries_fwer_correction(pvals=pvals, xind=pd.Series(range(pvals.shape[-1])), alpha=0.07, color=colors[i], yloc=i, ax=ax)
            
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*self.sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:       
            tw_begin = int(event_onset + (twi[0]*self.sample_rate))
            tw_end = int(event_onset + (twi[1]*self.sample_rate))
            ax.axvspan(tw_begin,tw_end, facecolor='k', alpha=0.1)
            
        xticks = [event_onset, event_onset+(500*1), event_onset+(500*2), event_onset+(500*3), event_onset+(500*4), event_onset+(500*5), event_onset+(500*6), event_onset+(500*7)]
        ax.set_xticks(xticks)
        ax.set_xticklabels([0, self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])

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
        
                
    def average_conditions(self, ):
        """Average the phasic pupil per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
        
        '''
        ######## SUBJECT AVERAGE ########
        '''
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: 

            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject',])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.jasp_folder,'{}_subject_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING
           
        #interaction accuracy and frequency
        for pupil_dv in ['RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x correct x frequency
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
            # MEANS subject x correct
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
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            
        ### SPLIT BY BLOCK ###
        #interaction accuracy and frequency
        for pupil_dv in ['RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: #interaction accuracy and frequency
            
            '''
            ######## BLOCK x CORRECT x FREQUENCY ########
            '''
            # MEANS subject x block x correct x frequency
            DFOUT = DF.groupby(['subject','block','correct','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency','correct','block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## BLOCK x CORRECT ########
            '''
            # MEANS subject x block x correct
            DFOUT = DF.groupby(['subject','block','correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_block-correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct','block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        '''
        ######## BLOCK x FREQUENCY ########
        '''
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: # mean accuracy
            DFOUT = DF.groupby(['subject','block','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_block-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency','block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_conditions')
        
        
    def regression_pupil_AQ(self, df):
        """Multiple regression of AQ components (IVs) onto average pupil response in early time window. Use 20%-80% pupil response as DV.

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
        
    
    def correlation_AQ(self, df):
        """Calculate the Spearman rank correlations between AQ score (and sub-scores) with other DVs of interest. Plot data.

        Notes
        -----
        ivs = ['aq_score', 'social', 'attention', 'communication', 'fantasy', 'detail']
        dvs = ['pupil_feed_locked_t1', 'pupil_baseline_feed_locked', 'RT', 'correct']
        conditions = ['subject', 'correct', 'frequency', 'correct-frequency']
        freqs = ['20', '80'] # low, high to contrast
        """
        
        AQ = df
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        ivs = ['aq_score', 'social', 'attention', 'communication', 'fantasy', 'detail']
        # ivs = ['aq_score', ]
        dvs = ['pupil_feed_locked_t1', 'pupil_baseline_feed_locked', 'RT', 'correct']
        conditions = ['subject', 'correct', 'frequency', 'correct-frequency']
        freqs = ['20', '80'] # low, high to contrast
        
        for iv in ivs:
            # new figure for every IV
            fig = plt.figure(figsize=(2*len(dvs),2*len(conditions)))
            counter = 1 # subplot counter
            
            for cond in conditions:
                for i,dv in enumerate(dvs):
                    
                    # can't correlate accuracy with 'correct' as condition
                    if (dv =='correct' and cond == 'correct') or (dv =='correct' and cond == 'correct-frequency'):
                        pass
                    else:
                        # run correlation
                        ax = fig.add_subplot(len(conditions), len(dvs), counter) # 1 subplot per bin window
                        ax.set_box_aspect(1)
                        
                        P = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_rmanova.csv'.format(self.exp, cond, dv)))
                        P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
                        # make sure subjects are aligned
                        M = AQ.merge(P, how='inner', on=['subject'])
                    
                        if cond == 'subject':
                            M['main_effect_{}'.format(cond)] = M[dv]
                            ax.set_ylabel('{} average'.format(dv))
                        
                        elif cond == 'frequency':            
                            # frequency effect
                            M['main_effect_{}'.format(cond)] = (M[freqs[0]]-M[freqs[1]])
                            ax.set_ylabel('{} ({}%-{}%)'.format(dv, freqs[0], freqs[1]))
                        
                        elif cond == 'correct':
                            # frequency effect
                            M['main_effect_{}'.format(cond)] = (M['False']-M['True'])
                            ax.set_ylabel('{} (Error-Correct)'.format(dv))
                        
                        elif cond == 'correct-frequency':
                            # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                            M['main_effect_{}'.format(cond)] = (M['({}, False)'.format(freqs[1])]-M['({}, True)'.format(freqs[1])])-(M['({}, False)'.format(freqs[0])]-M['({}, True)'.format(freqs[0])])
                            ax.set_ylabel('{} (interaction)'.format(dv))
                        
                        # correlation
                        x = np.array(M[iv])
                        y = np.array(M['main_effect_{}'.format(cond)])           
                        r,pval = stats.spearmanr(x,y)

                        # fit regression line
                        ax.plot(x, y, 'o', markersize=3, color='purple') # marker, line, black
                        m, b = np.polyfit(x, y, 1)
                        ax.plot(x, m*x+b, color='black',alpha=.5)
                        
                        # set figure parameters
                        ax.set_title('rs = {}, p = {}'.format(np.round(r,2),np.round(pval,3)))
                        ax.set_xlabel(iv)
                    counter = counter + 1
                    
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_correlation_{}.pdf'.format(self.exp, iv)))
        print('success: correlation_AQ')
        

    def plot_AQ_histogram(self, df):
        """Plot a histogram of the AQ score distribution (and sub-scores).
        """
        
        AQ = df
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        ivs = ['aq_score', 'social', 'attention', 'communication', 'fantasy', 'detail']
        
        # new figure for every IV
        fig = plt.figure(figsize=(2,2*len(ivs)))
        counter = 1 # subplot counter
        
        for iv in ivs:
            ax = fig.add_subplot(len(ivs), 1, counter) # 1 subplot per bin window
            ax.set_box_aspect(1)

            ax.hist(np.array(AQ[iv]), bins=10)
            
            # set figure parameters
            # ax.set_title('{}'.format(iv))
            ax.set_ylabel('# participants')
            ax.set_xlabel(iv)
            
            counter = counter + 1
                    
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_AQ_histogram.pdf'.format(self.exp)))
        print('success: plot_AQ_histogram')
        
        
    def plot_phasic_pupil_unsigned_pe(self,):
        """Plot the phasic pupil across frequency conditions.
        
        Notes
        -----
        GROUP LEVEL DATA
        x-axis is frequency conditions.
        """        
        dvs = ['pupil_feed_locked_t1', 'pupil_baseline_feed_locked']
        ylabels = ['Pupil response\n(% signal change)', 'Pupil response\n(% signal change)', ]
        factor = 'frequency'
        xlabel = 'Letter-color frequency'
        xticklabels = ['20%','40%','80%'] 
        
        xind = np.arange(len(xticklabels))
                
        for dvi,pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(111)
            
            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder, '{}_{}_{}.csv'.format(self.exp, factor, pupil_dv)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean', 'std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'], np.sqrt(len(self.subjects)))
            print(GROUP)
                        
            # plot
            ax.bar(xind, np.array(GROUP['mean']), yerr=np.array(GROUP['sem']),  capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
            
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.2) # marker, line, black

            # set figure parameters
            ax.set_title('{}'.format(pupil_dv))                
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)

            sns.despine(offset=10, trim=True)
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder, '{}_frequency_{}.pdf'.format(self.exp, pupil_dv)))
        print('success: plot_phasic_pupil_unsigned_pe')
        
        
        