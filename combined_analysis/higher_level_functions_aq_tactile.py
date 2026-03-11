#!/usr/bin/env python
# encoding: utf-8
"""
================================================
TACTILE DATA SET
A detail-oriented approach: Autistic traits scale with prediction errors during associative learning

Python code O.Colizoli 2025 (olympia.colizoli@donders.ru.nl)
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
    

    def cluster_sig_bar_1samp(self, array, x, yloc, color, ax, threshold=0.05, nrand=5000, cluster_correct=True):
        """Add permutation-based cluster-correction bar on time series plot.
        
        Parameters
        ----------
        array : array
            The data in matrix of format: subject x timepoints

        x : array
            x-axis of plot

        yloc : int
            Location on y-axis to draw bar

        color : string
            Color of bar

        ax : matplotlib.axes._subplots.AxesSubplot
            The subplot handle to plot in

        threshold : float
            Alpha value for p-value significance (default 0.05)

        nrand : int 
            Number of permutations (default 5000)

        cluster_correct : bool 
            Perform cluster-based multiple comparison correction if True (default True).
        """
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
        
    
    def compute_aq_score(self, aq_test, aq_scoring):
        """Compute the AQ score based on both the traditional and the 50-200 scoring versions.
        Scores based on the Hoekstra et al., 2008 version in which item #1 is corrected as compared with Baron Cohen et al., 2001.
        
        Parameters
        ----------
        aq_test : dataframe
            The responses to the individual AQ questionnaire items.
        
        aq_scoring : dataframe
            The file indicating how to reverse items and put items in each subscale.
        """
                
        aq_scoring = aq_scoring.loc[:, ~aq_scoring.columns.str.contains('^Unnamed')] # remove all unnamed columns
        aq_test = aq_test.loc[:, ~aq_test.columns.str.contains('^Unnamed')] # remove all unnamed columns
        subjects = aq_test['subject']
        questions = np.array(aq_scoring['question'])
        
        # Recode response options to numerical values: 
        # 1 = Definitely disagree (Geheel mee oneens)
        # 2 = Slightly disagree (Enigszins mee oneens)
        # 3 = Slightly agree (Enigszins mee eens)
        # 4 = Definitely agree (Geheel mee eens)
        
        survey_mapping = {
            'Definitely disagree': 4,
            'Slightly disagree': 3,
            'Slightly agree': 2,
            'Definitely agree': 1,
            'Geheel mee oneens': 4,     # Definitely disagree
            'Enigszins mee oneens': 3,  # Slightly disagree
            'Enigszins mee eens': 2,    # Slightly agree
            'Geheel mee eens': 1        # Definitely agree
        }
        aq_test = aq_test[questions].apply(lambda col: col.map(survey_mapping))

        # fix reverse scoring
        items = aq_test.loc[:,questions] # get 50 questionnaire items only
        mask = np.array(aq_scoring['hoekstra_reverse']==1)
        new_items = 5 - items.loc[:, mask] # only the reverse items
        cols=new_items.columns.values
        aq_test[cols] = new_items[cols]

        # subscales 
        items = aq_test.loc[:,questions]
        # social
        mask = np.array(aq_scoring['subscale']=='social')
        aq_test['social'] = np.sum(items.loc[:, mask], axis=1)
        # attention switching
        mask = np.array(aq_scoring['subscale']=='attention')
        aq_test['attention'] = np.sum(items.loc[:, mask], axis=1)
        # communication
        mask = np.array(aq_scoring['subscale']=='communication')
        aq_test['communication'] = np.sum(items.loc[:, mask], axis=1)
        # imagination/imagination
        mask = np.array(aq_scoring['subscale']=='imagination')
        aq_test['imagination'] = np.sum(items.loc[:, mask], axis=1)
        # detail
        mask = np.array(aq_scoring['subscale']=='detail')
        aq_test['detail'] = np.sum(items.loc[:, mask], axis=1)
        
        # aq score (out of 200)
        aq_test['aq_score'] = np.sum(aq_test[['social', 'attention', 'communication', 'imagination', 'detail']], axis=1)

        # COMPUTE ORIGINAL AQ SCORE OUT OF 50
        # Define the mapping dictionary
        # Keys are the original values, and values are the new ones
        response_mapping = {
            1: 0,
            2: 0,
            3: 1,
            4: 1
        }

        # Apply the mapping to the selected columns and store them in a new DataFrame
        recoded_df = aq_test[questions].apply(lambda col: col.map(response_mapping))

        # Calculate the sum across the recoded columns (axis=1 for rows)
        aq_test['aq_50'] = recoded_df.sum(axis=1)
        
        # sanity check
        correlation, p_value = stats.pearsonr(aq_test['aq_50'], aq_test['aq_score'])
        # Print the results
        print(f"The AQ50 vs. AQ200 Pearson correlation coefficient is: {correlation}")
        print(f"The corresponding p-value is: {p_value}")
        
        aq_test['subject'] = subjects
        aq_test.to_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_format='%.16f')
        
        print('success: compute_aq_score')
        
    
    def compute_blink_percentages(self, thresh):
        """Marks trials to be excluded based on blinks as a percentage of the total trial (threshold).
        
        Notes
        -----
        this_df['blinks_ratio'] = blinks_per_trial / (self.sample_rate*(pupil_step_lim[1]-pupil_step_lim[0])) # fraction of whole trial
        If threshold = 0.25, then all trials with more than a quarter of interpolated data will be excluded from further analysis.
        """
        
        for s,subj in enumerate(self.subjects):
            # loop through each type of event to lock events to...
            for t,time_locked in enumerate(self.time_locked):
                
                this_blinks = pd.read_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_trials_blinks.csv'.format(subj, self.exp, time_locked)))
                this_blinks = this_blinks.loc[:, ~this_blinks.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
                this_blinks['blinks_exclude'] = this_blinks['blinks_ratio'] > thresh
                
                excluded_trials = this_blinks[this_blinks['blinks_exclude']].index.tolist()
                this_blinks.to_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_trials_blinks.csv'.format(subj, self.exp, time_locked)))
                
                print("{} {} Trials to exclude (>{} interpolated): {}".format(time_locked, subj, thresh, len(excluded_trials)))
 
        print('success: compute_blink_percentages')
        
        
    def higherlevel_get_phasics(self,):
        """Computes phasic pupil (evoked average) in selected time window per trial and add phasics to behavioral data frame. 
        
        Notes
        -----
        Overwrites original log file (this_log).
        """
        for s,subj in enumerate(self.subjects):
            this_log = os.path.join(self.project_directory, subj, '{}_task-touch_prediction2_events.csv'.format(subj)) # derivatives folder
            B = pd.read_csv(this_log, float_precision='high') # behavioral file
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
                    P = pd.read_csv(os.path.join(self.project_directory,subj, '{}_{}_{}_evoked_basecorr.csv'.format(subj, self.exp, time_locked)), float_precision='high') 
                    P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    P = np.array(P)
                    
                    # get baselines for cleaning phasics
                    if 'response' in time_locked:
                        baseline_time_locked = 'cue_locked'
                    else:
                        baseline_time_locked = time_locked
                    # open baseline pupil to add to dataframes as well
                    this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_baselines.csv'.format(subj, self.exp, baseline_time_locked)), float_precision='high')
                    this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B['pupil_baseline_{}'.format(baseline_time_locked)] = np.array(this_baseline)

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
                    
                    # remove baseline and RT linear regression, save residuals as '_clean'
                    Y = B['pupil_{}_t{}'.format(time_locked, twi+1)]
                    X = B[['pupil_baseline_{}'.format(baseline_time_locked), 'RT']]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X, missing='drop').fit()
                    B['pupil_{}_t{}_clean'.format(time_locked,twi+1)] = model.resid

                    #######################
                    B = B.loc[:, ~B.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    B.to_csv(this_log, float_format='%.16f')
                    print('subject {}, {} phasic pupil extracted {}'.format(subj,time_locked, pupil_time_of_interest))
        print('success: higherlevel_get_phasics')


    def create_subjects_dataframe(self, exclude_interp=0):
        """Combine behavior and phasic pupil dataframes of all subjects into a single large dataframe, drop all trials after flip.
        
        Parameters
        ----------
        exclude_inter : boolean (default = 0)
            Exclude the trials the have too much interpolate data (1) or not (0).
        
        Notes
        -----
        Flag missing trials from concantenated dataframe.
        Output in dataframe folder: task-experiment_name_subjects.csv
        """
        DF = pd.DataFrame()
        
        # loop through subjects, get behavioral log files
        for s,subj in enumerate(self.subjects):
            
            this_data = pd.read_csv(os.path.join(self.project_directory, subj, '{}_task-touch_prediction2_events.csv'.format(subj)), float_precision='high')
            this_data = this_data.loc[:, ~this_data.columns.str.contains('^Unnamed')] # remove all unnamed columns
            
            # open baseline pupil to add to dataframes as well
            this_baseline = pd.read_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_baselines.csv'.format(subj, self.exp, 'feed_locked')), float_precision='high')
            this_baseline = this_baseline.loc[:, ~this_baseline.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['pupil_baseline_feed_locked'] = np.array(this_baseline)
            
            # open blinks dataframes as well
            this_blinks = pd.read_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_trials_blinks.csv'.format(subj, self.exp, 'feed_locked')), float_precision='high')
            this_blinks = this_blinks.loc[:, ~this_blinks.columns.str.contains('^Unnamed')] # remove all unnamed columns
            this_data['blinks_exclude'] = np.array(this_blinks['blinks_exclude'])
            
            ###############################
            # flag missing trials
            this_data['missing'] = this_data['response']=='missing'
                        
            ###############################            
            # concatenate all subjects
            DF = pd.concat([DF,this_data],axis=0)
            
        # drop all trials after flip moment (keep first 5 blocks)
        DF = DF[DF['trial_num']<=105]
        
        ### mark all trials to exclude 
        DF['drop_trial'] = DF['missing']
        DF['drop_trial'] = (DF['drop_trial'] > 0).astype(int)
        # drop trials based on too much interpolated data?
        if exclude_interp: 
            DF['drop_trial'] = DF['drop_trial'] + DF['blinks_exclude']
            DF['drop_trial'] = (DF['drop_trial'] > 0).astype(int)
       
        # how many trials excluded due to blinks? 
        print('Total blink trials excluded = {}%'.format(np.true_divide(np.sum(DF['blinks_exclude']),DF.shape[0])*100))
        # per subject?
        blinks_excluded = DF.groupby(['subject',])['blinks_exclude'].sum()
        blinks_excluded.to_csv(os.path.join(self.dataframe_folder,'{}_blinks_excluded_counts_subject.csv'.format(self.exp)), float_format='%.16f')
        
        # count missing
        M = DF[DF['response']!='missing'] 
        missing = M.groupby(['subject','response'])['response'].count()
        missing.to_csv(os.path.join(self.dataframe_folder,'{}_behavior_missing.csv'.format(self.exp)), float_format='%.16f')
        
        ### print how many outliers
        print('Missing = {}%'.format(np.true_divide(np.sum(DF['missing']),DF.shape[0])*100))
        print('Dropped trials = {}%'.format(np.true_divide(np.sum(DF['drop_trial']),DF.shape[0])*100))

        #####################
        # save whole dataframe with all subjects
        DF.to_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_format='%.16f')
        #####################
        print('success: higherlevel_dataframe')
        

    def code_stimuli(self, ):
        """Add a new column in the subjects dataframe to give each letter-color pair a unique identifier.
        
        Notes
        -----
        3 fingers ^ 2 touches -> 9 different letter-color pair combinations.
        High frequency mappings were: 
        ring-index (1-3) 
        middle-middle (2-2)
        index-ring (3-1)
        
        New column name is "touch_pair"
        """
        fn_in = os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp))
        df_in = pd.read_csv(fn_in, float_precision='high')
        
        # make new column to give each touch1-touch2 combination a unique identifier (0 - 8)        
        mapping = [
            (df_in['touch1'] == 1) & (df_in['touch2'] == 1), # 0 LOW NONE
            (df_in['touch1'] == 1) & (df_in['touch2'] == 2), # 1 LOW SHORT
            (df_in['touch1'] == 1) & (df_in['touch2'] == 3), # 2 HIGH LONG
            
            (df_in['touch1'] == 2) & (df_in['touch2'] == 1), # 3 LOW SHORT
            (df_in['touch1'] == 2) & (df_in['touch2'] == 2), # 4 HIGH NONE
            (df_in['touch1'] == 2) & (df_in['touch2'] == 3), # 5 LOW SHORT
            
            (df_in['touch1'] == 3) & (df_in['touch2'] == 1), # 6 HIGH LONG
            (df_in['touch1'] == 3) & (df_in['touch2'] == 2), # 7 LOW SHORT
            (df_in['touch1'] == 3) & (df_in['touch2'] == 3), # 8 LOW NONE
            ]
        
        elements = np.arange(9) # also elements is the same as priors (start with 0 so they can be indexed by element)
        df_in['touch_pair'] = np.select(mapping, elements)
        
        # add frequency conditions
        elements = ['low', 'low', 'high', 'low', 'high', 'low' ,'high', 'low', 'low']
        df_in['frequency'] = np.select(mapping, elements, default='NaN')
        
        # add finger distance
        elements = ['none', 'short', 'long', 'short', 'none', 'short', 'long', 'short', 'none']
        df_in['finger_distance'] = np.select(mapping, elements, default='NaN')
        
        df_in.to_csv(fn_in, float_format='%.16f') # save with new columns
        print('success: code_stimuli')   
    
    
    def calculate_actual_frequencies(self):
        """Calculate the actual frequencies of the touch-pairs presented during the task.

        Notes
        -----
            The lists per finger were drawn randomly based on a uniform distribution.
        """
        
        ntrials = 105 # per participant
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)))
        DF['for_counts'] = np.repeat(1,len(DF)) # to count something
        
        counts_pairs = pd.DataFrame(DF.groupby(['subject', 'touch1', 'touch2'])['for_counts'].count())
        counts_touch1 = pd.DataFrame(DF.groupby(['subject', 'touch1'])['for_counts'].count())
        
        finger_trials = np.unique(counts_touch1['for_counts'])
        
        # calculate as percentage per finger
        counts_pairs['actual_frequency'] = np.true_divide(counts_pairs['for_counts'],finger_trials)*100
        counts_pairs.to_csv(os.path.join(self.dataframe_folder, '{}_actual_frequency_pairs.csv'.format(self.exp)))
        
        # do again for low-high frequency conditions
        counts_frequency = pd.DataFrame(DF.groupby(['subject', 'frequency'])['for_counts'].count())
        counts_frequency['actual_frequency'] = np.true_divide(counts_frequency['for_counts'], ntrials)*100
        counts_frequency.to_csv(os.path.join(self.dataframe_folder, '{}_actual_frequency_conditions.csv'.format(self.exp)))
        
        print('success: calculate_actual_frequencies')
        

    def dataframe_evoked_pupil_higher(self):
        """Compute evoked pupil responses.
        
        Notes
        -----
        Split by conditions of interest. Save as higher level dataframe per condition of interest. 
        Evoked dataframes need to be combined with behavioral data frame, looping through subjects. 
        DROP EXTRA BLOCKS.
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)), float_precision='high')
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
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory,subj, '{}_{}_{}_evoked_basecorr.csv'.format(subj, self.exp, time_locked)), float_precision='high'))
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    
                    #############################
                    # KEEP FIRST 5 BLOCKS ONLY from evoked DF
                    SPUPIL = SPUPIL.iloc[:105]
                    
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
                COND.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_higher')
    
    
    def dataframe_evoked_correlation_AQ(self, ):
        """Timeseries individual differences (Spearman rank) correlation between pupil and AQ score for:
        all trials, correct vs. error, frequency (20-80%), interaction term (20-80%)

        Notes
        -----
        Omissions dropped in dataframe_evoked_pupil_higher()
        Output correlation coefficients per time point in dataframe folder.
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')    
        iv = 'aq_score'
        aq_scores = np.array(AQ[iv])
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        df_out = pd.DataFrame() # timepoints x condition
        
        for t,time_locked in enumerate(self.time_locked):
            
            for cond in ['subject', 'correct', 'frequency', 'correct-frequency']:
            
                DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_{}.csv'.format(self.exp, time_locked, cond)), float_precision='high')
                DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
                # merge AQ and pupil dataframes on subject to make sure have only data containing AQ scores
                this_pupil = DF.loc[DF['subject'].isin(AQ['subject'])]
                
                # get columns of pupil sample points only
                kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*self.sample_rate)
                evoked_cols = np.char.mod('%d', np.arange(kernel)) 
                
                if cond == 'correct':
                    # error-correct
                    error = this_pupil[this_pupil['correct']==0].copy()
                    correct = this_pupil[this_pupil['correct']==1].copy()
                    error.reset_index(inplace=True)
                    correct.reset_index(inplace=True)
                    this_pupil = np.subtract(error, correct)
                                    
                elif 'frequency' in cond:
                    
                    # need to recode to make numeric for subtraction
                    mapping = [this_pupil['frequency']=='high', this_pupil['frequency']=='low']
                    elements = [80, 20] # also elements is the same as priors (start with 0 so they can be indexed by element)
                    this_pupil['frequency'] = np.select(mapping, elements)
                                        
                    if cond == 'frequency':
                        # Low-High%
                        low = this_pupil[this_pupil['frequency']==20].copy()
                        high = this_pupil[this_pupil['frequency']==80].copy()
                        low.reset_index(inplace=True)
                        high.reset_index(inplace=True)
                        try: # conservative analysis
                            this_pupil = np.subtract(low, high)
                        except:
                            this_pupil = low - high
                                            
                    elif cond == 'correct-frequency':
                        # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                        easy_error   = this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==0)].copy()
                        easy_correct = this_pupil[(this_pupil['frequency']==80) & (this_pupil['correct']==1)].copy() 
                        hard_error   = this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==0)].copy() 
                        hard_correct = this_pupil[(this_pupil['frequency']==20) & (this_pupil['correct']==1)].copy() 
                        easy_error.reset_index(inplace=True)
                        easy_correct.reset_index(inplace=True)
                        hard_error.reset_index(inplace=True)
                        hard_correct.reset_index(inplace=True)
                        # n = 10 participants missing trials for "hard_correct" 
                        # Keep only indices where not NaN
                        # hard_error_filtered = hard_error[hard_error['subject'].isin(hard_correct['subject'])]
                        # easy_error_filtered = easy_error[easy_error['subject'].isin(hard_correct['subject'])]
                        # easy_correct_filtered = easy_correct[easy_correct['subject'].isin(hard_correct['subject'])]
                        #
                        # term1 = np.subtract(easy_error_filtered, easy_correct_filtered)
                        # term2 = np.subtract(hard_error_filtered, hard_correct)
                        # this_pupil = np.subtract(term1, term2)
                        
                        term1 = easy_error - easy_correct
                        term2 = hard_error - hard_correct
                        this_pupil = term1 - term2

                                 
                # loop timepoints, regress
                save_timepoint_r = []
                save_timepoint_p = []
                
                for col in evoked_cols:
                    Y = this_pupil[col] # pupil
                    X = aq_scores # iv
                    
                    # Keep only indices where both are not NaN (conservative analysis)
                    try:
                        mask = ~np.isnan(X) & ~np.isnan(Y)
                        X = X[mask]
                        Y = Y[mask]
                    except:
                        shell()
                        
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
            df_out.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_correlation_{}.csv'.format(self.exp, time_locked, iv)), float_format='%.16f')
        print('success: dataframe_evoked_correlation_AQ')
        

    def plot_evoked_correlation_AQ(self):
        """Plot partial correlation between pupil response and model estimates.
        
        Notes
        -----
        Always feed_locked pupil response.
        Partial correlations are done for all trials as well as for correct and error trials separately.
        """
        
        sample_rate = self.sample_rate 
        
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        iv = 'aq_score'
    
        colors = ['black', 'red', 'blue', 'orange'] 
        alphas = [1]
        labels = ['All Trials' , 'Error-Correct', 'Low-High%', 'Accuracy*Frequency']
        
        
        time_locked = 'feed_locked'
        t = 0
        CORR = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_{}.csv'.format(self.exp, time_locked, iv)), float_precision='high')
        
        #######################
        # FEEDBACK PLOT R FOR EACH PUPIl COND
        #######################
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111)
        factor = 'subject'
        kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*sample_rate) # length of evoked responses
        # determine time points x-axis given sample rate
        event_onset = int(abs(self.pupil_step_lim[t][0]*sample_rate))
        end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*sample_rate)
        mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                        
        for i,cond in enumerate(['subject', 'correct', 'frequency', 'correct-frequency']):
            
            # Compute means, sems across group
            TS = np.array(CORR['{}_r'.format(cond)])
            pvals = np.array(CORR['{}_pval'.format(cond)])
                        
            ax.plot(pd.Series(range(TS.shape[-1])), TS, color=colors[i], label=labels[i])
                
            # stats        
            self.timeseries_fwer_correction(pvals=pvals, xind=pd.Series(range(pvals.shape[-1])), alpha=0.07, color=colors[i], yloc=i, ax=ax)
            
        # set figure parameters
        ax.axvline(int(abs(self.pupil_step_lim[t][0]*sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
        ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
        # Shade all time windows of interest in grey, will be different for events
        for twi in self.pupil_time_of_interest[t]:
            tw_begin = int(event_onset + (twi[0]*sample_rate))
            tw_end = int(event_onset + (twi[1]*sample_rate))
            ax.axvspan(tw_begin, tw_end, facecolor='k', alpha=0.1)

        xticks = [event_onset, event_onset+(0.5*sample_rate*1), event_onset+(0.5*sample_rate*2), event_onset+(0.5*sample_rate*3), event_onset+(0.5*sample_rate*4), event_onset+(0.5*sample_rate*5), event_onset+(0.5*sample_rate*6), event_onset+(0.5*sample_rate*7)]
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
        
                
    def average_conditions(self, ):
        """Average the phasic pupil per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='high')
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
            DFANOVA =  DFOUT.unstack(['frequency', 'correct', 'block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## BLOCK x CORRECT ########
            '''
            # MEANS subject x block x correct
            DFOUT = DF.groupby(['subject', 'block', 'correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_block-correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct','block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        for pupil_dv in ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']: # mean accuracy
        
            '''
            ######## BLOCK x FREQUENCY ########
            '''
            DFOUT = DF.groupby(['subject','block','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_block-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency','block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## BLOCK x FREQUENCY ########
            '''
            DFOUT = DF.groupby(['subject','block'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_block_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['block']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_block_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_conditions')
        
        
    def regression_pupil_AQ(self, ):
        """Multiple regression of AQ components (IVs) onto average pupil response in early time window. Use 20%-80% pupil response as DV.

        Notes
        -----
        print(results.summary())
        Quantities of interest can be extracted directly from the fitted model. Type dir(results) for a full list. Here are some examples:
        print('Parameters: ', results.params)
        print('R2: ', results.rsquared)
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')        
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        ivs = ['social', 'attention', 'communication', 'imagination', 'detail']
        cond = 'frequency'
        dv = 'pupil_feed_locked_t1'
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        df_out = pd.DataFrame() # timepoints x condition
                        
        for cond in ['frequency']:
            
            # task-letter_color_visual_decision_frequency_pupil_feed_locked_t1.csv
            this_condition = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_rmanova.csv'.format(self.exp, cond, dv)), float_precision = 'high')
            this_condition = this_condition.loc[:, ~this_condition.columns.str.contains('^Unnamed')] # remove all unnamed columns
            # merge pupil and AQ dataframes
            merged = pd.merge(this_condition, AQ, on='subject', how='inner')
            
            if cond == 'frequency': 
                # 20%-80%
                merged['pupil'] = np.array(merged['low']-merged['high'])

            # ordinary least squares regression
            Y = merged['pupil']
            X = merged[ivs]
            X = sm.add_constant(X)  
            # multicollinearity check
            vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            print('VIF multicollinearity check results:')
            print(vif)
                      
            model = sm.OLS(Y,X, missing='drop') 
            results = model.fit()
            results.params
            print(results.summary())

            # save model results in output file
            out_path = os.path.join(self.jasp_folder, '{}_{}_{}_OLS_summary.csv'.format(self.exp, cond, dv))
            text_file = open(out_path, "w")
            text_file.write(results.summary().as_text())
            text_file.close()
                    
        print('success: regression_pupil_AQ')
        
    
    def correlation_AQ(self,):
        """Calculate the Spearman rank correlations between AQ score (and sub-scores) with other DVs of interest. Plot data.

        Notes
        -----
        ivs = ['aq_score', 'social', 'attention', 'communication', 'imagination', 'detail']
        dvs = ['pupil_feed_locked_t1', 'pupil_baseline_feed_locked', 'RT', 'correct']
        conditions = ['subject', 'correct', 'frequency', 'correct-frequency']
        freqs = ['low', 'high'] # low, high to contrast
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')    
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        ivs = ['aq_score', 'social', 'attention', 'communication', 'imagination', 'detail']
        # ivs = ['aq_score', ]
        dvs = ['pupil_feed_locked_t1', 'pupil_baseline_feed_locked', 'RT', 'correct']
        conditions = ['subject', 'correct', 'frequency', 'correct-frequency', 'low_frequency', 'high_frequency']
        freqs = ['low', 'high'] # low, high to contrast
        
        for iv in ivs:
            # new figure for every IV and block
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
                        
                        if cond == 'low_frequency' or cond == 'high_frequency':
                            P = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_rmanova.csv'.format(self.exp, 'frequency', dv)))
                        else:
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
                            
                        elif cond == 'low_frequency':            
                            # low frequency condition only
                            M['main_effect_{}'.format(cond)] = (M[freqs[0]])
                            ax.set_ylabel('{} ({}%)'.format(dv, freqs[0]))
                            
                        elif cond == 'high_frequency':            
                            # high frequency condition only
                            M['main_effect_{}'.format(cond)] = (M[freqs[1]])
                            ax.set_ylabel('{} ({}%)'.format(dv, freqs[1]))
                        
                        elif cond == 'correct':
                            # frequency effect
                            M['main_effect_{}'.format(cond)] = (M['0']-M['1'])
                            ax.set_ylabel('{} (Error-Correct)'.format(dv))
                        
                        elif cond == 'correct-frequency':
                            # interaction effect (Easy Error- Easy Correct) - (Hard Error - Hard Correct)
                            M['main_effect_{}'.format(cond)] = (M["('{}', 0)".format(freqs[1])]-M["('{}', 1)".format(freqs[1])])-(M["('{}', 0)".format(freqs[0])]-M["('{}', 1)".format(freqs[0])])
                            ax.set_ylabel('{} (interaction)'.format(dv))
                        
                        # correlation
                        x = np.array(M[iv])
                        y = np.array(M['main_effect_{}'.format(cond)])  
                        
                        # Keep only indices where both are not NaN (conservative analysis)
                        mask = ~np.isnan(x) & ~np.isnan(y)
                        x = x[mask]
                        y = y[mask]
                                 
                        r,pval = stats.spearmanr(x,y)
                        
                        # fit regression line
                        ax.plot(x, y, 'o', markersize=3, color='purple') # marker, line, black
                        m, b = np.polyfit(x, y, 1)
                        ax.plot(x, m*x+b, color='black',alpha=.5)
                        
                        # set figure parameters
                        ax.set_title('rs = {}, p = {}'.format(np.round(r,2),np.round(pval,3)))
                        ax.set_xlabel(iv)
                        if 'aq' in iv:
                            ax.set_xticks([75, 100, 125, 150])
                        else:
                            ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
                        if 'aq' in iv and dv == 'RT' and cond == 'correct-frequency':
                            ax.set_ylim([-1, 2])
                        else:
                            ax.set_ylim([-2, 4])
                        ax.set_xlabel(iv)
                        
                    counter = counter + 1
                    
            plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_correlation_{}.pdf'.format(self.exp, iv)))
        print('success: correlation_AQ')
        

    def plot_AQ_histogram(self,):
        """Plot a histogram of the AQ score distribution (and sub-scores).
        """
        y_min = 0 
        y_max = 12
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        ivs = ['aq_score', 'social', 'attention', 'communication', 'imagination', 'detail']
        colors = ['grey', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
        
        # new figure for every IV
        fig = plt.figure(figsize=(2,2*len(ivs)))
        counter = 1 # subplot counter
        
        for idx,iv in enumerate(ivs):
            ax = fig.add_subplot(len(ivs), 1, counter) # 1 subplot per bin window
            ax.set_box_aspect(1)
            
            if not 'aq' in iv:
                ax.hist(np.array(AQ[iv]), color=colors[idx], bins=12, range=(10,40))
                ax.set_ylim(y_min, y_max)
                # ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
                ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
            else:
                ax.hist(np.array(AQ[iv]), color=colors[idx], bins=10)
                ax.set_xticks([75, 100, 125, 150])            
                
            counter = counter + 1
                    
            # set figure parameters
            # ax.set_title('{}'.format(iv))
            ax.set_ylabel('# participants')
            ax.set_xlabel(iv)
    
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
        xlabel = 'Touch-pair frequency'
        xticklabels = ['Low','High'] # CAREFUL! high is first column in dataframe, must take into account in plotting
        
        xind = np.arange(len(xticklabels))
                
        for dvi,pupil_dv in enumerate(dvs):
            
            fig = plt.figure(figsize=(2, 2))
            ax = fig.add_subplot(111)
            
            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder, '{}_{}_{}.csv'.format(self.exp, factor, pupil_dv)), float_precision='high')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average per BIN WINDOW
            GROUP = pd.DataFrame(DFIN.groupby(factor)[pupil_dv].agg(['mean', 'std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'], np.sqrt(len(self.subjects)))
            print(GROUP)
                        
            # plot (flip xind so that low comes first in dataframe!)
            ax.bar(xind[::-1], np.array(GROUP['mean']), yerr=np.array(GROUP['sem']),  capsize=3, color=(0,0,0,0), edgecolor='black', ecolor='black')
            
            # individual points, repeated measures connected with lines
            DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            DFIN = DFIN.unstack(factor)
            for s in np.array(DFIN):
                # plot (flip xind so that low comes first in dataframe!)
                ax.plot(xind[::-1], s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.2) # marker, line, black

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


    def plot_AQ_covariance(self, ):
        """Plot the covariance matrix of the AQ sub-scores.
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')
        AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        ivs = ['social', 'attention', 'communication', 'imagination', 'detail']
        this_df = AQ[ivs].copy()
                
        # new figure for every IV
        fig = plt.figure(figsize=(4,4))        
        ax = fig.add_subplot(111) # 1 subplot per bin window
        ax.set_box_aspect(1)

        ax = sns.heatmap(this_df.corr(method='spearman'), vmin=0, vmax=1, annot=True)
        
        # set figure parameters
        # ax.set_title('{}'.format(iv))
        # ax.set_ylabel('# participants')
        # ax.set_xlabel(iv)
                        
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_AQ_correlation_matrix.pdf'.format(self.exp)))
        print('success: plot_AQ_covariance')
    
    
    def plot_behav_blocks(self):
        """Plot the accuracy and RT per block.
        """
        
        dvs = ['correct', 'RT', 'pupil_feed_locked_t1', 'pupil_baseline_feed_locked']
        blocks = [1,2,3,4,5]
        chance = 0.33
        
        # --- Plot ---
        fig = plt.figure(figsize=(2*len(dvs),2))
        counter = 1
        
        for dv in dvs:
            ax = fig.add_subplot(1,len(dvs),counter) # 1 subplot per bin window
            
            df = pd.read_csv(os.path.join(self.jasp_folder,'{}_block_{}_rmanova.csv'.format(self.exp, dv)), float_precision = 'high')
            
            df = df[[str(b) for b in blocks]] # get only block data
            means = df.mean()
            sems = df.sem()   # SEM = SD / sqrt(n)

            ax.errorbar(blocks, means, yerr=sems, fmt='-o', color='black',
                         ecolor='black', elinewidth=1, capsize=4, capthick=1)
                         
            ax.set_xticks(blocks)
            ax.set_xlabel('Block')
            ax.set_ylabel(dv)
            if dv == 'correct':
                ax.set_ylim([0.60, 1.0])
            if dv == 'RT':
                ax.set_ylim([0.60, 1.0])
            if dv == 'pupil_feed_locked_t1':
                ax.set_ylim([0.75, 3.0])    
            if dv == 'pupil_baseline_feed_locked':
                ax.set_ylim([-2, 1.0])    
            
            counter = counter + 1
            
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_behav_blocks.pdf'.format(self.exp)))
        print('success: plot_behav_blocks')
        
        
    # def correlation_AQ_blocks(self,):
    #     """PER BLOCK calculate the Spearman rank correlations between AQ score and unsigned prediction errors. Plot data.
    #
    #     Notes
    #     -----
    #     ivs = ['aq_score']
    #     dvs = ['pupil_feed_locked_t1']
    #     conditions = ['frequency']
    #     freqs = ['20', '80'] # low, high to contrast
    #     """
    #
    #     AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')
    #     AQ = AQ.loc[:, ~AQ.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #     iv = 'aq_score'
    #     dv = 'pupil_feed_locked_t1'
    #     cond = 'block-frequency'
    #     freqs = ['low', 'high'] # low, high to contrast
    #     blocks = [1,2,3,4,5]
    #
    #     # new figure for every IV, block
    #     fig = plt.figure(figsize=(2,2*len(blocks)))
    #     counter = 1 # subplot counter
    #
    #     P = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_rmanova.csv'.format(self.exp, cond, dv)), float_precision = 'high')
    #     P = P.loc[:, ~P.columns.str.contains('^Unnamed')] # remove all unnamed columns
    #
    #     # make sure subjects are aligned
    #     M = AQ.merge(P, how='inner', on=['subject'])
    #
    #     # SPLIT BY BLOCK
    #     for block in blocks:
    #
    #         ax = fig.add_subplot(len(blocks), 1, counter) # 1 subplot per bin window
    #         ax.set_box_aspect(1)
    #
    #         M['main_effect_{}_block{}'.format(cond, block)] = (M["('low', {})".format(block)]-M["('high', {})".format(block)])
    #         ax.set_ylabel('{} ({}%-{}%)'.format(dv, freqs[0], freqs[1]))
    #
    #         # correlation
    #         x = np.array(M[iv])
    #         y = np.array(M['main_effect_{}_block{}'.format(cond, block)])
    #         r,pval = stats.spearmanr(x,y)
    #
    #         # fit regression line
    #         ax.plot(x, y, 'o', markersize=3, color='purple') # marker, line, black
    #         m, b = np.polyfit(x, y, 1)
    #         ax.plot(x, m*x+b, color='black',alpha=.5)
    #
    #         # set figure parameters
    #         ax.set_title('block={}, rs = {}, p = {}'.format(block, np.round(r,2),np.round(pval,3)))
    #         ax.set_xticks([75, 100, 125, 150])
    #         ax.set_ylim([-2, 4])
    #         ax.set_xlabel(iv)
    #
    #         counter = counter + 1
    #
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(self.figure_folder,'{}_correlation_{}_blocks.pdf'.format(self.exp, iv)))
    #     print('success: correlation_AQ_blocks')
        
        
    def regression_pupil_AQ_blocks(self, ):
        """BY BLOCK Multiple regression of AQ components (IVs) onto average pupil response in early time window. Use 20%-80% pupil response as DV.

        Notes
        -----
        print(results.summary())
        Quantities of interest can be extracted directly from the fitted model. Type dir(results) for a full list. Here are some examples:
        print('Parameters: ', results.params)
        print('R2: ', results.rsquared)
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')     
        ivs = ['social', 'attention','communication','imagination','detail']
        cond = 'block-frequency'
        dv = 'pupil_feed_locked_t1'
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        df_out = pd.DataFrame() # timepoints x condition
        df_conf_out = pd.DataFrame() # save confidence intervals for plotting
        
        # task-letter_color_visual_decision_frequency_pupil_feed_locked_t1.csv
        this_condition = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_rmanova.csv'.format(self.exp, cond, dv)), float_precision = 'high')
        this_condition = this_condition.loc[:, ~this_condition.columns.str.contains('^Unnamed')] # remove all unnamed columns
        # merge pupil and AQ dataframes
        merged = pd.merge(this_condition, AQ, on='subject', how='inner')
        
        for block in [1,2,3,4,5]:

            if cond == 'block-frequency': 
                # 20%-80%
                merged['pupil'] = np.array(merged["('low', {})".format(block)]-merged["('high', {})".format(block)]) # blocks are different columns

            # ordinary least squares regression
            Y = merged['pupil']
            X = merged[ivs]
            X = sm.add_constant(X)
            
            model = sm.OLS(Y,X, missing='drop')
            results = model.fit()
            results.params
            print('BLOCK {}'.format(block))
            print(results.summary())
        
            # save model results in output file
            out_path = os.path.join(self.jasp_folder,'{}_{}_{}_block{}_OLS_summary.csv'.format(self.exp, cond, dv, block))

            text_file = open(out_path, "w")
            text_file.write(results.summary().as_text())
            text_file.close()
        
            # save confidence intervals, betas, and model fit for plotting
            save_conf_int = pd.DataFrame(results.conf_int())
            save_conf_int['block'] = np.repeat(block, len(save_conf_int))
            save_conf_int['beta'] = results.params # beta coefficients
            save_conf_int['bse'] = results.bse # standard error
            save_conf_int['tvalues'] = results.tvalues # tvalues beta coefficients
            save_conf_int['pvalues'] = results.pvalues # signficance of beta coefficients
            save_conf_int['fvalue'] = np.repeat(results.fvalue, len(save_conf_int)) # for model
            save_conf_int['f_pvalue'] = np.repeat(results.f_pvalue, len(save_conf_int)) # for model
            save_conf_int['rsquared'] = np.repeat(results.rsquared, len(save_conf_int)) # for model
            df_conf_out = pd.concat([df_conf_out, save_conf_int])
        
        df_conf_out.to_csv(os.path.join(self.jasp_folder,'{}_{}_{}_blocks_OLS_results.csv'.format(self.exp, cond, dv)))
        
        print('success: regression_pupil_AQ_blocks')        


    def plot_regression_pupil_AQ_blocks(self, ):
        """By block, plot pupil~AQ regression results.
        """
        cond = 'block-frequency'
        dv = 'pupil_feed_locked_t1'
        ivs = ['constant', 'social', 'attention', 'communication', 'imagination', 'detail']
        ylim = [-1.0, 1.0]
        
        df_results = pd.read_csv(os.path.join(self.jasp_folder,'{}_{}_{}_blocks_OLS_results.csv'.format(self.exp, cond, dv)))
        df_results = df_results.loc[:, ~df_results.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
        # new figure for every IV
        fig = plt.figure(figsize=(8,3))
        counter = 1 # subplot counter
        
        for block in [1,2,3,4,5]:
            ax = fig.add_subplot(1, 5, counter) # 1 subplot per bin window
            ax.set_box_aspect(1)
            
            try:
                this_block = df_results[df_results['block']==block].copy()
            
                x = np.arange(len(ivs))
                y = np.array(this_block['beta'])
                yerr = np.array(this_block['bse'])
                # drop constant before plotting
                x = x[1:]
                y = y[1:]
                yerr = yerr[1:]
            
                # flag significant betas with green markers
                sig_array = np.array(this_block['pvalues']<0.05)[1:]
                marker_size_array = 20*sig_array # size * boolean mask
                # plot bars and markers separately 
                ax.errorbar(x, y, yerr=yerr, marker='o', markersize=1, mfc='black', mec='black', ls="None", ecolor='grey', elinewidth=1, capsize=2, barsabove=False)
                ax.scatter(x, y, s=marker_size_array, marker='*', color='black')
            
                # model results
                fvalue = np.round(np.array(this_block['fvalue'])[0],2)
                f_pvalue = np.round(np.array(this_block['f_pvalue'])[0], 3)
                rsquared = np.round(np.array(this_block['rsquared'])[0], 2)
            
                # set figure parameters
                ax.set_title('Block {}'.format(block))
                ax.set_ylabel('Beta coefficient')
                ax.set_xlabel('F(5,41) = {}, p = {}, R2 = {}'.format(fvalue, f_pvalue, rsquared))
                ax.set_xticks(x)
                ax.set_xticklabels(ivs[1:], rotation=45, ha='right')
                ax.set_ylim(ylim)
            
                counter = counter + 1
                    
                plt.tight_layout()
            except:
                pass
        fig.savefig(os.path.join(self.figure_folder,'{}_{}_{}_OLS_results.pdf'.format(self.exp, cond, dv)))
        print('success: plot_regression_pupil_AQ_blocks')
        

    def idt_model(self, df, df_data_column, elements, priors, flat_prior=False):
        """Process Ideal Learner Model.
        
        Parameters
        ----------
        df : pandas dataframe
            The dataframe to apply the Ideal Learner Model to.
        
        df_data_column : str
            The name of the column that refers to the cue-target pairs for all trials in the experiment.
        
        elements : list
            The list of unique indentifiers for the cue-target pairs.
        
        priors : list
            The list of priors as probabilities.
        
        Returns
        -------
        [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]: list
            A list containing all model parameters (see notes).
            
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Priors are generated from the probabilities of the letter-color pair in the odd-ball task.
        
        Model Output Notes:
        model_e = trial sequence
        model_P = probabilities of all elements at each trial
        model_p = probability of current element at current trial
        model_I = surprise of all elements at each trial (i.e., complexity)
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_CH = cross-entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        data = np.array(df[df_data_column])
    
        # initialize output variables for current subject
        model_e = [] # trial sequence
        model_P = [] # probabilities of all elements
        model_p = [] # probability of current element 
        model_I = [] # surprise of all elements 
        model_i = [] # surprise of current element 
        model_H = [] # entropy at current trial
        model_CH = [] # cross-entropy at current trial
        model_D = []  # KL-divergence at current trial
    
        # loop trials
        for t in np.arange(df.shape[0]):
            vector = data[:t+1] #  trial number starts at 0, all the targets that have been seen so far
            
            model_e.append(vector[-1])  # element in current trial = last element in the vector
            
            # print(vector)
            if t < 1: # if it's the first trial, our expectations are based only on the prior (values)
                
                if not flat_prior:
                    # PRIORS BASED ON ODDBALL TASK
                    alpha1 = priors*len(elements) # np.sum(alpha) == len(elements), priors from odd-ball task
                    p1 = priors # priors based on odd-ball task, np.sum(priors) should equal 1
                    p = p1
                else:
                    # UNIFORM PRIOR DISTRIBUTION
                    alpha1 = np.ones(len(elements)) # np.sum(alpha) == len(elements), flat prior
                    p1 = alpha1 / len(elements) # probablity, i.e., np.sum(p1) == 1
                    p = p1
            
            # at every trial, we compute surprise based on the probability
            model_P.append(p)           # probability (all elements)
            model_p.append(p[vector[-1]]) # probability of current element
            # Surprise is defined by the negative log of the probability of the current trial given the previous trials.
            I = -np.log2(p)     # complexity of every event (each cue_target_pair is a potential event)
            i = I[vector[-1]]   # surprise of the current event (last element in vector)
            model_I.append(I)
            model_i.append(i)
            
            # EVERYTHING AFTER HERE IS CALCULATED INCLUDING CURRENT EVENT
            # Updated estimated probabilities (posterior)
            p = []
            for k in elements:
                # +1 because in the prior there is one element of the same type; +len(alpha) because in the prior there are #alpha elements
                # The influence of the prior should be sampled by a distribution or
                # set to a certain value based on Kidd et al. (2012, 2014)
                p.append((np.sum(vector == k) + alpha1[k]) / (len(vector) + len(alpha1)))       
            
            H = -np.sum(p * np.log2(p)) # entropy (note that np.log2(1/p) is equivalent to multiplying the whole sum by -1)
            model_H.append(H)   # entropy
            
            # once we have the updated probabilities, we can compute KL Divergence, Entropy and Cross-Entropy
            prevtrial = t-1
            if prevtrial < 0: # first trial
                D = np.sum(p * (np.log2(p / np.array(p1)))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            else:
                D = np.sum(p * (np.log2(p / np.array(model_P[prevtrial])))) # KL divergence, after vs. before, same direction as Poli et al. 2020
            
            CH = H + D # Cross-entropy
    
            model_CH.append(CH) # cross-entropy
            model_D.append(D)   # KL divergence
        
        return [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D]
        
        
    def information_theory_estimates(self, flat_prior=True):
        """Run subject loop on Ideal Learner Model and save model estimates.
        
        Parameters
        ----------
        flat_prior : boolean
            Use a uniform prior (True) or input priors (False); default True
        
        Notes
        -----
        Ideal Learner Model adapted from Poli, Mars, & Hunnius (2020).
        See also: https://github.com/FrancescPoli/eye_processing/blob/master/ITDmodel.m
        
        Model estimates that are saved in subject's dataframe:
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_D = KL-divergence at current trial
        """
        
        fn_in = os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp))
        
        if flat_prior:
            this_priors = []
        else:
            print('Need to load prior dataframe!')
            
        df_in = pd.read_csv(fn_in, float_precision='%.16f')
        df_in = df_in.loc[:, ~df_in.columns.str.contains('^Unnamed')]
        # sort by subjects then trial_counter in ascending order
        df_in.sort_values(by=['subject', 'trial_num'], ascending=True, inplace=True)
        
        df_out = pd.DataFrame()
        df_prob_out = pd.DataFrame() # last probabilities all elements saved
        
        elements = np.unique(df_in['touch_pair'])
        
        # loop subjects
        for s,subj in enumerate(self.subjects):
            
            this_subj = int(''.join(filter(str.isdigit, subj))) # get number of subject only
            
            if not flat_prior:
                # get current subjects data only
                this_priors = priors[str(this_subj)] # priors for current subject
            this_df = df_in[df_in['subject']==this_subj].copy()
            
            # the input to the model is the trial sequence = the order of letter-color pair for each participant
            [model_e, model_P, model_p, model_I, model_i, model_H, model_CH, model_D] = self.idt_model(this_df, 'touch_pair', elements, this_priors, flat_prior)
            
            # add to subject dataframe
            this_df['model_p'] = np.array(model_p)
            this_df['model_i'] = np.array(model_i)
            this_df['model_H'] = np.array(model_H)
            this_df['model_D'] = np.array(model_D)
            df_out = pd.concat([df_out, this_df])    # add current subject df to larger df
            
            df_prob_out['{}'.format(this_subj)] = np.array(model_P[-1])
            print(subj)
        
        # save whole DF
        df_out.to_csv(fn_in, float_format='%.16f') # overwrite subjects dataframe
        print('success: information_theory_estimates')
        
    
    def average_information_conditions(self, ):
        """Average the model parameters per subject per condition of interest. 

        Notes
        -----
        Save separate dataframes for the different combinations of factors in trial bin folder for plotting and jasp folders for statistical testing.
        'frequency' argument determines how the trials were split
        """     
        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # drop all unnamed columns
        DF.sort_values(by=['subject','trial_num'],inplace=True)
        DF.reset_index()
        
        ############################
        # drop outliers and missing trials
        DF = DF[DF['drop_trial']==0]
        ############################
        
        #interaction accuracy and frequency
        for pupil_dv in ['model_i', 'model_H', 'model_D']: #interaction accuracy and frequency
            
            '''
            ######## CORRECT x FREQUENCY ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject', 'correct', 'frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct-frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency', 'correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct-frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
            '''
            ######## CORRECT ########
            '''
            # MEANS subject x bin x tone x congruent
            DFOUT = DF.groupby(['subject', 'correct'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_correct_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # FOR PLOTTING

            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['correct',]) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_correct_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        
        '''
        ######## FREQUENCY ########
        '''
        for pupil_dv in ['model_i', 'model_H', 'model_D']: # mean accuracy
            DFOUT = DF.groupby(['subject','frequency'])[pupil_dv].mean()
            DFOUT.to_csv(os.path.join(self.trial_bin_folder,'{}_frequency_{}.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # For descriptives
            # save for RMANOVA format
            DFANOVA =  DFOUT.unstack(['frequency']) 
            print(DFANOVA.columns)
            DFANOVA.columns = DFANOVA.columns.to_flat_index() # flatten column index
            DFANOVA.to_csv(os.path.join(self.jasp_folder,'{}_frequency_{}_rmanova.csv'.format(self.exp,pupil_dv)), float_format='%.16f') # for stats
        print('success: average_information_conditions')


    def plot_information(self, ):
        """Plot the model parameters across trials and average over subjects
        Then, plot the model parameters by frequency

        Notes
        -----
        1 figure, GROUP LEVEL DATA
        x-axis is trials or frequency conditions.
        Figure output as PDF in figure folder.
        """
        dvs = ['model_D', 'model_i','model_H']
        ylabels = ['KL divergence', 'Surprise', 'Entropy', ]
        xlabel = 'Trials'
        colors = [ 'purple', 'teal', 'orange',]    
        
        fig = plt.figure(figsize=(4,4))
        
        subplot_counter = 1
        # PLOT ACROSS TRIALS
        for dvi, pupil_dv in enumerate(dvs):

            ax = fig.add_subplot(3, 3, subplot_counter) # 1 subplot per bin windo
            
            DFIN = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
                        
            subject_array = np.zeros((len(self.subjects), np.max(DFIN['trial_num'])))
        
            for s, subj in enumerate(self.subjects):
                this_subj = int(''.join(filter(str.isdigit, subj)))
                this_df = DFIN[DFIN['subject']==this_subj].copy()        
                subject_array[s,:] = np.ravel(this_df[[pupil_dv]])
                            
            self.tsplot(ax, subject_array, color=colors[dvi], label=ylabels[dvi])
    
            # set figure parameters
            ax.set_xlim([0, np.max(DFIN['trial_num'])+1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabels[dvi])
            # ax.legend()
            subplot_counter += 1
        
        # PLOT ACROSS FREQUENCY CONDITIONS
        factor = 'frequency'
        xlabel = 'Touch-pair frequency'
        xticklabels = ['Low','High'] # be careful and check order!
        bar_width = 0.7
        xind = np.arange(len(xticklabels))
        
        for dvi, pupil_dv in enumerate(dvs):
            
            ax = fig.add_subplot(3, 3, subplot_counter) # 1 subplot per bin window

            DFIN = pd.read_csv(os.path.join(self.trial_bin_folder,'{}_{}_{}.csv'.format(self.exp,'frequency',pupil_dv)), float_precision='%.16f')
            DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
            
            # Group average 
            GROUP = pd.DataFrame(DFIN.groupby([factor])[pupil_dv].agg(['mean','std']).reset_index())
            GROUP['sem'] = np.true_divide(GROUP['std'],np.sqrt(len(self.subjects)))
            print(GROUP)
            
            # ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
                       
            # plot bar graph
            # make sure to do low then high, flip x index as compared with GROUP index
            ax.bar(xind[0],np.array(GROUP['mean'][1]), width=bar_width, yerr=np.array(GROUP['sem'][1]), capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
            ax.bar(xind[1],np.array(GROUP['mean'][0]), width=bar_width, yerr=np.array(GROUP['sem'][0]), capsize=3, color=colors[dvi], edgecolor='black', ecolor='black')
                
            # individual points, repeated measures connected with lines
            # DFIN = DFIN.groupby(['subject',factor])[pupil_dv].mean() # hack for unstacking to work
            # DFIN = DFIN.unstack(factor)
            # for s in np.array(DFIN):
            #     ax.plot(xind, s, linestyle='-', marker='o', markersize=3, fillstyle='full', color='black', alpha=.1) # marker, line, black
                
            # set figure parameters
            # ax.set_title(ylabels[dvi]) # repeat for consistent formatting
            ax.set_ylabel(ylabels[dvi])
            ax.set_xlabel(xlabel)
            ax.set_xticks(xind)
            ax.set_xticklabels(xticklabels)
            # if pupil_dv == 'model_D':
            #     ax.set_ylim([0.004, 0.007])
            # if pupil_dv == 'model_i':
            #     ax.set_ylim([4.9, 5.2])
            # if pupil_dv == 'model_H':
            #     ax.set_ylim([4.5, 4.75])
            # if pupil_dv == 'model_D':
            #     ax.set_ylim([0.004, 0.00475])
            # if pupil_dv == 'model_i':
            #     ax.set_ylim([4.95, 5.05])
            # if pupil_dv == 'model_H':
            #     ax.set_ylim([4.85, 4.9])
            subplot_counter += 1
            
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_information.pdf'.format(self.exp)))
        print('success: plot_information')
        
        
    def pupil_information_correlation_matrix(self,):
        """Correlate information variables to evaluate multicollinearity.
        
        Notes
        -----
        Model estimates that are correlated per subject the tested at group level:
        model_i = surprise of current element at current trial
        model_H = entropy at current trial
        model_D = KL-divergence at current trial
        
        See figure folder for plot and output of t-test.
        """
        
        ivs = ['model_i', 'model_H', 'model_D',]
        labels = ['i' , 'H', 'KL',]

        DF = pd.read_csv(os.path.join(self.dataframe_folder,'{}_subjects.csv'.format(self.exp)), float_precision='%.16f')

        #### DROP OMISSIONS HERE ####
        DF = DF[DF['drop_trial'] == 0] # drop outliers based on RT
        #############################
        
        corr_out = []

        # loop subjects
        for s, subj in enumerate(self.subjects):
            
            this_subj = int(''.join(filter(str.isdigit, subj))) 
            # get current subject's data only
            this_df = DF[DF['subject']==this_subj].copy(deep=False)
                            
            x = this_df[ivs] # select information variable columns
            x_corr = x.corr() # correlation matrix
            
            
            corr_out.append(x_corr) # beta KLdivergence (target-prediction)
        
        corr_subjects = np.array(corr_out)
        corr_mean = np.mean(corr_subjects, axis=0)
        corr_std = np.std(corr_subjects, axis=0)
        
        t, pvals = sp.stats.ttest_1samp(corr_subjects, 0, axis=0)
        
        f = open(os.path.join(self.figure_folder, '{}_pupil_information_correlation_matrix.txt'.format(self.exp)), "w")
        f.write('corr_mean')
        f.write('\n')
        f.write('{}'.format(corr_mean))
        f.write('\n')
        f.write('\n')
        f.write('tvals')
        f.write('\n')
        f.write('{}'.format(t))
        f.write('\n')
        f.write('\n')
        f.write('pvals')
        f.write('\n')
        f.write('{}'.format(pvals))
        f.close
        
        ### PLOT ###
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(121)
        cbar_ax = fig.add_subplot(122)
        
        # mask for significance
        mask_pvals = pvals < 0.05
        mask_pvals = ~mask_pvals # True means mask this cell
        
        # plot only lower triangle
        mask = np.triu(np.ones_like(corr_mean))
        mask = mask + mask_pvals # only show sigificant correlations in heatmap
        
        # ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, mask=mask, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        ax = sns.heatmap(corr_mean, vmin=-1, vmax=1, cmap='bwr', cbar_ax=cbar_ax, xticklabels=labels, yticklabels=labels, square=True, annot=True, ax=ax)
        
        # whole figure format
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        fig.savefig(os.path.join(self.figure_folder,'{}_pupil_information_correlation_matrix.pdf'.format(self.exp)))
                        
        print('success: pupil_information_correlation_matrix')
        

    def dataframe_evoked_pupil_information_betas(self):
        """Multiple regression of theoretic variables onto pupil with other variables removed across timecourse.

        Notes
        -----
        Drop omission trials (in subject loop).
        Output in dataframe folder.
        """
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns

        ivs = ['model_i', 'model_D', 'pupil_baseline_feed_locked', 'RT']

        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas

        for t,time_locked in enumerate(self.time_locked):

            for cond in ['all_trials']:
            
                # save betas per timepoint
                df_out_c = pd.DataFrame() # timepoints x subjects
                df_out_i = pd.DataFrame() # timepoints x subjects
                df_out_D = pd.DataFrame() # timepoints x subjects
                df_out_b = pd.DataFrame() # timepoints x subjects
                df_out_r = pd.DataFrame() # timepoints x subjects
                
                # loop subjects
                for s, subj in enumerate(self.subjects):

                    this_subj = int(''.join(filter(str.isdigit, subj)))
                    # get current subject's data only

                    SBEHAV = DF[DF['subject']==this_subj].reset_index()
                    SPUPIL = pd.DataFrame(pd.read_csv(os.path.join(self.project_directory, subj, '{}_{}_{}_evoked_basecorr.csv'.format(subj, self.exp, time_locked)), float_precision='%.16f'))
                    
                    SPUPIL = SPUPIL.loc[:, ~SPUPIL.columns.str.contains('^Unnamed')] # remove all unnamed columns

                    # merge behavioral and evoked dataframes so we can group by conditions
                    SBEHAV
                    SDATA = pd.concat([SBEHAV,SPUPIL],axis=1)

                    #### DROP OMISSIONS HERE ####
                    SDATA = SDATA[SDATA['drop_trial'] == 0] # drop outliers based on RT
                    #############################

                    evoked_cols = np.char.mod('%d', np.arange(SPUPIL.shape[-1])) # get columns of pupil sample points only
                    
                    # save betas
                    save_timepoint_c = [] # constant
                    save_timepoint_i = [] # surprise
                    save_timepoint_D = [] # information gain
                    save_timepoint_b = [] # baselines
                    save_timepoint_r = [] # RTs
                    
                    # get IVs
                    X = SDATA[ivs].copy()  # true copy
                    
                    # Z-score IVs to account for large differences in scaling
                    X = X.apply(stats.zscore)

                    # loop timepoints, regress
                    for col in evoked_cols:
                        Y = SDATA[col] # pupil dv
                        X = sm.add_constant(X) # ivs

                        # Fit OLS model
                        model = sm.OLS(Y, X).fit()
                                                
                        # save betas for this timepoint
                        save_timepoint_c.append(model.params['const'])
                        save_timepoint_i.append(model.params['model_i'])
                        save_timepoint_D.append(model.params['model_D'])
                        save_timepoint_b.append(model.params['pupil_baseline_feed_locked'])
                        save_timepoint_r.append(model.params['RT'])
                        
                    # add column for each subject with timepoints as rows
                    df_out_c[subj] = np.array(save_timepoint_c)
                    df_out_c[subj] = df_out_c[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
                    # add column for each subject with timepoints as rows
                    df_out_i[subj] = np.array(save_timepoint_i)
                    df_out_i[subj] = df_out_i[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
                    # add column for each subject with timepoints as rows
                    df_out_D[subj] = np.array(save_timepoint_D)
                    df_out_D[subj] = df_out_D[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
                    # # add column for each subject with timepoints as rows
                    df_out_b[subj] = np.array(save_timepoint_b)
                    df_out_b[subj] = df_out_b[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df
                    # add column for each subject with timepoints as rows
                    df_out_r[subj] = np.array(save_timepoint_r)
                    df_out_r[subj] = df_out_r[subj].apply(lambda x: '%.16f' % x) # remove scientific notation from df

                # save output file
                df_out_c.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_constant.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
                df_out_i.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_model_i.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
                df_out_D.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_model_D.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
                df_out_b.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_pupil_baseline_feed_locked.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
                df_out_r.to_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_RT.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
        print('success: dataframe_evoked_pupil_information_betas')
        
    
    def plot_evoked_pupil_information_betas(self):
        """Plot beta coefficients from regression between pupil and information variables across timecourse.
        
        Notes
        -----
        Always feed_locked pupil response.
        """
        
        sample_rate = self.sample_rate 
        
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        iv = 'aq_score'
    
        # colors = ['teal', 'purple', 'blue', 'red']
        # alphas = [1]
        # labels = ['model_i' , 'model_D', 'pupil_baseline_feed_locked', 'RT']
        colors = ['teal', 'purple', ] 
        alphas = [1]
        labels = ['model_i' , 'model_D', ]
        
        # colors = ['black', ]
        # alphas = [1]
        # labels = ['constant' ,]
        
        time_locked = 'feed_locked'
        t = 0
        
        for i,cond in enumerate(['all_trials']):
            
            #######################
            # FEEDBACK PLOT R FOR EACH PUPIl COND
            #######################
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            factor = 'subject'
            kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*sample_rate) # length of evoked responses
            # determine time points x-axis given sample rate
            event_onset = int(abs(self.pupil_step_lim[t][0]*sample_rate))
            end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*sample_rate)
            mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                        
            # Compute means, sems across group
            for i,model_iv in enumerate(labels): # conservative analysis, no correct-frequency condition
                CORR = pd.read_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_{}.csv'.format(self.exp, time_locked, cond, model_iv)), float_precision='high')
                CORR = CORR.loc[:, ~CORR.columns.str.contains('^Unnamed')] # remove all unnamed columns
                
                TS = np.array(CORR.T)   
                
                self.tsplot(ax, TS, alpha_fill=0.2, alpha_line=1, color=colors[i], label=model_iv)
            
                # stats        
                self.cluster_sig_bar_1samp(array=TS, x=pd.Series(range(TS.shape[-1])), yloc=1+i, color=colors[i], ax=ax, threshold=0.05, nrand=5000, cluster_correct=True)           
            
            # set figure parameters
            ax.axvline(int(abs(self.pupil_step_lim[t][0]*sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # Shade all time windows of interest in grey, will be different for events
            for twi in self.pupil_time_of_interest[t]:
                tw_begin = int(event_onset + (twi[0]*sample_rate))
                tw_end = int(event_onset + (twi[1]*sample_rate))
                ax.axvspan(tw_begin, tw_end, facecolor='k', alpha=0.1)
            
            xticks = [event_onset, event_onset+(0.5*sample_rate*1), event_onset+(0.5*sample_rate*2), event_onset+(0.5*sample_rate*3), event_onset+(0.5*sample_rate*4), event_onset+(0.5*sample_rate*5), event_onset+(0.5*sample_rate*6), event_onset+(0.5*sample_rate*7)]
            ax.set_xticks(xticks)
            ax.set_xticklabels([0, self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])

            # ax.set_ylim(ylim_feed)
            # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
            ax.set_xlabel('Time from feedback (s)')
            ax.set_ylabel('Beta coefficient')
            ax.set_title('{} {}'.format(time_locked, cond))
            ax.legend()
        
            # whole figure format
            sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_{}_evoked_betas_{}.pdf'.format(self.exp, time_locked, cond)))
        print('success: plot_evoked_pupil_information_betas')
        
        
    def dataframe_evoked_correlation_information_betas_AQ(self, ):
        """Timeseries individual differences (Spearman rank) correlation between beta coefficients (surprise and information gain) and AQ score for:
        all trials

        Notes
        -----
        Omissions dropped in dataframe_evoked_regression()
        Output correlation coefficients per time point in dataframe folder.
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')       
        AQ.sort_values(by='subject', ascending=True, inplace=True) # sort by participant number
        iv = 'aq_score'
        aq_scores = np.array(AQ[iv])
                
        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas
        
        
        for t,time_locked in enumerate(self.time_locked):
            
            df_out = pd.DataFrame() # timepoints x participants
            
            for cond in ['all_trials']:
                
                for model_iv in ['model_i', 'model_D']:
                                        
                    DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_{}_evoked_regression_{}_{}.csv'.format(self.exp, time_locked, cond, model_iv)), float_precision='high')
                    DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
                    DF = DF[sorted(DF.columns)] # make sure to sort by participant number 
                    
                    # loop timepoints, correlation
                    save_timepoint_r = []
                    save_timepoint_p = []
                    
                    for timepoint in np.arange(DF.shape[0]):
                        
                        y = DF.iloc[timepoint,:] # current timepoint, all participants
                        x = aq_scores # iv
                    
                        r, pval = sp.stats.spearmanr(np.array(x), np.array(y))

                        save_timepoint_r.append(self.fisher_transform(r))
                        save_timepoint_p.append(pval)
                    
                    # add column for each subject with timepoints as rows
                    df_out['{}_{}_r'.format(cond, model_iv)] = np.array(save_timepoint_r)
                    df_out['{}_{}_pval'.format(cond, model_iv)] = np.array(save_timepoint_p)
                    # remove scientific notation from df
                    df_out['{}_{}_r'.format(cond, model_iv)] = df_out['{}_{}_r'.format(cond, model_iv)].apply(lambda x: '%.16f' % x) 
                    df_out['{}_{}_pval'.format(cond, model_iv)] = df_out['{}_{}_pval'.format(cond, model_iv)].apply(lambda x: '%.16f' % x)
                
            # save output file
            df_out.to_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_information_betas_AQ.csv'.format(self.exp, time_locked)), float_format='%.16f')
        print('success: dataframe_evoked_correlation_information_betas_AQ')
        

    def plot_evoked_correlation_information_betas_AQ(self):
        """Plot correlation between pupil beta coefficients (surprise and information gain) and AQ scores.
        
        Notes
        -----
        Always feed_locked pupil response.
        """
        
        sample_rate = self.sample_rate 
        
        ylim_feed = [-0.2, 0.2]
        tick_spacer = 0.1
        
        iv = 'aq_score'
    
        colors = ['teal', 'purple'] 
        alphas = [1]
        labels = ['model_i' , 'model_D']
        
        time_locked = 'feed_locked'
        t = 0
        CORR = pd.read_csv(os.path.join(self.dataframe_folder,'{}_{}_evoked_correlation_information_betas_AQ.csv'.format(self.exp, time_locked)), float_precision='high')
        
        for i,cond in enumerate(['all_trials']):
        
            #######################
            # FEEDBACK PLOT R FOR EACH PUPIl COND
            #######################
            fig = plt.figure(figsize=(8,5))
            ax = fig.add_subplot(111)
            factor = 'subject'
            kernel = int((self.pupil_step_lim[t][1]-self.pupil_step_lim[t][0])*sample_rate) # length of evoked responses
            # determine time points x-axis given sample rate
            event_onset = int(abs(self.pupil_step_lim[t][0]*sample_rate))
            end_sample = int((self.pupil_step_lim[t][1] - self.pupil_step_lim[t][0])*sample_rate)
            mid_point = int(np.true_divide(end_sample-event_onset,2) + event_onset)
                        
            # Compute means, sems across group
            for i,model_iv in enumerate(labels): # conservative analysis, no correct-frequency condition
                TS = np.array(CORR['{}_{}_r'.format(cond, model_iv)])
                pvals = np.array(CORR['{}_{}_pval'.format(cond, model_iv)])
                    
                ax.plot(pd.Series(range(TS.shape[-1])), TS, color=colors[i], label=model_iv)
            
                # stats        
                self.timeseries_fwer_correction(pvals=pvals, xind=pd.Series(range(pvals.shape[-1])), alpha=0.05, color=colors[i], yloc=i, ax=ax)
            
            # set figure parameters
            ax.axvline(int(abs(self.pupil_step_lim[t][0]*sample_rate)), lw=1, alpha=1, color = 'k') # Add vertical line at t=0
            ax.axhline(0, lw=1, alpha=1, color = 'k') # Add horizontal line at t=0
        
            # Shade all time windows of interest in grey, will be different for events
            for twi in self.pupil_time_of_interest[t]:
                tw_begin = int(event_onset + (twi[0]*sample_rate))
                tw_end = int(event_onset + (twi[1]*sample_rate))
                ax.axvspan(tw_begin, tw_end, facecolor='k', alpha=0.1)
            
            xticks = [event_onset, event_onset+(0.5*sample_rate*1), event_onset+(0.5*sample_rate*2), event_onset+(0.5*sample_rate*3), event_onset+(0.5*sample_rate*4), event_onset+(0.5*sample_rate*5), event_onset+(0.5*sample_rate*6), event_onset+(0.5*sample_rate*7)]
            ax.set_xticks(xticks)
            ax.set_xticklabels([0, self.pupil_step_lim[t][1]-(.5*6), self.pupil_step_lim[t][1]-(.5*5), self.pupil_step_lim[t][1]-(.5*4), self.pupil_step_lim[t][1]-(.5*3), self.pupil_step_lim[t][1]-(.5*2),  self.pupil_step_lim[t][1]-(.5*1), self.pupil_step_lim[t][1]])

            # ax.set_ylim(ylim_feed)
            # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_spacer))
            ax.set_xlabel('Time from feedback (s)')
            ax.set_ylabel('rs')
            ax.set_title('{} {}'.format(time_locked, cond))
            ax.legend()
        
            # whole figure format
            sns.despine(offset=10, trim=True)
            # plt.tight_layout()
            fig.savefig(os.path.join(self.figure_folder,'{}_{}_evoked_correlation_information_betas_AQ_{}.pdf'.format(self.exp, time_locked, cond)))
        print('success: plot_evoked_correlation_information_betas_AQ')
        
        
    def phasic_correlation_information_betas_AQ(self,):
        """Multiple regression of model parameters (surprise and information gain) onto pupil data.
        
        Notes:
        -----
        Drop omission trials.
        """
        
        AQ = pd.read_csv(os.path.join(self.dataframe_folder, '{}_aq_scores_coded.csv'.format(self.exp)), float_precision='high')       
        AQ.sort_values(by='subject', ascending=True, inplace=True) # sort by participant number
        iv = 'aq_score'
        aq_scores = np.array(AQ[iv])
        
        # Run multiple regression: pupil_feed_locked_t1 ~ model_i + model_D + pupil_baseline_feed_locked + RT
        DF = pd.read_csv(os.path.join(self.dataframe_folder, '{}_subjects.csv'.format(self.exp)), float_precision='%.16f')
        DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')] # remove all unnamed columns
        
        #### DROP OMISSIONS HERE ####
        DF = DF[DF['drop_trial'] == 0] # drop trials
        #############################
        
        dv = 'pupil_feed_locked_t1'
        ivs = ['model_i', 'model_D', 'pupil_baseline_feed_locked', 'RT']

        pd.set_option('display.float_format', lambda x: '%.16f' % x) # suppress scientific notation in pandas

        for t,time_locked in enumerate(self.time_locked):

            for cond in ['all_trials']:
                
                all_subjects = []
                # loop subjects
                for s, subj in enumerate(self.subjects):
                    
                    # get current subject's data only
                    this_subj = int(''.join(filter(str.isdigit, subj)))
                    SDATA = DF[DF['subject']==this_subj].reset_index()
                    
                    # get IVs
                    X = SDATA[ivs].copy()  # true copy
                    
                    # Z-score IVs to account for large differences in scaling
                    X = X.apply(stats.zscore)

                    Y = SDATA[dv] # pupil dv
                    X = sm.add_constant(X) # ivs

                    # Fit OLS model
                    model = sm.OLS(Y, X).fit()
                    # save betas
                    series = model.params
                    series.name = subj
                    all_subjects.append(series)
                
                # save output file
                df_out = pd.concat(all_subjects, axis=1)
                df_out = df_out.T # subjects x ivs
                df_out['aq_score'] = aq_scores
                df_out.to_csv(os.path.join(self.dataframe_folder, '{}_{}_phasic_regression_{}.csv'.format(self.exp, time_locked, cond)), float_format='%.16f')
        print('success: phasic_correlation_information_betas_AQ')


    def plot_phasic_correlation_information_betas_AQ(self,):
        """Correlate and plot the information beta coefficients by AQ scores within the time window of interest.
        
        Notes
        -----
        GROUP LEVEL DATA
        x-axis is model ivs.
        """        
        ylabel = 'rs'
        xlabel = 'Predictors'
        betas = ['model_i', 'model_D']
                
        for t,time_locked in enumerate(self.time_locked):
            
            for cond in ['all_trials']:
                
                DFIN = pd.read_csv(os.path.join(self.dataframe_folder, '{}_{}_phasic_regression_{}.csv'.format(self.exp, time_locked, cond)), float_precision='high')
                DFIN = DFIN.loc[:, ~DFIN.columns.str.contains('^Unnamed')] # drop all unnamed columns
                
                n = len(betas)
                fig, axes = plt.subplots(1, n, figsize=(5*n, 4))

                for ax, model_iv in zip(axes, betas):
                
                    x = np.array(DFIN['aq_score'])
                    y = np.array(DFIN[model_iv])
                    r, pval = stats.spearmanr(x,y)
                      
                    # fit regression line
                    ax.plot(x, y, 'o', markersize=3, color='purple') # marker, line, black
                    m, b = np.polyfit(x, y, 1)
                    ax.plot(x, m*x+b, color='black', alpha=.5)
                
                    # set figure parameters
                    ax.set_title('rs = {}, p = {}'.format(np.round(r,2),np.round(pval,3)))

                    # set figure parameters
                    ax.set_xlabel('AQ score')
                    ax.set_ylabel('Beta coefficient {}'.format(model_iv))

                sns.despine(offset=10, trim=True)
                plt.tight_layout()
                fig.savefig(os.path.join(self.figure_folder, '{}_{}_phasic_correlation_information_betas_AQ_{}.pdf'.format(self.exp, time_locked, cond)))
        print('success: plot_phasic_correlation_information_betas_AQ')
        
                