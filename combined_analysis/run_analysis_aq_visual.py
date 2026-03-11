#!/usr/bin/env python
# encoding: utf-8
"""
================================================
VISUAL DATA SET
A detail-oriented approach: Autistic traits scale with prediction errors during associative learning

Python code O.Colizoli 2025 (olympia.colizoli@donders.ru.nl)
Python 3.6

anaconda environment aq36

Each trial:
# phase 1 new trial onset
# phase 2 letter stimulus onset
# phase 3 delay onset
# phase 4 target onset
# phase 5 response
# phase 6 pupil baseline feedback
# phase 7 feedback onset

================================================
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time, shutil
import numpy as np
import pandas as pd
import oddball_training_aq_visual as training_higher
import preprocessing_functions_aq as pupil_preprocessing
import higher_level_functions_aq_visual as higher
from IPython import embed as shell # for debugging
# Need to have the EYELINK software installed on the terminal

# -----------------------
# Levels (toggle True/False)
# ----------------------- 
training        = False  # process the logfiles and average the performance on the odd-ball training task
pre_process     = False  # pupil preprocessing is done on entire time series during the 2AFC decision task
trial_process   = False  # cut out events for each trial and calculate trial-wise baselines, baseline correct evoked responses (2AFC decision)
higher_level    = True   # all subjects' dataframe, pupil and behavior higher level analyses & figures (2AFC decision)
 
# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'dataset-aq_visual', 'sourcedata')
data_dir        = os.path.join(home_dir, 'dataset-aq_visual', 'derivatives')
experiment_name = 'task-aq_visual' # 2AFC Decision Task

# copy 'raw' to derivatives if it doesn't exist:
if not os.path.isdir(data_dir):
    shutil.copytree(source_dir, data_dir) 
else:
    print('Derivatives directory exists. Continuing...')

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'combined_analysis', 'participants_aq_visual.csv'))
subjects = ['sub-{}'.format(int(s)) for s in ppns['subject']]

# -----------------------
# AQ Scoring
# -----------------------
aq_scoring = pd.read_csv(os.path.join(home_dir, 'combined_analysis', 'aq_scoring.csv'))

# -----------------------
# Odd-ball Training Task, MEAN responses and group level statistics
# ----------------------- 
if training:  
    oddballTraining = training_higher.higherLevel(
        subjects          = subjects, 
        experiment_name   = 'task-aq_visual_training',
        project_directory = data_dir
        )
    oddballTraining.create_subjects_dataframe()       # drops missed trials, saves higher level data frame
    oddballTraining.average_conditions()              # group level data frames for all main effects + interaction
    oddballTraining.plot_behav()                      # plots behavior, group level, main effects + interaction
    oddballTraining.calculate_actual_frequencies()    # calculates the actual frequencies of pairs
    oddballTraining.information_theory_estimates()    # run ideal learner model to get probabilities at end of oddball task
    oddballTraining.plot_information_frequency()       # plot probability and surprise at the end of oddball task for each frequency condition
    
# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
msgs                    = ['start recording', 'stop recording', 'phase 1', 'phase 2', 'phase 5', 'phase 7']; # this will change for each task (keep phase 1 for locking to breaks)
phases                  = ['phase 2', 'phase 5', 'phase 7'] # of interest for analysis
time_locked             = ['cue_locked', 'response_locked', 'feed_locked'] # events to consider (note: these have to match phases variable above)
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window, 3.5], [-baseline_window, 3.5], [-baseline_window, 3.5]]  # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
sample_rate             = 1000 # Hz
break_trials            = [60, 120, 180]  # which trial comes AFTER each break

# -----------------------
# 2AFC Decision Task, Pupil preprocessing, full time series
# -----------------------
if pre_process:
    # preprocessing-specific parameters
    tolkens = ['ESACC', 'EBLINK' ]      # check saccades and blinks based on EyeLink
    tw_blinks = 0.15                    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors

    for s,subj in enumerate(subjects):
        edf = '{}_task-letter_color_visual_decision_recording-eyetracking_physio'.format(subj, experiment_name)
        
        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
            experiment_name     = experiment_name,
            project_directory   = data_dir,
            eye                 = ppns['eye'][s],
            break_trials        = break_trials,
            msgs                = msgs, 
            tolkens             = tolkens,
            sample_rate         = sample_rate,
            tw_blinks           = tw_blinks,
            mph                 = mph,
            mpd                 = mpd,
            threshold           = threshold,
            )
        # pupilPreprocess.convert_edfs()      # converts EDF to asc, msg and gaze files (run locally)
        # pupilPreprocess.extract_pupil()     # read trials, and saves time locked pupil series as NPY array in processed folder
        pupilPreprocess.preprocess_pupil()  # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

# -----------------------
# 2AFC Decision Task, Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)
        trialLevel = pupil_preprocessing.trials(
            subject             = subj,
            experiment_name     = experiment_name,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            phases              = phases,
            time_locked         = time_locked,
            pupil_step_lim      = pupil_step_lim, 
            baseline_window     = baseline_window
            )
        trialLevel.event_related_subjects(pupil_dv='ESACC')      # saves the saccade markers across timecourse for each event of interest
        trialLevel.event_related_subjects(pupil_dv='ESACC_END')  # saves the saccade markers across timecourse for each event of interest
        trialLevel.event_related_subjects(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.save_baselines()                              # saves pre-cue and pre-feedback baselines
        trialLevel.event_related_baseline_correction()           # per event of interest, baseline corrrects evoked responses
        trialLevel.event_related_saccades()                      # marks each sample of the evoked response if a saccade was present, save counts per trial
        trialLevel.event_related_blinks()                        # marks each sample of the evoked response if a blink was present, save counts per trial

# -----------------------
# MEAN responses and group level statistics 
# ----------------------- 
if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = experiment_name,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = ['feed_locked'], 
        pupil_step_lim          = [[-baseline_window, 3.5]],                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = [[[0.075, 0.95]]], # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil
        )
    
    # higherLevel.compute_aq_score(ppns, aq_scoring)  # calculate AQ score and subscales
    # higherLevel.compute_blink_percentages(thresh = 0.30)      # computes the amount of interpolated data per trial and excludes based on threshold
    # higherLevel.higherlevel_get_phasics()           # computes phasic pupil in time window of interest for each subject (adds to log files), removes baseline pupil and RT from phasics (per subject)
    # higherLevel.create_subjects_dataframe(exclude_interp=0)   # add baselines, concantenates all subjects, flags missed trials, saves higher level data frame
    ''' Note: the functions after this are using: task-aq_visual_subjects.csv
    '''
    ''' Evoked pupil response
    '''
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series

    ''' Correlation with AQ score across evoked time course
    '''
    # higherLevel.dataframe_evoked_correlation_AQ()  # compute correlation of AQ onto pupil conditions across pupil time course
    # higherLevel.plot_evoked_correlation_AQ()       # plot correlation of AQ onto pupil conditions across pupil timecourse

    ''' Average pupil response (time window) and c
    '''
    # higherLevel.average_conditions()              # group level data frames for all main effects + interaction
    # higherLevel.regression_pupil_AQ()             # multiple regression of AQ components (IVs) onto average pupil response in time window of interest
    # higherLevel.correlation_AQ()                  # correlations between pupil, RT conditions and AQ
    # higherLevel.plot_phasic_pupil_unsigned_pe()   # plots the mean pupil response as a function of the frequency conditions
    # higherLevel.plot_AQ_histogram()               # plot a histogram of the AQ score
    # higherLevel.plot_AQ_covariance()              # plots the correlation matrix of the AQ sub-scores

    ''' Regression pupil~AQ by block
    '''
    # higherLevel.plot_behav_blocks()                     # plot the accuracy are RT per block
    ## higherLevel.correlation_AQ_blocks()               # by block, correlations between pupil and AQ
    # higherLevel.regression_pupil_AQ_blocks()        # by block, multiple regression of AQ components (IVs) onto average pupil response in time window of interest
    # higherLevel.plot_regression_pupil_AQ_blocks()   # by block, plot pupil~AQ regression results
    
    ''' Ideal Learner Model
    '''
    # higherLevel.information_theory_estimates(flat_prior=False) # run model with uniform prior distribution
    # higherLevel.average_information_conditions()
    # higherLevel.plot_information()
    # higherLevel.pupil_information_correlation_matrix()
    # higherLevel.dataframe_evoked_pupil_information_betas()
    # higherLevel.plot_evoked_pupil_information_betas()
    # higherLevel.dataframe_evoked_correlation_information_betas_AQ()  # compute correlation of AQ onto beta coefficients (surprise & information gain) across pupil time course
    # higherLevel.plot_evoked_correlation_information_betas_AQ()
    # higherLevel.phasic_correlation_information_betas_AQ()
    higherLevel.plot_phasic_correlation_information_betas_AQ()
    
    
    
    
    
    