#!/usr/bin/env python
# encoding: utf-8
"""
================================================
Pupil dilation offers a time-window in prediction error

Data set #2 Letter-color 2AFC task - RUN ANALYSIS HERE
Python code O.Colizoli 2024 (olympia.colizoli@donders.ru.nl)
Python 3.6

Notes
-----
>> conda install matplotlib # fixed the matplotlib crashing error in 3.6
================================================
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time, shutil
import numpy as np
import pandas as pd
import oddball_training_visual as training_higher
import preprocessing_functions_visual as pupil_preprocessing
import higher_level_functions_visual as higher
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
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-letter_color_visual_decision' # 2AFC Decision Task

# copy 'raw' to derivatives if it doesn't exist:
if not os.path.isdir(data_dir):
    shutil.copytree(source_dir, data_dir) 
else:
    print('Derivatives directory exists. Continuing...')

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'analysis', 'participants_letter_color_visual.csv'))
subjects = ['sub-{}'.format(int(s)) for s in ppns['subject']]

# -----------------------
# Odd-ball Training Task, MEAN responses and group level statistics
# ----------------------- 
if training:  
    oddballTraining = training_higher.higherLevel(
        subjects          = subjects, 
        experiment_name   = 'task-letter_color_visual_training',
        project_directory = data_dir
        )
    oddballTraining.create_subjects_dataframe()       # drops missed trials, saves higher level data frame
    oddballTraining.average_conditions()              # group level data frames for all main effects + interaction
    oddballTraining.plot_behav()                      # plots behavior, group level, main effects + interaction
    oddballTraining.calculate_actual_frequencies()    # calculates the actual frequencies of pairs
    oddballTraining.information_theory_estimates()
    oddballTraining.plot_information_frequency()
    
# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
msgs                    = ['start recording', 'stop recording', 'phase 1', 'phase 7']; # this will change for each task (keep phase 1 for locking to breaks)
phases                  = ['phase 7'] # of interest for analysis
time_locked             = ['feed_locked'] # events to consider (note: these have to match phases variable above)
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window, 3.5]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
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
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)

        pupilPreprocess = pupil_preprocessing.pupilPreprocess(
            subject             = subj,
            edf                 = edf,
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
        pupilPreprocess.convert_edfs()              # converts EDF to asc, msg and gaze files (run locally)
        pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
        pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

# -----------------------
# 2AFC Decision Task, Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj, experiment_name)
        trialLevel = pupil_preprocessing.trials(
            subject             = subj,
            edf                 = edf,
            project_directory   = data_dir,
            sample_rate         = sample_rate,
            phases              = phases,
            time_locked         = time_locked,
            pupil_step_lim      = pupil_step_lim, 
            baseline_window     = baseline_window
            )
        trialLevel.event_related_subjects(pupil_dv='pupil_psc')  # psc: percent signal change, per event of interest, 1 output for all trials+subjects
        trialLevel.event_related_baseline_correction()           # per event of interest, baseline corrrects evoked responses

# -----------------------
# 2AFC Decision Task, MEAN responses and group level statistics 
# ----------------------- 

if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = experiment_name,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim,                
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = [[[0.075,0.95]]], # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil
        )
    # higherLevel.create_subjects_dataframe(blocks = break_trials+[240])  # add baselines, concantenates all subjects, flags missed trials, saves higher level data frame
    ''' Note: the functions after this are using: task-letter_color_visual_decision_subjects.csv
    '''
    ''' Evoked pupil response
    '''
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    
    ''' Correlation with AQ score across evoked
    '''
    # higherLevel.dataframe_evoked_correlation_AQ(df=ppns)  # compute correlation of AQ onto pupil conditions across pupil time course
    # higherLevel.plot_evoked_correlation_AQ()              # plot correlation of AQ onto pupil conditions across pupil timecourse
    
    ''' Average pupil response (time window) and c
    '''
    # higherLevel.higherlevel_get_phasics()         # computes phasic pupil in time window of interest for each subject (adds to log files)
    # higherLevel.create_subjects_dataframe(blocks = break_trials+[240])  # update after phasic pupil added
    # higherLevel.average_conditions()              # group level data frames for all main effects + interaction
    # higherLevel.regression_pupil_AQ(df=ppns)      # multiple regression of AQ components (IVs) onto average pupil response in time window of interest
    # higherLevel.correlation_AQ(df=ppns)           # correlations between pupil, RT conditions and AQ
    # higherLevel.plot_AQ_histogram(df=ppns)        # plot a histogram of the AQ score
    # higherLevel.plot_phasic_pupil_unsigned_pe()   # plots the mean pupil response as a function of the frequency conditions
    # higherLevel.plot_AQ_covariance(df=ppns)         # plots the correlation matrix of the AQ sub-scores
    
    ''' Regression pupil~AQ by block
    '''
    # higherLevel.regression_pupil_AQ_blocks(df=ppns)    # by block, multiple regression of AQ components (IVs) onto average pupil response in time window of interest
    higherLevel.plot_regression_pupil_AQ_blocks()       # by block, plot pupil~AQ regression results
    