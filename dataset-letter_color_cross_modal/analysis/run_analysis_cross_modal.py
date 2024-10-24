#!/usr/bin/env python
# -*- coding: utf-8 -*-
# LISA VERSION
"""
Letter-color associations formed through cross-modal statistical learning
"Letter-color cross modal 2AFC task" for short
Python code by O.Colizoli 2022
Python 3.6
"""

############################################################################
# PUPIL ANALYSES
############################################################################
# importing python packages
import os, sys, datetime, time
import numpy as np
import pandas as pd
import oddball_training_cross_modal as training_higher
import preprocessing_functions_cross_modal as pupil_preprocessing
import higher_level_functions_cross_modal as higher
# conda install matplotlib # fixed the matplotlib crashing error in 3.6
from IPython import embed as shell # for debugging

# -----------------------
# Levels
# ----------------------- 
training        = False  # process the logfiles and average the performance on the odd-ball training task
pre_process     = True # pupil preprocessing is done on entire time series during the 2AFC decision task
trial_process   = False # cut out events for each trial and calculate trial-wise baselines, baseline correct evoked responses (2AFC decision)
higher_level    = False  # all subjects' dataframe, pupil and behavior higher level analyses & figures (2AFC decision)

# -----------------------
# Paths
# ----------------------- 
# set path to home directory
home_dir        = os.path.dirname(os.getcwd()) # one level up from analysis folder
source_dir      = os.path.join(home_dir, 'raw')
data_dir        = os.path.join(home_dir, 'derivatives')
experiment_name = 'task-letter_color_cross_modal_decision'

# -----------------------
# Participants
# -----------------------
ppns     = pd.read_csv(os.path.join(home_dir, 'analysis', 'participants_letter_color_cross_modal.csv'))
# only run participants who scored higher than chance in 70% condition
ppns =ppns[ ppns['higher_chance_70']==1].copy()
subjects = ['sub-{}'.format(int(s)) for s in ppns['subject']]


# -----------------------
# Odd-ball Training Task, MEAN responses and group level statistics
# ----------------------- 
if training:
    oddballTraining = training_higher.higherLevel(
        subjects          = subjects, 
        experiment_name   = 'task-letter_color_cross_modal_training',
        project_directory = data_dir
        )
    # oddballTraining.create_subjects_dataframe()       # drops missed trials, saves higher level data frame
    # oddballTraining.average_conditions()              # group level data frames for all main effects + interaction
    # oddballTraining.plot_behav()                      # plots behavior, group level, main effects + interaction
    oddballTraining.calculate_actual_frequencies()    # calculates the actual frequencies of pairs

# -----------------------
# Event-locked pupil parameters (shared)
# -----------------------
msgs                    = ['start recording', 'stop recording','phase 1','phase 4','phase 5']; # this will change for each task (keep phase 1 for locking to breaks)
phases                  = ['phase 4', 'phase 5'] # of interest for analysis (phase 4 is feedback, phase 5 is response/choice)
time_locked             = ['feed_locked','resp_locked'] # events to consider (note: these have to match phases variable above)
baseline_window         = 0.5 # seconds before event of interest
pupil_step_lim          = [[-baseline_window,3.0],[-baseline_window,3.0]] # size of pupil trial kernels in seconds with respect to first event, first element should max = 0!
sample_rate             = 1000 # Hz
break_trials            = [48,96,144]  # which trial comes AFTER each break

# -----------------------
# Pupil preprocessing, full time series
# -----------------------
if pre_process:
    # preprocessing-specific parameters
    tolkens = ['ESACC', 'EBLINK' ]      # check saccades and blinks based on EyeLink
    tw_blinks = 0.15                    # seconds before and after blink periods for interpolation
    mph       = 10      # detect peaks that are greater than minimum peak height
    mpd       = 1       # blinks separated by minimum number of samples
    threshold = 0       # detect peaks (valleys) that are greater (smaller) than `threshold` in relation to their immediate neighbors

    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)

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
        # pupilPreprocess.convert_edfs()              # converts EDF to asc, msg and gaze files (run locally)
        pupilPreprocess.extract_pupil()             # read trials, and saves time locked pupil series as NPY array in processed folder
        pupilPreprocess.preprocess_pupil()          # blink interpolation, filtering, remove blinks/saccades, split blocks, percent signal change, plots output

# -----------------------
# Pupil trials & mean response per event type
# -----------------------      
if trial_process:  
    # process 1 subject at a time
    for s,subj in enumerate(subjects):
        edf = '{}_{}_recording-eyetracking_physio'.format(subj,experiment_name)
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
        trialLevel.save_baselines('feed_locked')                   # save the baseline pupil dilation locked to feedback for the response-locked pupil dilation
        trialLevel.event_related_baseline_correction()           # baseline corrrects evoked responses, with alwasy respect to feedback (stimulus onset)

# -----------------------
# MEAN responses and group level statistics
# ----------------------- 
if higher_level:  
    higherLevel = higher.higherLevel(
        subjects                = subjects, 
        experiment_name         = experiment_name,
        project_directory       = data_dir, 
        sample_rate             = sample_rate,
        time_locked             = time_locked,
        pupil_step_lim          = pupil_step_lim, # matching resp-locked               
        baseline_window         = baseline_window,              
        pupil_time_of_interest  = [[[0.25,1.0], [2.5,3.0]], [[0.25,1.0], [2.5,3.0]]], # time windows to average phasic pupil, per event, in higher.plot_evoked_pupil 
        )
    # higherLevel.higherlevel_get_phasics()        # computes phasic pupil for each subject (adds to log files)
    # higherLevel.create_subjects_dataframe(blocks=break_trials+[240])  # concantenates all subjects, flags missed trials, saves higher level data frame
    ''' Note: the functions after this are using: task-letter_color_cross_modal_decision_subjects.csv
    '''
    higherLevel.average_conditions()           # regress RT out of phasic pupil, group level data frames for all main effects + interaction
    # higherLevel.plot_phasic_pupil_pe()         # plots the interaction between the frequency and accuracy in phasic pupil time windows
    # higherLevel.plot_behavior()                # simple bar plots of accuracy and RT per mapping condition
    # higherLevel.plot_uncertainty_rt()          # plots the interaction between frequency and accuracy in RT
    # higherLevel.individual_differences()
    
    ''' Evoked pupil response
    '''
    # higherLevel.dataframe_evoked_pupil_higher()  # per event of interest, outputs one dataframe or np.array? for all trials for all subject on pupil time series
    # higherLevel.plot_evoked_pupil()           # plots evoked pupil per event of interest, group level, main effects + interaction
    
    ''' Correlation with AQ score
    '''
    # higherLevel.dataframe_evoked_correlation_AQ(df=ppns)
    # higherLevel.plot_evoked_correlation_AQ()
    # higherLevel.regression_pupil_AQ(df=ppns)  # multiple regression of AQ components (IVs) onto average pupil response in early time window
    higherLevel.correlation_frequency_AQ(df=ppns)
    higherLevel.correlation_accuracy_AQ(df=ppns)
    