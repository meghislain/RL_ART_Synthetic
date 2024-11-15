from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
import sys
import torch as th
from ARIES.evaluation import *
from ARIES.network import Custom_CombinedExtractor, Custom_CNN
from ARIES.learning_rate import linear_schedule, stair_schedule
from ARIES.parser import parameters
from stable_baselines3.common.utils import get_linear_fn
import pickle

import os
import logging

import numpy as np
sav = ""

from RL.environment import TumorEnv
saving_path = sav + "/Results"

# Instantiate the env
name = parameters.date +"_DQ" +str(parameters.dq) +"_tarSize"+str(parameters.ts)+"_SL" + str(parameters.sl)
num_col = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position
num_row = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position

n_value = []
n_cpu_cores = 2
th.set_num_threads(n_cpu_cores)

env = TumorEnv(n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)
env_eval = TumorEnv(mode=0, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)
env_test = TumorEnv(mode=1, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc)
testing = 150
M = testing//5
model = DQN.load(sav + "/Model.zip", env, device="cuda:1")

for j in range(M):
       mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV = evaluate_over_treatment_daily_3D(model, env_test, n_eval_episodes=parameters.n_test, epoch=j)
       dic = {"Accumulation Dose": Accum_DM, "actions": actions, "Beam position": beam_pos, 
              "real tumor position": real_pos, "noisy tumor position": noisy_pos, "target size": parameters.ts, "form" : parameters.form, "reward" : rewards, "Signal Length" : parameters.sl,"D98_PTV" : D98_PTV_treat, "D80_PTV" : D80_PTV_treat, "D30_OAR" : D30_OAR_treat, "d_meanOAR" : mean_OAR_treat, "DMi_inRef_noisy" : DMi_inRef_noisy}
       with open(saving_path + "/" + name + "/testingepoch" + str(j) + ".pickle", 'wb') as handle:
              pickle.dump(dic, handle)
