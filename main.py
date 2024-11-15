from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, PPO
import sys
import torch as th
from RL.evaluation import *
from RL.network import Custom_CombinedExtractor, Custom_CNN
from RL.parser import parameters
from stable_baselines3.common.utils import get_linear_fn
import pickle

import os
import logging

import wandb
os.environ["WANDB__SERVICE_WAIT"] = "300"

import numpy as np
sav = ""

if parameters.dim =="3Dsansdose" :
       from ARIES.environment import TumorEnv 
       saving_path = sav + "/Results"

# Instantiate the env
name = parameters.date +"_DQ" +str(parameters.dq) +"_tarSize"+str(parameters.ts)+"_SL" + str(parameters.sl)
num_col = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position
num_row = 3 + (2*parameters.ts) + (2*parameters.ampl) + (2*parameters.gk) + 15 #random noise on tumor position

n_value = []
n_cpu_cores = 2
th.set_num_threads(n_cpu_cores)
wandb.run.name = name

env = TumorEnv(n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc, initial_energy = parameters.initial_energy)
env_eval = TumorEnv(mode=0, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc, initial_energy = parameters.initial_energy)
env_test = TumorEnv(mode=1, n_test_episode=parameters.n_test, n_train_episode=parameters.n_train, signal_length = parameters.sl, saving_path = saving_path, name = name, save_gif = parameters.savgif, target_size=parameters.ts, dose_quantity=parameters.dq, num_col=num_col, num_row=num_row, moving = parameters.mov, amplitude = parameters.ampl, form = parameters.form, frequency=parameters.freq, le=parameters.gk, inc = parameters.inc, initial_energy = parameters.initial_energy)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 64, 32], normalize_images=False, features_extractor_class=Custom_CombinedExtractor)

if parameters.expl_rate != 1.0 :
       model = DQN("MultiInputPolicy", env, verbose=0, learning_rate=parameters.lr, batch_size=16, device=parameters.dev, policy_kwargs=policy_kwargs, exploration_fraction=0.8, exploration_initial_eps=parameters.eps, exploration_final_eps=parameters.eps, target_update_interval=parameters.sl)
else : 
       model = DQN("MultiInputPolicy", env, verbose=0, learning_rate=parameters.lr, batch_size=16, device=parameters.dev, policy_kwargs=policy_kwargs, exploration_fraction=0.8, exploration_initial_eps=parameters.eps, exploration_final_eps=0.3, target_update_interval=parameters.sl)

TIMESTEPS = parameters.sl*parameters.n_train
print(model.policy)
# definition of the number of epoch with the same accuracy in the dosimetric parameters for the early stopping
patience = 10 
for i in range(parameters.n_epoch):
       print("epoch number : ", str(i))

       model.learn(total_timesteps=TIMESTEPS)
       mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV = evaluate_over_treatment_daily_3D(model, env_eval, n_eval_episodes=parameters.n_test, epoch=i)
        
       if i % 50 == 0 :
              _ = plot_treatment_3D(env_eval,n_eval_episodes=parameters.n_test, D98_PTV_treat=D98_PTV_treat, D80_PTV_treat=D80_PTV_treat, D30_OAR_treat=D30_OAR_treat, mean_OAR_treat=mean_OAR_treat, Accum_DM=Accum_DM, epoch=i)
        
       model.learning_rate = model.learning_rate*parameters.fct
       if parameters.expl_rate != 1.0 : 
              model.exploration_initial_eps = model.exploration_initial_eps*parameters.expl_rate 
              model.exploration_final_eps = model.exploration_final_eps*parameters.expl_rate 
              model.exploration_schedule = get_linear_fn(
              model.exploration_initial_eps,
              model.exploration_final_eps,
              model.exploration_fraction, 
              )
        
       if (D80_PTV_treat >= 60) and (D98_GTV_treat >= 55) and (D98_PTV_treat >= 50) and (D30_PTV_treat <= 78) and (maxPTVs <= 85) and (maxPTV <= 95):
              if n == patience :
                     print("Early stopping at epoch ", i)
                     print(n_value)
                     for j in range(5):
                            mean_reward, std_reward, time, DM, DM_PTV, d_meanPTV, d_meanOAR, d_meanAutour, D98_PTV, D80_PTV, D30_OAR, mini, D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV = evaluate_over_treatment_daily_3D(model, env_test, n_eval_episodes=parameters.n_test, epoch=j)
                            dic = {"Accumulation Dose": Accum_DM, "actions": actions, "Beam position": beam_pos, 
                            "real tumor position": real_pos, "noisy tumor position": noisy_pos, "target size": parameters.ts, "form" : parameters.form, "reward" : rewards, "Signal Length" : parameters.sl,"D98_PTV" : D98_PTV_treat, "D80_PTV" : D80_PTV_treat, "D30_OAR" : D30_OAR_treat, "d_meanOAR" : mean_OAR_treat, "DMi_inRef_noisy" : DMi_inRef_noisy}
                            with open(saving_path + "/" + name + "/testingepoch" + str(j) + ".pickle", 'wb') as handle:
                            pickle.dump(dic, handle)
                     model.save(sav + "/Model")
                     break
       else :
              n = 0
