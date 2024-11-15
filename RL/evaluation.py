import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.collections import LineCollection
from RL.dose_evaluation import *
from RL.target_contouring import *

def evaluate_over_treatment_daily_3D(model, env, n_eval_episodes=30, deterministic=True, epoch=0):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    time = np.zeros(n_eval_episodes)
    DM = np.zeros(n_eval_episodes)
    Accum_DM = np.zeros((env.depth ,env.num_row, env.num_col))
    d_meanOAR = np.zeros(n_eval_episodes)
    d_meanAutour = np.zeros(n_eval_episodes)
    d_meanPTV= np.zeros(n_eval_episodes)
    D98_PTV = np.zeros(n_eval_episodes)
    D80_PTV = np.zeros(n_eval_episodes)
    D30_OAR = np.zeros(n_eval_episodes)
    actions = np.zeros((n_eval_episodes,env.SignalLength,1))
    rewards = np.zeros((n_eval_episodes,env.SignalLength,1))
    real_pos = np.zeros((n_eval_episodes,env.SignalLength,3))
    beam_pos = np.zeros((n_eval_episodes,env.SignalLength,2))
    noisy_pos = np.zeros((n_eval_episodes,env.SignalLength,3))
    DM_PTV = np.zeros(n_eval_episodes)
    minim = np.zeros(n_eval_episodes)
    ref_position = [int(env.depth / 2), int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    DMi_inRef_noisy = np.zeros(((env.depth, env.num_row, env.num_col)))
    perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+2, form = env.form)
    perfectPTV_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    perfectDM_GTVinref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    perfectDM_inref = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    for i in range(n_eval_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        j = 0
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs, deterministic=deterministic)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            actions[i,j] = action
            rewards[i,j] = reward
            real_pos[i,j] = env._tumor_position
            noisy_pos[i,j] = env.noisy_tumor_position
            beam_pos[i,j] = env._beam_position
            episode_rewards.append(reward)
            j = j + 1 
        
        time[i] = env.curTimeStep
        Accum_DM += env.DMi_inRef
        d_OAR = env.DMi_inRef[perfectDM_inref_eval!=2]
        d_PTV = env.DMi_inRef[env.PTV==1]
        AutourPTV = perfectDM_inref_eval - perfectPTV_inref_eval
        d_Autour = env.DMi_inRef[AutourPTV!=0]
        DMi_inRef_noisy += env.DMi_inRef_noisy

        d_meanOAR[i] = np.mean(d_OAR)
        d_meanAutour[i] = np.mean(d_Autour)
        d_meanPTV[i] = np.mean(d_PTV)
        D98_PTV[i] = computeDx(d_PTV, 98, 4)
        D80_PTV[i] = computeDx(d_PTV, 80, 4)
        D30_OAR[i] = computeDx(d_OAR,30, 4)
        diff = perfectDM_inref-env.DMi_inRef
        DM[i] = np.sum(np.abs(diff))
        DM_PTV[i] = np.sum(np.abs(diff[env.PTV==1]))
        all_episode_rewards.append(sum(episode_rewards))
        minim[i] = np.min(d_PTV)
    std = np.std(all_episode_rewards)
    mean_episode_reward = np.mean(all_episode_rewards)
    perfect_q = n_eval_episodes*2
    d_PTV_all = Accum_DM[env.PTV==1] 
    d_GTV_all = Accum_DM[perfectDM_GTVinref==2.0]
    d30_OAR_all = Accum_DM[perfectDM_inref_eval==0]
    D98_PTV_treat = computeDx(d_PTV_all, 98, perfect_q+50)
    D98_GTV_treat = computeDx(d_GTV_all, 98, perfect_q+50)
    D80_PTV_treat = computeDx(d_PTV_all, 80, perfect_q+50)
    D30_PTV_treat = computeDx(d_PTV_all, 30, perfect_q+50)
    D30_OAR_treat = computeDx(d30_OAR_all, 30, perfect_q+50)
    mean_OAR_treat = np.mean(d30_OAR_all)
    maxPTVs = np.max(Accum_DM[(perfectDM_inref_eval-perfectPTV_inref_eval)==2.0])
    maxPTV = np.max(d_PTV_all)

    return mean_episode_reward, std, np.mean(time), np.mean(DM), np.mean(DM_PTV), np.mean(d_meanPTV), np.mean(d_meanOAR), np.mean(d_meanAutour), np.mean(D98_PTV), np.mean(D80_PTV), np.mean(D30_OAR), np.mean(minim), D98_PTV_treat, D80_PTV_treat, D30_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, actions, real_pos, noisy_pos, beam_pos, rewards, DMi_inRef_noisy, D98_GTV_treat, maxPTVs, maxPTV


def plot_treatment_3D(env, n_eval_episodes, D98_PTV_treat, D80_PTV_treat, D30_OAR_treat, mean_OAR_treat, Accum_DM, epoch):
    ref_position = [int(env.depth / 2),int(env.num_row / 2),int(env.num_col / 2)] #AJOUTTTTER
    perfectDM_inref_eval = env.observeEnvAs2DImage(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    fig = plt.figure()
    plt.suptitle("D98_PTV : " + str(np.round(D98_PTV_treat,2)) +"D80_PTV : " + str(np.round(D80_PTV_treat,2)) + "   D30_OAR : " + str(np.round(D30_OAR_treat,2)) + "    d_meanOAR : " + str(np.round(mean_OAR_treat,2)))
    
    ax5 = fig.add_subplot(2, 3, 1)
    shw0 = ax5.imshow(-((n_eval_episodes*env.perfectDM_inref)-Accum_DM)[int(env.depth/2)], cmap='bwr',vmin = -25, vmax=25, interpolation = "gaussian")
    plot_outlines(env.PTV[int(env.depth/2)].T, ax=ax5, lw=1.4, color='yellowgreen')
    ax5.set_title("(A) DoseMaps Difference")

    ax2 = fig.add_subplot(2,3,2)
    shw0 = ax2.imshow(-((n_eval_episodes*2*env.PTV)-Accum_DM)[int(env.depth/2)], cmap='bwr',vmin = -25, vmax=25, interpolation = "gaussian")
    plot_outlines(env.PTV[int(env.depth/2)].T, ax=ax2, lw=1.4, color='yellowgreen')
    ax2.set_title("(B) Medical objective")
    perfect_q = n_eval_episodes*2
    ax2 = fig.add_subplot(2,3,3)
    shw1 = ax2.imshow(env.PTV[int(env.depth/2)], cmap='gray')
    maxDose = n_eval_episodes*env.dose_quantity
    shw2 = ax2.imshow(Accum_DM[int(env.depth/2)], cmap='jet', alpha=0.4*(Accum_DM[int(env.depth/2)] > 0), vmin=0, vmax=perfect_q+20, interpolation = "gaussian")
    ax2.set_title("(C) Accumulated dose")
    
    ax6 = fig.add_subplot(2,1,2)
    perfectDM_inref_eval = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize+2, form = env.form)
    perfectDM_PTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize, form = env.form)
    perfectDM_AroundTumor_inref = perfectDM_inref_eval-perfectDM_PTV_inref
    perfectDM_GTV_inref = env.observeEnvAs2DImage_plain(pos = ref_position, dose_q=2.0, targetSize = env.targetSize-1, form = env.form)
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_PTV_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="PTV", color="yellowgreen")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_GTV_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="GTV", color="orange")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH(Accum_DM, perfectDM_AroundTumor_inref, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="HPV", color="steelblue")
    bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(Accum_DM, perfectDM_inref_eval, perfect_q + 30, 300)
    spl = CubicSpline(bin_edges, dvh)
    ax6.plot(bin_edges_interpolate, spl(bin_edges_interpolate), label="OAR", color="firebrick")
    ax6.axvline(n_eval_episodes*2, linestyle='dashed', label="Dose obj.")
    ax6.set_xlabel("Dose deposited (Gy)")
    ax6.set_ylabel("Pourcentage of volume (%)")
    ax6.set_title("(D) Dose Volume Histogram of the different zones")
    plt.ylim([-5, 105])
    ax6.legend(loc='upper right', labelspacing=0.15)
    fig.tight_layout()
    plt.savefig(env.saving_path + "/" + env.name + "/final_epoch_all_treat" +str(epoch)+".png")
    plt.close
    return 0