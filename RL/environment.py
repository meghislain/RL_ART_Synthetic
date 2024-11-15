from typing import Dict, Union

import gym
import os
import sys
import random
import numpy as np
from gym import spaces
from skimage.morphology import disk, square, rectangle, cube, octahedron, ball, octagon, star
import random
import wandb
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter
from stable_baselines3.common.type_aliases import GymStepReturn
from RL.trajectory import create_breathing_signals_reel_3D
from scipy.interpolate import CubicSpline
from RL.dose_evaluation import compute_DVH, compute_DVH_OAR
from scipy.ndimage import gaussian_filter
from RL.kalman import KF, EKF


class TumorEnv(gym.Env):
    """
    Base class for GridWorld-based MultiObsJ Environments 4x4  grid world.

    """
    
    def __init__(
        self,
        num_col: int = 12,
        num_row: int = 12,
        size: int = 8,
        target_size: int = 2,
        n_test_episode: int = 5,
        n_train_episode: int = 40,
        n_epoch: int = 30,
        signal_length: int = 75,
        dose_quantity: float = 1.0,
        name: str = "1",
        saving_path: str = "None",
        mode: int = -1,
        discrete_actions: bool = True,
        channel_last: bool = True,
        breathingSignal: str = "None",
        save_gif: int = 1,
        moving: int = 1,
        amplitude: float = 2,
        form: str = "ball",
        frequency : int = 10,
        le : int = 2,
        inc : int = 1
    ):
        super().__init__()

        self.mode = mode
        self.save_gif = save_gif
        self.targetSize = target_size
        self.n_obs = 3
        self.saving_path = saving_path
        self.name = name
        self.moving = moving
        self.amplitude = amplitude
        self.time_to_shuffle = 0
        self.count_time_shuffle = 0
        self.form = form
        if mode == 1 :
          self.targetSize = target_size-1
          self.form = "cube"
        self.inc = inc
        self.frequency = frequency
        self.gauss = self.gkern(long = (2*le) +1, sig = 0.5)
        print(self.gauss)
        maximum_gauss = np.max(self.gauss)
        rapport = 1/maximum_gauss
        self.distr_tir = self.gauss*rapport
        self.le = int(np.floor(len(self.gauss)/2))
        self.dose_deposition = True
        self.bragg = np.array([0.23411258, 0.23821531, 0.23805338, 0.24165836, 0.24149691, 0.244378,
                               0.24738092, 0.24038384, 0.24338676, 0.24702656, 0.25066838, 0.2543102,
                               0.24411258, 0.24821531, 0.24805338, 0.25165836, 0.25149691, 0.254378,
                               0.25738092, 0.26038384, 0.26338676, 0.26702656, 0.27066838, 0.2743102,
                               0.27795202, 0.28279427, 0.28767036, 0.29254644, 0.29749069, 0.30858278,
                               0.31967487, 0.33076696, 0.34185905, 0.36314137, 0.38494948, 0.43223005,
                               0.48097867, 0.59482089, 0.99684796])
        if save_gif :
            if not os.path.exists(self.saving_path + "/" + self.name):
                 os.umask(0)
                 os.makedirs(self.saving_path + "/" + self.name) # Create a new directory because it does not exist
                 print("New directory created to save the data: ", self.saving_path + "/" + self.name)

        self.zone = 2 # zone considered around the tumor for the accumulated dosemap
        self.discrete_actions = discrete_actions
        self.SignalLength=int(signal_length)
        if discrete_actions:
            self.action_space = spaces.Discrete(5 + 1 + 1)
        else:
            self.action_space = spaces.Box(0, 1, (5+ 1 + 1,))
        self.l = [2,1]
        if self.n_obs == 3 :
            self.img_size = [3, num_row, num_col]
            self.doseMaps = np.zeros(self.img_size, dtype=np.float64)
            self.observation_space = spaces.Dict(
            spaces={
                "doseMaps": spaces.Box(-1,1,self.img_size, dtype=np.float64)
            }
            )
        
        self.num_col = num_col
        self.num_row = num_row
        self.depth = self.num_row
        self.ref_position = [int(self.depth / 2),int(self.num_row / 2),int(self.num_col / 2)]
        self.beam_pos = np.zeros((num_row, num_col), dtype=np.float64)
        self.beam_pos[self.ref_position[1], self.ref_position[2]] = 1
        self.perfect_DM = np.zeros((self.depth, num_row, num_col), dtype=np.float64)
        self.DM_i = np.zeros((self.depth, num_row, num_col), dtype=np.float64)
        self.PTV = self.observeEnvAs2DImage_plain(pos = self.ref_position, dose_q=1, targetSize = self.targetSize, form = self.form)
        self.number_pixel_to_touch = len(self.PTV[self.PTV==1])
        self.mask = self.observeEnvAs2DImage(pos = self.ref_position, dose_q=1, targetSize = self.targetSize, form = self.form)
        self.incert_image = self.mask
        self.n_test_episode = n_test_episode
        self.n_train_episode = n_train_episode
        self.count = 0
        self.max_count = 100
        self.state = 0

        self.observation = []
        self._beam_position = np.zeros((2, 1), dtype=np.uint8)
        self.n_targets = 9
        self.n_dose = self.n_targets*dose_quantity
        self.n_energy = 5
        self.energy = int(self.depth/2) + self.n_energy 
        if self.n_obs == 3 :
            self.doseMaps[0] = self.DM_i[self.energy]
            self.doseMaps[2] = self.beam_pos
            self.doseMaps[1] = self.mask[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})
        
        if breathingSignal == "None":

            grid = np.ones((self.n_train_episode+self.n_test_episode+150,3))*self.ref_position
            
            np.random.seed(0);np.random.shuffle(grid)

            signal_matrice = create_breathing_signals_reel_3D(grid, self.amplitude, self.moving, self.SignalLength)
            self.breathingSignal_training = signal_matrice[int(self.n_test_episode):int(self.n_test_episode+self.n_train_episode)]
            self.breathingSignal_validation = signal_matrice[:int(self.n_test_episode)]
            self.breathingSignal_testing = signal_matrice[-150::]
            print(" training : ", self.breathingSignal_training[:,0,:], len(self.breathingSignal_training))
            print(" validation : ", self.breathingSignal_validation[:,0,:], len(self.breathingSignal_validation))
        
        self._beam_position = [int(self.num_row/ 2),int(self.num_col / 2)]
        self._beam_dx= self.targetSize + 1 
        self._beam_dy= self.targetSize + 1
        self.count_validation = 0
        self.count_testing = 0
        self.count_training = 0
        self.n_training = 0
        self.n_validation = 0
        self.n_testing = 0
        self.count_episode = 0
        self.count_epoch = 0
        self.action = 0
        self.dose_quantity = dose_quantity
        self.curTimeStep = 0
        self.kf_instance = KF()
        self.done = False
        

    def step(self, action):
        self.observation = []
        self._tumor_position = self._signal_position[self.curTimeStep]
        
        if self.inc == 1 :
            if self.curTimeStep % 10 == 0:
                self.noisy_signal_position[self.curTimeStep] = self._signal_position[self.curTimeStep]
                self.kf_instance.kf.predict() #self.kf.x, self.kf.P, self.kf.F, self.kf.Q
                self.kf_instance.kf.update(self._signal_position[self.curTimeStep][1:])
                self.noisy_tumor_position = np.array([self.ref_position[0],self.kf_instance.kf.x[0],self.kf_instance.kf.x[1]])
                self.noisy_pos[self.curTimeStep] = self.noisy_tumor_position
                self.mask_shifted = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=1, targetSize = self.targetSize, form = self.form)
                self.incert_image = self.mask_shifted
                self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
                 
            else :
                self.previous_noisy_position = self.noisy_tumor_position
                self.kf_instance.kf.predict()
                self.noisy_tumor_position = np.array([self.ref_position[0],self.kf_instance.kf.x[0],self.kf_instance.kf.x[1]])
                self.noisy_pos[self.curTimeStep] = self.noisy_tumor_position
                sigma = 0.3
                shift = np.zeros(3)
                shift[0] = 0
                shift[2] = self.noisy_tumor_position[2] - self.previous_noisy_position[2]
                shift[1] = self.noisy_tumor_position[1] - self.previous_noisy_position[1]
                self.noisy_perfect_DM = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
                self.incert_image = scipy.ndimage.shift(self.incert_image, shift, order=1)
                self.incert_image[int(self.depth/2)-self.targetSize:int(self.depth/2)+self.targetSize+1] = gaussian_filter(self.incert_image, sigma)[int(self.depth/2)-self.targetSize:int(self.depth/2)+self.targetSize+1]
        else :
            self.noisy_tumor_position = self._signal_position[self.curTimeStep]
            self.mask_shifted = self.observeEnvAs2DImage(pos = self.noisy_tumor_position, dose_q=1, targetSize = self.targetSize, form = self.form)
            self.incert_image = self.mask_shifted
            
        self.perfect_DM = self.observeEnvAs2DImage(pos = np.round(self._tumor_position), dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        sigma = 1.3
        self.DM_i = np.zeros((self.depth,self.num_row,self.num_col))
        shift = np.zeros(3)
        shift[0] = 0
        shift[2] = np.round(self._tumor_position[2]) - self.ref_position[2]
        shift[1] = np.round(self._tumor_position[1]) - self.ref_position[1]
        self.DM_i = scipy.ndimage.shift(self.DMi_inRef, shift, order=1)
        
        self.curTimeStep += 1
        reward_dose = 0
        reward_distance = 0
        reward = 0
        tir_inside = False
        tir = False
        
        if action == 0: # move to the left
            self._beam_position[0] = max(self._beam_position[0]-self._beam_dx, self.le)
        elif action == 1: # move upward
            self._beam_position[1] = min(self._beam_position[1]+self._beam_dy, self.num_row-1-self.le)
        elif action == 2: # move to the right
            self._beam_position[0] = min(self._beam_position[0]+self._beam_dx, self.num_col-1-self.le)
        elif action == 3: # move downward
            self._beam_position[1] = max(self._beam_position[1]-self._beam_dy, self.le)
        elif action == 4 : # shoot a gaussian dose
            tir = True
            x_ = int(np.round(self._beam_position[0]))
            y_ = int(np.round(self._beam_position[1]))
            if self.perfect_DM[self.energy,x_,y_]-self.DM_i[self.energy,x_,y_] > 0.0 :
                reward_dose += 1
            else :
                reward_dose -= 1
            if self.dose_deposition == True:
                dose = np.ones((self.energy+1,np.shape(self.gauss)[0],np.shape(self.gauss)[1]))*self.gauss
                dose = dose*self.bragg[-self.energy-1:, np.newaxis, np.newaxis]
                self.DM_i[:self.energy+1, x_ - self.le : x_ + self.le +1 , y_ - self.le : y_ + self.le + 1] += dose 
            else :
                self.DM_i[self.energy, x_ - self.le : x_ + self.le +1 , y_ - self.le : y_ + self.le + 1] += self.gauss
            
            x_in_ref = int(np.round(self._beam_position[0] - self._tumor_position[1])) + int(self.num_row/2)
            y_in_ref = int(np.round(self._beam_position[1] - self._tumor_position[2])) + int(self.num_col/2)
            if x_in_ref >= self.le and x_in_ref < (self.num_row-self.le) :
                if y_in_ref >= self.le and y_in_ref < (self.num_col-self.le) :
                    if self.dose_deposition == True:
                        self.DMi_inRef[:self.energy+1, x_in_ref - self.le : x_in_ref + self.le +1 , y_in_ref - self.le : y_in_ref + self.le + 1] += dose
                    else :
                        self.DMi_inRef[self.energy, x_in_ref - self.le : x_in_ref + self.le +1 , y_in_ref - self.le : y_in_ref + self.le + 1] += self.gauss
            x_in_ref_noisy = int(np.round(self._beam_position[0] - self.noisy_tumor_position[1])) + int(self.num_row/2)
            y_in_ref_noisy = int(np.round(self._beam_position[1] - self.noisy_tumor_position[2])) + int(self.num_col/2)
            if x_in_ref_noisy >= self.le and x_in_ref_noisy < (self.num_row-self.le) :
                if y_in_ref_noisy >= self.le and y_in_ref_noisy < (self.num_col-self.le) :
                    self.DMi_inRef_noisy[self.energy, x_in_ref_noisy, y_in_ref_noisy] += 1
        if action == 5 : 
            if len(self.PTV[self.PTV[self.energy] == 1]) == 0:
              reward_dose -= 0.1
            if len(self.PTV[self.PTV[self.energy] == 1]) != 0:
                self.per_recovery = len(self.DMi_inRef[(self.PTV[self.energy] == 1) & (self.DMi_inRef[self.energy] >= 1.8)])/len(self.PTV[self.PTV[self.energy] == 1])
                reward_dose -= 2*(1-self.per_recovery)
                reward_dose -= 0.1
            if self.energy == (int(self.depth/2) - self.n_energy) :
                reward_dose -= 3
                self.energy = int(self.depth/2) + self.n_energy
            else : 
                self.energy = self.energy-1
        self.distance = np.linalg.norm(self._beam_position - self._tumor_position[1:])
        if action != 5 and action != 4 and action != 6:
            if self.distance > self.targetSize +1 :
                reward_distance -= self.distance/(self.num_row/1.5)
            reward_distance -= 0.1
                
        self._beam_dx = self.targetSize + 1
        self._beam_dy = self.targetSize + 1

        self._last_ponctual_distance = np.linalg.norm(self._beam_position - self.noisy_tumor_position[1:])
        
        if self._last_ponctual_distance <= (self.targetSize + 1):
            self._beam_dx = 1 
            self._beam_dy = 1 
            
        self.done = bool(self.curTimeStep >= self.SignalLength or action == 6)
        
        reward = reward_distance + reward_dose  
        self.sum_reward += reward

        info = {}
        
        self.beam_pos = np.zeros((self.num_row,self.num_col), dtype=np.float64)
        self.beam_pos[self._beam_position[0],self._beam_position[1]] = 1
        if self.n_obs == 3 :
            self.doseMaps[0] = self.DMi_inRef_noisy[self.energy]/4
            self.doseMaps[2] = self.beam_pos
            self.doseMaps[1] = self.incert_image[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})
        return self.observation[0], reward, self.done, info

    def reset(self):
        self.noisy_pos = np.zeros((self.SignalLength,3))
        self.observation = []
        self.curTimeStep = 0
        self.count_episode += 1
        if self.mode == -1:
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle):
                self.count_time_shuffle += 1
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle) and self.count_time_shuffle == 1: 
                np.random.shuffle(self.breathingSignal_training)
                self.n_training = 0
            if (int(self.count_training % self.n_train_episode)==self.time_to_shuffle) and self.count_time_shuffle == 2 :
                if self.time_to_shuffle == (self.n_train_episode-1):
                    self.time_to_shuffle = 0
                else :
                    self.time_to_shuffle += 1
                self.count_time_shuffle = 0
            self.positions = self.breathingSignal_training
            self._signal_position = self.positions[int(self.count_training % self.n_train_episode)]
            print('training mode : ', self.n_training, self._tumor_position)
            self.count_training += 1
            self.n_training += 1
        elif self.mode == 0:
            if int(self.n_validation % self.n_test_episode) == 0 :
                np.random.shuffle(self.breathingSignal_validation)
            self.positions = self.breathingSignal_validation
            self._signal_position = self.positions[int(self.n_validation % self.n_test_episode)]
            print('validation mode : ' + str(self.count_validation), self._tumor_position)
            self.count_validation += 1
            self.n_validation += 1
        else :
            self.positions = self.breathingSignal_testing
            self._signal_position = self.positions[int(self.n_testing % 150)]
            print('testing mode : ' + str(self.count_testing), self._tumor_position)
            self.count_testing += 1
            self.n_testing += 1

        self.sum_reward = 0
        self.positions = self.breathingSignal_validation
        self._signal_position = self.positions[int(self.n_validation % self.n_test_episode)]
        self._tumor_position = self._signal_position[self.curTimeStep]
        self.noisy_signal_position = self._signal_position 
        self.noisy_tumor_position = self.noisy_signal_position[self.curTimeStep]

        self.perfect_DM = self.observeEnvAs2DImage(pos = np.round(self._tumor_position), dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        sigma = 1.5
        self.n_dose = np.sum(self.perfect_DM)
        self._targets_position = np.argwhere(self.perfect_DM!=0)
        self.n_targets = len(self._targets_position)
        ref_position = [int(self.depth / 2),int(self.num_row / 2),int(self.num_col / 2)]
        self.DM_i = np.zeros((self.depth,self.num_row,self.num_col))
        self.DMi_inRef = np.zeros((self.depth,self.num_row,self.num_col))
        self.DMi_inRef_noisy = np.zeros((self.depth,self.num_row,self.num_col))
        self.perfectDM_inref = self.observeEnvAs2DImage(pos = ref_position, dose_q=self.dose_quantity, targetSize = self.targetSize, form = self.form)
        self._beam_dx= self.targetSize + 1
        self._beam_dy= self.targetSize + 1
        self._beam_position = np.array((int(self.num_row / 2), int(self.num_col / 2)))
        self.beam_pos = np.zeros((self.num_row,self.num_col), dtype=np.float64)
        self.beam_pos[self._beam_position[0],self._beam_position[1]] = 1
        self.kf_instance = KF()
        self.mask_shifted = self.observeEnvAs2DImage(pos = self._tumor_position, dose_q=1, targetSize = self.targetSize, form = self.form)
        self.energy = int(self.depth/2) + self.n_energy
        if self.n_obs == 3 :
            self.doseMaps[0] = self.DMi_inRef_noisy[self.energy]/4
            self.doseMaps[2] = self.beam_pos
            self.doseMaps[1] = self.mask_shifted[self.energy]
            self.observation.append({"doseMaps": self.doseMaps.reshape(self.img_size)})
        return self.observation[0] 
    
    def observeEnvAs2DImage(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
        """
        The noise thing is a work in progress
        """

        envImg1 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
        
        if self.form == "cube":
            target = (dose_q/2)*cube(2*(targetSize+2) + 1) 
        if self.form == "ball":
            target = (dose_q/2)*ball(targetSize+2)
        if pos is not None:
            targetCenter = ref_position
        targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
        
        envImg2 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        if self.form == "cube":
            target = (dose_q)*cube(2*(targetSize+1) + 1) 
        if self.form == "ball":
            target = (dose_q)*ball(targetSize+1)
        if pos is not None:
            targetCenter = ref_position
        envImg2[targetCenterInPixels[0] - int(targetSize): targetCenterInPixels[0] + int(targetSize + 1),
                targetCenterInPixels[1] - int(targetSize+1): targetCenterInPixels[1] + int(targetSize + 2),
                targetCenterInPixels[2] - int(targetSize+1): targetCenterInPixels[2] + int(targetSize + 2)] = target[1:-1,:,:]
        
        envImg = envImg1+envImg2 
        shift = np.zeros(3)
        shift[0] = pos[0] - targetCenterInPixels[0]
        shift[1] = pos[1] - targetCenterInPixels[1]
        shift[2] = pos[2] - targetCenterInPixels[2]
        envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
        
        return envImg_shifted
            
    def observeEnvAs2DImage_plain(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
        """
        The noise thing is a work in progress
        """
        envImg = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
        ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
        
        if self.form == "cube":
            target = (dose_q)*cube(2*targetSize + 1)
        if self.form == "ball":
            target = (dose_q)*ball(targetSize)
        if pos is not None:
            targetCenter = ref_position
        targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
        envImg[targetCenterInPixels[0] - int(targetSize): targetCenterInPixels[0] + int(targetSize + 1),
                targetCenterInPixels[1] - int(targetSize): targetCenterInPixels[1] + int(targetSize + 1),
                targetCenterInPixels[2] - int(targetSize): targetCenterInPixels[2] + int(targetSize + 1)] = target
        shift = np.zeros(3)
        shift[0] = pos[0] - targetCenterInPixels[0]
        shift[1] = pos[1] - targetCenterInPixels[1]
        shift[2] = pos[2] - targetCenterInPixels[2]
        envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
        
        return envImg_shifted
    
    def gkern(self, long, sig):
        """
        creates gaussian kernel with side length `long` and a sigma of `sig`
        """
        ax = np.linspace(-(long - 1) / 2., (long - 1) / 2., long)
        gauss = np.exp(-0.25 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return 2*kernel / np.sum(kernel)