# -*- coding: utf-8 -*-
"""
Visualization code for assessing dose deposition in target and surrounding zones in reinforcement learning framework
for real-time proton therapy treatment of mobile tumors.

Description:
    - Show dose deposition profiles within defined areas: GTV, target volume (PTV), peritumoral volume (HPV), 
      and other organs at risk (OAR).
    - Uses synthetic 3D data for testing and visualization purposes.
    
Author: meghislain
Created on: Sat Nov 9 20:56:18 2024
"""

import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import cube, ball
import pickle
import scipy
from scipy.ndimage import shift
from scipy.interpolate import CubicSpline
from matplotlib.collections import LineCollection
from RL.dose_evaluation import computeDx, compute_DVH, compute_DVH_OAR
from RL.target_contouring import plot_outlines

matplotlib.rcParams['interactive'] = True

# Visualization grid parameters
depth = 34
num_row, num_col = depth, depth

    
def observeEnvAs2DImage( pos = None, dose_q = 1, form = "ball", targetSize = 2):

     """
    Generate a 3D image of the environment with objective dose applied to the target and a small part of its surrounding volume.
    Generate the attended dose map for one fraction

    Parameters:
        pos (list): Position of the dose center.
        dose_q (float): Dose quantity.
        form (str): Shape of the target ('cube' or 'ball').
        targetSize (int): Radius of the target in voxels.

    Returns:
        envImg_shifted (ndarray): Shifted 3D environment image with dose deposition.
    """
    envImg = np.zeros((depth, num_row, num_col), dtype=np.float64)
    ref_position = [depth // 2, num_row // 2, num_col // 2]

    targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
        
    if  form == "cube":
        target = (dose_q)*cube(2*(targetSize +1) + 1) 
    if  form == "ball":
        target = (dose_q)*ball(targetSize+1)
    if pos is not None:
        targetCenter = ref_position
    envImg[targetCenterInPixels[0] - int(targetSize+1): targetCenterInPixels[0] + int(targetSize + 2),
            targetCenterInPixels[1] - int(targetSize+1): targetCenterInPixels[1] + int(targetSize + 2),
            targetCenterInPixels[2] - int(targetSize+1): targetCenterInPixels[2] + int(targetSize + 2)] = target
    
    
    envImg = envImg
    shift = np.zeros(3)
    shift[0] = pos[0] - targetCenterInPixels[0]
    shift[1] = pos[1] - targetCenterInPixels[1]
    shift[2] = pos[2] - targetCenterInPixels[2]
    envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
    
    return envImg_shifted
            
def observeEnvAs2DImage_plain(pos = None, dose_q = 1, form = "ball", targetSize = 2):

        envImg = np.zeros(( depth,  num_row,  num_col), np.float64)
        ref_position = [int( depth / 2), int( num_row / 2), int( num_col / 2)]
        
        if  form == "cube":
            target = (dose_q)*cube(2*targetSize + 1)
        if  form == "ball":
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

#-----------------------------------------------------------------------------------
    
dose_quantity = 2
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['lines.linewidth'] = 2
import math 

n_eval_episodes = 30
maxDose =  (dose_quantity*n_eval_episodes) + 30

#--------------------------------------------------------------------------------------
# COMPUTE DOSIMETRIC VARIABLES
#---------------------------------------------------------------------------------------

base_path = ""
file_extension = ".pickle"
file_number = 5

FILE = [base_path + str(i) + file_extension for i in range(10)]
GTV = np.zeros((file_number,10000))
PTV = np.zeros((file_number,10000))
PTVs = np.zeros((file_number,10000))
OAR = np.zeros((file_number,100000))
GTV_metrics = np.zeros((file_number,6))
PTV_metrics = np.zeros((file_number,6))
PTVs_metrics = np.zeros((file_number,6))
OAR_metrics = np.zeros((file_number,6))


targetSize = 2
form = "cube"
DM_i = np.zeros((depth,num_row,num_col))
DM_i_noisy = np.zeros((depth,num_row,num_col))
DMi_inRef = np.zeros((depth,num_row,num_col))
DMi_inRef_noisy = np.zeros((depth,num_row,num_col))
ref_position = [int( depth / 2),int( num_row / 2), int( num_col / 2)]
PTV_ = observeEnvAs2DImage_plain(pos = ref_position, dose_q= dose_quantity, targetSize =  targetSize, form =  form)
perfectDM_inref_code =  observeEnvAs2DImage(pos = ref_position, dose_q= dose_quantity, targetSize =  targetSize, form =  form)
perfectDM_inref =  observeEnvAs2DImage_plain(pos = ref_position, dose_q= dose_quantity, targetSize =  targetSize+2, form =  form)
perfectDM_PTV_inref =  observeEnvAs2DImage_plain(pos = ref_position, dose_q= dose_quantity, targetSize =  targetSize, form =  form)
perfectDM_GTV_inref =  observeEnvAs2DImage_plain(pos = ref_position, dose_q= dose_quantity, targetSize =  targetSize-1, form =  form)
perfectDM_AroundTumor_inref = perfectDM_inref-perfectDM_PTV_inref


for j in range(file_number):
    
    with open(FILE[j], 'rb') as f:
        # Load the data from the file
        my_loaded_dict = pickle.load(f)

    Accum_DM = my_loaded_dict["Accumulation Dose"] 
    
    d_PTV_all = Accum_DM[perfectDM_PTV_inref==dose_quantity] 
    perfect_q = 110
    PTV_metrics[j,0] = computeDx(d_PTV_all, 98, perfect_q)
    PTV_metrics[j,1] = computeDx(d_PTV_all, 80, perfect_q)
    D2_PTV_treat = computeDx(d_PTV_all, 2, perfect_q)
    D50_PTV_treat = computeDx(d_PTV_all, 50, perfect_q)
    PTV_metrics[j,3] = np.min(d_PTV_all)
    PTV_metrics[j,4] = np.max(d_PTV_all)
    PTV_metrics[j,5] = (D2_PTV_treat-PTV_metrics[j,0])/D50_PTV_treat
    PTV_metrics[j,2] = np.mean(d_PTV_all)
    
    print("GTV")
    d_PTV_all = Accum_DM[perfectDM_GTV_inref==dose_quantity]
    GTV_metrics[j,0] = computeDx(d_PTV_all, 98, perfect_q)
    GTV_metrics[j,1] = computeDx(d_PTV_all, 80, perfect_q)
    D2_PTV_treat = computeDx(d_PTV_all, 2, perfect_q)
    D50_PTV_treat = computeDx(d_PTV_all, 50, perfect_q)
    GTV_metrics[j,3] = np.min(d_PTV_all)
    GTV_metrics[j,4] = np.max(d_PTV_all)
    GTV_metrics[j,5] = (D2_PTV_treat-GTV_metrics[j,0])/D50_PTV_treat
    GTV_metrics[j,2] = np.mean(d_PTV_all)
    
    print("PTVs")
    d_PTV_all = Accum_DM[perfectDM_AroundTumor_inref==dose_quantity]
    PTVs_metrics[j,0] = computeDx(d_PTV_all, 98, perfect_q)
    PTVs_metrics[j,1] = computeDx(d_PTV_all, 80, perfect_q)
    D2_PTVs_treat = computeDx(d_PTV_all, 2, perfect_q)
    D50_PTVs_treat = computeDx(d_PTV_all, 50, perfect_q)
    PTVs_metrics[j,3] = np.min(d_PTV_all)
    PTVs_metrics[j,4] = np.max(d_PTV_all)
    PTVs_metrics[j,5] = (D2_PTVs_treat-PTVs_metrics[j,0])/D50_PTVs_treat
    PTVs_metrics[j,2] = np.mean(d_PTV_all)
    
    print("OAR")
    d_PTV_all = Accum_DM[perfectDM_inref==0]
    OAR_metrics[j,0] = computeDx(d_PTV_all, 98, perfect_q)
    OAR_metrics[j,1] = computeDx(d_PTV_all, 80, perfect_q)
    D2_PTV_treat = computeDx(d_PTV_all, 2, perfect_q)
    D50_PTV_treat = computeDx(d_PTV_all, 50, perfect_q)
    OAR_metrics[j,3] = np.min(d_PTV_all)
    OAR_metrics[j,4] = np.max(d_PTV_all)
    OAR_metrics[j,2] = np.mean(d_PTV_all)
    colors = [(1., 1., 1.),(0.8, 0.8, 0.8), (0., 0., 0.5),(0., 0.25, 1),(0.1,1,0.8), (0.3,1,0.3), (1,1,0), (1, 0, 0), (0.3, 0, 0)]  # Gris clair à Bleu à Rouge foncé
    
    
    bin_edges,dvh, bin_edges_interpolated = compute_DVH(Accum_DM, perfectDM_GTV_inref, maxDose, 1000)
    spl = CubicSpline(bin_edges, dvh)
    GTV[j] = spl(bin_edges_interpolated)
    bin_edges,dvh, bin_edges_interpolated = compute_DVH(Accum_DM, perfectDM_PTV_inref, maxDose, 1000)
    spl = CubicSpline(bin_edges, dvh)
    PTV[j] = spl(bin_edges_interpolated)
    bin_edges,dvh, bin_edges_interpolated = compute_DVH(Accum_DM, perfectDM_AroundTumor_inref, maxDose, 1000)
    spl = CubicSpline(bin_edges, dvh)
    PTVs[j] = spl(bin_edges_interpolated)
    bin_edges,dvh, bin_edges_interpolate = compute_DVH_OAR(Accum_DM, perfectDM_inref, maxDose, 10000)
    spl = CubicSpline(bin_edges, dvh)
    OAR[j] = spl(bin_edges_interpolate)
    
    
import matplotlib as mpl
cmap = mpl.colors.LinearSegmentedColormap.from_list('gris_clair_bleu_rouge_foncé_continu', colors)
 
# Définition de la norme pour la colormap
norm = mpl.colors.Normalize(vmin=0, vmax=75) 
# Créer la figure et les axes (subplots)
fig = plt.figure()  # Ajustez la taille de la figure selon vos besoins
for i in range(-14,10):
    ax = fig.add_subplot(4, 6, i+15)
    targetSize = 3
    sh = ax.imshow(Accum_DM[int(depth/2)-targetSize+i], cmap=cmap, vmin=0, vmax=75, interpolation = "gaussian", alpha=0.4*(Accum_DM[int(depth/2)-targetSize+i]>=0))#,facecolors='jet',edgecolor='k')
    if i == -9 or i == -3 or i == 3 or i == 9 or i == 19:
        custom_ticks = [0,10, 20,30, 40,50,60,70]
        cbar = plt.colorbar(sh, ax=ax, shrink=0.7)
        cbar.set_ticks(custom_ticks)
        cbar.ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_ylim([24,10])
    ax.set_xlim([10,24])
    ax.set_title('Dose on slice ' + str(int(depth/2)-targetSize +i), fontname='Times New Roman', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    for tick in ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontname('Times New Roman')    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plot_outlines(PTV_[int(depth/2)-targetSize+i].T, ax=ax, lw=3.4, color='k')
    plot_outlines(perfectDM_inref[int(depth/2)-targetSize+i].T, ax=ax, lw=1.4, color='k')
plt.tight_layout()
plt.show()

fig = plt.figure()
maxDose = (dose_quantity*n_eval_episodes) + 30
ax = fig.add_subplot(1,1, 1)
GTV_mean = np.mean(GTV,axis=0)
GTV_plus = GTV_mean + np.std(GTV,axis=0)
GTV_moins = GTV_mean - np.std(GTV,axis=0)
ax.plot(bin_edges_interpolated, np.mean(GTV,axis=0), linestyle="-", label="GTV DVH", color="orange")
ax.fill_between(bin_edges_interpolated, GTV_moins,GTV_plus, alpha=0.2, color="orange")
PTV_mean = np.mean(PTV,axis=0)
PTV_moins = PTV_mean + np.std(PTV,axis=0)
PTV_plus = PTV_mean - np.std(PTV,axis=0)
ax.plot(bin_edges_interpolated, PTV_mean, linestyle="-", label="PTV DVH", color="yellowgreen")
ax.fill_between(bin_edges_interpolated, PTV_moins,PTV_plus, alpha=0.2, color="yellowgreen")
x1 = np.linspace(0, 60, 1000)
y1 = np.full_like(x1, 100)
# After x = 60, y drops to 0 and stays 0
x1 = np.append(x1, 60)
y1 = np.append(y1, 0)
x1 = np.append(x1, np.linspace(60, 90, 1000))
y1 = np.append(y1, np.zeros(1000))

# Plot the first line
ax.plot(x1, y1, c= "k", linestyle=":", label='Ideal GTV-PTV DVH')

ax.plot(bin_edges_interpolated, np.mean(PTVs,axis=0), linestyle = "-", label="HPV DVH", color="steelblue")
PTVs_mean = np.mean(PTVs,axis=0)
PTVs_moins = PTVs_mean + np.std(PTVs,axis=0)
PTVs_plus = PTVs_mean - np.std(PTVs,axis=0)
ax.fill_between(bin_edges_interpolated, PTVs_moins,PTVs_plus, alpha=0.2, color="steelblue")
ax.plot(bin_edges_interpolate, np.mean(OAR,axis=0),linestyle = "-", label="OAR DVH", color="firebrick")
OAR_mean = np.mean(OAR, axis=0)
OAR_moins = OAR_mean + np.std(OAR,axis=0)
OAR_plus = OAR_mean - np.std(OAR,axis=0)
ax.fill_between(bin_edges_interpolate, OAR_moins,OAR_plus, alpha=0.2, color="firebrick")
# Define the second line
# From y = 100 to y = 0 at x = 0, then y = 0 for x > 0
x2 = np.array([0, 0, 90])
y2 = np.array([100, 0, 0])

# Plot the second line
ax.plot(x2, y2, c="k", linestyle = "--", label='Ideal HPV-OAR DVH')
ax.set_xlabel("Dose deposited (Gy)", fontname='Times New Roman', fontsize=25)
ax.set_ylabel("Percentage of volume (%)", fontname='Times New Roman', fontsize=25)
ax.set_title("Dose Volume Histogram of the different zones", fontname='Times New Roman', fontsize=25)

plt.ylim([-5, 105])
# Add a legend outside the plot
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), labelspacing=0.15, prop={'family': 'Times New Roman', 'size': 25})
ax.tick_params(axis='both', labelsize=20)

# Add a title above the legend
ax.tick_params(axis='both', labelsize=20)
for tick in ax.get_xticklabels():
    tick.set_fontname('Times New Roman')
for tick in ax.get_yticklabels():
    tick.set_fontname('Times New Roman')

# Ajuster l'espacement entre les subplots pour éviter les chevauchements
plt.tight_layout()
# Afficher la figure
plt.show()          


# -*- coding: utf-8 -*-
"""
Visualization code for assessing dose deposition in target and surrounding zones in reinforcement learning framework
for real-time proton therapy treatment of mobile tumors.

Description:
    - Simulates dose deposition profiles within defined areas: target volume (PTV), peritumoral volume (HPV), 
      and other organs at risk (OAR).
    - Uses synthetic 3D data for testing and visualization purposes.
    
Author: meghislain
Created on: Sat Nov 9 20:56:18 2024
"""

import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import cube, ball
import pickle
import scipy
from scipy.ndimage import shift
from scipy.interpolate import CubicSpline
from matplotlib.collections import LineCollection
from RL.dose_evaluation import computeDx, compute_DVH, compute_DVH_OAR
from RL.target_contouring import plot_outlines

matplotlib.rcParams['interactive'] = True

# Visualization grid parameters
depth = 34
num_row, num_col = depth, depth

def observeEnvAs2DImage(pos=None, dose_q=1, form="ball", targetSize=2):
    """
    Generate a 3D image of the environment with dose applied to the target and its surrounding volume.

    Parameters:
        pos (list): Position of the dose center.
        dose_q (float): Dose quantity.
        form (str): Shape of the target ('cube' or 'ball').
        targetSize (int): Radius of the target in voxels.

    Returns:
        envImg_shifted (ndarray): Shifted 3D environment image with dose deposition.
    """
    envImg1 = np.zeros((depth, num_row, num_col), dtype=np.float64)
    envImg2 = np.zeros((depth, num_row, num_col), dtype=np.float64)
    ref_position = [depth // 2, num_row // 2, num_col // 2]

    # Define surrounding dose distribution (HPV)
    if form == "cube":
        target = (dose_q / 2) * cube(2 * (targetSize + 2) + 1)
    elif form == "ball":
        target = (dose_q / 2) * ball(targetSize + 2)
    
    targetCenterInPixels = np.array(ref_position, dtype=int)
    # Define target dose distribution (PTV)
    if form == "cube":
        target = dose_q * cube(2 * (targetSize + 1) + 1)
    elif form == "ball":
        target = dose_q * ball(targetSize + 1)

    envImg2[targetCenterInPixels[0] - targetSize - 1: targetCenterInPixels[0] + targetSize + 2,
            targetCenterInPixels[1] - targetSize - 1: targetCenterInPixels[1] + targetSize + 2,
            targetCenterInPixels[2] - targetSize - 1: targetCenterInPixels[2] + targetSize + 2] = target

    envImg = envImg1 + envImg2
    shift_vals = [pos[i] - targetCenterInPixels[i] for i in range(3)]
    envImg_shifted = shift(envImg, shift_vals, order=1)

    return envImg_shifted

# Functions for generating plain images
def observeEnvAs2DImage_plain(pos=None, dose_q=1, form="ball", targetSize=2):
    # Similar to observeEnvAs2DImage, but without the surrounding volume
    envImg = np.zeros((depth, num_row, num_col), dtype=np.float64)
    ref_position = [depth // 2, num_row // 2, num_col // 2]

    if form == "cube":
        target = dose_q * cube(2 * targetSize + 1)
    elif form == "ball":
        target = dose_q * ball(targetSize)
    
    targetCenterInPixels = np.array(ref_position, dtype=int)
    envImg[targetCenterInPixels[0] - targetSize: targetCenterInPixels[0] + targetSize + 1,
           targetCenterInPixels[1] - targetSize: targetCenterInPixels[1] + targetSize + 1,
           targetCenterInPixels[2] - targetSize: targetCenterInPixels[2] + targetSize + 1] = target

    shift_vals = [pos[i] - targetCenterInPixels[i] for i in range(3)]
    envImg_shifted = shift(envImg, shift_vals, order=1)

    return envImg_shifted

# Initialize variables for dosimetric data
dose_quantity = 2
n_eval_episodes = 30
maxDose = (dose_quantity * n_eval_episodes) + 30
file_number = 5
base_path, file_extension = "", ".pickle"
FILE = [base_path + str(i) + file_extension for i in range(10)]

# Containers for dose metrics
GTV = np.zeros((file_number, 10000))
PTV = np.zeros((file_number, 10000))
PTVs = np.zeros((file_number, 10000))
OAR = np.zeros((file_number, 100000))

# Load dose files and compute DVH metrics
for j in range(file_number):
    with open(FILE[j], 'rb') as f:
        data = pickle.load(f)
    Accum_DM = data["Accumulation Dose"]
    
    # Compute DVH metrics
    d_PTV_all = Accum_DM[perfectDM_PTV_inref == dose_quantity] 
    perfect_q = 110
    PTV_metrics[j] = computeDx(d_PTV_all, 98, perfect_q)  # example, adapt for each metric
    # Similar computation for other regions...

# Plot the dose volume histogram
fig, ax = plt.subplots()
ax.plot(bin_edges_interpolated, np.mean(GTV, axis=0), linestyle="-", label="GTV DVH", color="orange")
ax.set_xlabel("Dose deposited (Gy)")
ax.set_ylabel("Percentage of volume (%)")
ax.legend()
plt.show()
