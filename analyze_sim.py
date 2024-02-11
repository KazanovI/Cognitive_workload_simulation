import Neurates_sim
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors




NUM_SESSIONS = 20 # how many sessions in simulation
NUM_LEVELS = 10 # maximum level that can be reached
NUM_CHARACTERS = 100
mu, sigma = 0.5, 0.1 # mean and standard deviation
BASE_CWL = np.random.normal(mu, sigma, NUM_CHARACTERS) # baseline level of each character's cognitive working load
BASE_CWL = Neurates_sim.adjust_perf(BASE_CWL)
CWL_LEVEL = np.zeros((NUM_CHARACTERS,NUM_SESSIONS,2))
CWL_LEVEL[:,0,:] = np.tile(BASE_CWL,(2,1)).T
PERFORMANCE_LEVEL = np.array(np.zeros((NUM_CHARACTERS,NUM_SESSIONS,2)))
SIM_LEVEL = np.array(np.zeros((NUM_CHARACTERS,NUM_SESSIONS+1,2))) # level reached in simulation
SIM_LEVEL[:,0,:] = 1 # start from level 1
# run simulation
cwl,level,perf = Neurates_sim.run_sim(NUM_SESSIONS,NUM_LEVELS,NUM_CHARACTERS,CWL_LEVEL,PERFORMANCE_LEVEL,SIM_LEVEL)

# plots section
def_progress = level[:,:,0]
na_progress = level[:,:,1]

# heatmap to look at all the players together, sorted by base cwl
sort_ind = np.argsort(BASE_CWL)
na_prog_sorted = na_progress[sort_ind,:]
def_prog_sorted = def_progress[sort_ind,:]
images = []
fig, axs = plt.subplots(1,2)
images.append(axs[0].imshow(na_prog_sorted))
axs[0].set_title("Progression by NeuroAdaptive")
axs[0].set_xlabel('Session')
axs[0].set_ylabel('Players')
images.append(axs[1].imshow(def_prog_sorted))
axs[1].set_title("Progression by default")
axs[1].set_xlabel('Session')
axs[1].set_ylabel('Players')
# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, left = 0.1)
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1,label='Level')
plt.show()
# look at the difference
diff_sorted = na_prog_sorted - def_prog_sorted
symmetric = np.max([np.max(diff_sorted),np.abs(np.min(diff_sorted))])
plt.imshow(diff_sorted,cmap='seismic',vmin = -symmetric,vmax = symmetric)
plt.colorbar()
plt.title("LEVEL Difference between protocols(NA-def)")
plt.show()

# which base cwl has benefited/got worse
per_player = np.mean(na_prog_sorted - def_prog_sorted,axis=1)
BASE_CWL_sorted = BASE_CWL[sort_ind]
plt.plot(BASE_CWL_sorted,per_player,color= 'k')
plt.xlabel("BaseLine CWL")
plt.ylabel("Mean diff(NA-Def) of levels")
plt.axhline(linewidth=2, color='r',ls = '--')
plt.show()


def_cwl = cwl[:,:,0]
na_cwl = cwl[:,:,1]

# heatmap to look at all the players together, sorted by base cwl
na_cwl_sorted = na_cwl[sort_ind,:]
def_cwl_sorted = def_cwl[sort_ind,:]
images = []
fig, axs = plt.subplots(1,2)
images.append(axs[0].imshow(na_cwl_sorted))
axs[0].set_title("workload by NeuroAdaptive")
images.append(axs[1].imshow(def_cwl_sorted))
axs[1].set_title("workload by default")
# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, left = 0.1)
fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)
plt.show()

# look at the difference
diff_cwl_sorted = na_cwl_sorted - def_cwl_sorted
plt.imshow(diff_cwl_sorted,cmap='seismic')
plt.colorbar()
plt.title("CWL Difference between protocols(NA-def)")
plt.show()


# look at data by baseline CWL per player
base_cwl_sorted = BASE_CWL[sort_ind]
for ind in range(NUM_CHARACTERS):
    plt.style.use("ggplot")
    plt.plot(def_prog_sorted[ind, :-1], color="black", label="Default")
    plt.plot(na_prog_sorted[ind, :-1], color="red", label="Neuro-adaptive")
    plt.title('Compare progress, ' + str(ind) + '. baseline CWL: ' + '{0:.03f}'.format(base_cwl_sorted[ind]))
    plt.xlabel('Session number')
    plt.ylabel('level')
    plt.xticks(np.arange(NUM_SESSIONS))
    plt.tight_layout()
    plt.legend()
    plt.show()