import Neurates_sim
import numpy as np
from matplotlib import pyplot as plt



NUM_SESSIONS = 30 # how many sessions in simulation
NUM_LEVELS = 20 # maximum level that can be reached
NUM_CHARACTERS = 50
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
fig, (ax1,ax2) = plt.subplots(1,2)
im1 = ax1.imshow(na_prog_sorted)
fig.colorbar(im1, ax=ax1)
ax1.set_title("Progression by NeuroAdaptive")
im2 = ax2.imshow(def_prog_sorted)
fig.colorbar(im2, ax=ax2)
ax2.set_title("Progression by default")
# fig.tight_layout()
plt.show()

# look at the difference
diff_sorted = na_prog_sorted - def_prog_sorted
plt.imshow(diff_sorted,cmap='seismic')
plt.colorbar()
plt.title("Difference between protocols(NA-def)")
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