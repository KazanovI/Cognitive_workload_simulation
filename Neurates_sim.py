import numpy as np

def simulate_session(sess,cwl,level,perf):
    for protocol in range(level.shape[2]): # per protocol
        # adjust CWL based on previous levels
        if sess != 0:
            # change cognitive workload based on previous session
            cwl = change_cwl(cwl,level,sess,protocol)
            # get performance level from each player based on cwl
            level_per = calc_performance(cwl[:, sess, protocol])
            perf[:,sess, protocol] = level_per
        else:
            if protocol == 0: # on first day both protocols have the same properties
                # get performance level from each player
                level_per = calc_performance(cwl[:, sess, protocol])
                perf[:, sess, protocol] = level_per
        # get performance level from each player
        if protocol == 0: # default
            # adjust level based on performance
            level = change_level(level,level_per,sess,protocol)
        elif protocol == 1: # neuro-adaptive
            # adjust level based on performance
            level = change_level(level,level_per, sess, protocol)
            # adjust level based on CWL. this step can cause a player to jump 2 levels eventually, or alternatively
            # cancel the rise in level from the default protocol
            UP_BOUNDRY = [0.4, 0.6] # between values
            DOWN_BOUNDRY = [0.1, 0.9] # above/below values
            up_na = np.logical_and((cwl[:,sess,protocol] > UP_BOUNDRY[0]) , (cwl[:,sess,protocol] < UP_BOUNDRY[1]))
            down_na = np.logical_or((cwl[:,sess,protocol] < DOWN_BOUNDRY[0]) , (cwl[:,sess,protocol] > DOWN_BOUNDRY[1]))
            level[up_na, sess + 1, protocol] = level[up_na, sess + 1, protocol] + 1
            level[down_na, sess + 1, protocol] = level[down_na, sess + 1, protocol] - 1
            level[~up_na & ~down_na, sess + 1, protocol] = level[~up_na & ~down_na, sess + 1, protocol]
            # check not exceeding level boundaries [1 20]
            level[:, :, protocol] = adjust_level(level[:, :, protocol])
    return cwl,level,perf # returns output of session

def calc_performance(cwl_per):
    # relation between performance and cwl = inverted U shape
    num_xpoints = 1000
    xvals = np.arange(1,num_xpoints+1)
    # formula for inverted U shape relationship
    cwl_perf_relation = ((-1/250)*(xvals-500)**2)+num_xpoints
    # get current session cwl
    cwl_sess = cwl_per
    indices = np.floor(cwl_sess*(num_xpoints-1)).astype('int')
    # get performance based on cwl
    perf_in_session = cwl_perf_relation[indices]/num_xpoints
    # add random noise that represents factors other than CWL
    perf_in_session = perf_in_session + (perf_in_session * np.random.normal(-0.05, 0.1, N_CHARACTERS))
    # adjust performance to not pass limits [0 1]
    perf_in_session_ad = adjust_perf(perf_in_session)
    return perf_in_session_ad

def adjust_perf(perf):
    fix_top = np.where(perf > 1)
    if np.any(fix_top):
        perf[fix_top] = 1
    fix_bottom = np.where(perf < 0)
    if np.any(fix_bottom):
        perf[fix_bottom] = 0
    return perf

def adjust_level(level):
    level_ad = level
    fix_top = np.where(level > N_LEVELS)
    if np.any(fix_top):
        level[fix_top] = N_LEVELS
    fix_bottom = np.where(level < 1)
    if np.any(fix_bottom):
        level[fix_bottom] = 1
    return level_ad

def change_level(level_array,performance,sess,protocol):
    PER_BOUNDRY = [0.55, 0.9] # change level below/above performance boundary
    ind_down = performance <= PER_BOUNDRY[0]
    ind_up = performance >= PER_BOUNDRY[1]
    level_array[ind_up, sess + 1, protocol] = level_array[ind_up, sess, protocol] + 1
    level_array[ind_down, sess + 1, protocol] = level_array[ind_down, sess, protocol] - 1
    level_array[~ind_up & ~ind_down, sess + 1, protocol] = level_array[~ind_up & ~ind_down, sess, protocol]
    # check not exceeding level boundaries [1 20]
    level_array[:, :, protocol] = adjust_level(level_array[:, :, protocol])
    return level_array

def change_cwl(cwl,level,sess,protocol):
    level_rep = 0.1
    level_down = 0.2
    level_up = 0.2
    # check the direction the cwl needs to be adjusted towards
    too_easy_ind = cwl[:, sess - 1, protocol] < 0.5
    too_hard_ind = cwl[:, sess - 1, protocol] > 0.5
    # no change(same level again) - change cwl toward 0.5
    no_change_ind = level[:, sess, protocol] == level[:, sess - 1, protocol]
    cwl[no_change_ind & too_easy_ind, sess, protocol] = cwl[no_change_ind & too_easy_ind, sess - 1, protocol] + (
                cwl[no_change_ind & too_easy_ind, sess - 1, protocol] * level_rep)
    cwl[no_change_ind & too_hard_ind, sess, protocol] = cwl[no_change_ind & too_hard_ind, sess - 1, protocol] + (
                cwl[no_change_ind & too_hard_ind, sess - 1, protocol] * -level_rep)
    cwl[no_change_ind & ~too_easy_ind & ~too_hard_ind, sess, protocol] = cwl[
        no_change_ind & ~too_easy_ind & ~too_hard_ind, sess - 1, protocol]
    # level up - increase cwl
    up_ind = level[:, sess, protocol] - level[:, sess - 1, protocol] >= 1
    cwl[up_ind, sess, protocol] = cwl[up_ind, sess - 1, protocol] + (cwl[up_ind, sess - 1, protocol] * level_up)
    # level down - change cwl toward 0.5
    down_ind = level[:, sess, protocol] - level[:, sess - 1, protocol] <= -1
    cwl[down_ind & too_easy_ind, sess, protocol] = cwl[down_ind & too_easy_ind, sess - 1, protocol] + (
                cwl[down_ind & too_easy_ind, sess - 1, protocol] * level_down)
    cwl[down_ind & too_hard_ind, sess, protocol] = cwl[down_ind & too_hard_ind, sess - 1, protocol] + (
                cwl[down_ind & too_hard_ind, sess - 1, protocol] * -level_down)
    cwl[down_ind & ~too_easy_ind & ~too_hard_ind, sess, protocol] = cwl[
        down_ind & ~too_easy_ind & ~too_hard_ind, sess - 1, protocol]
    cwl = adjust_perf(cwl)
    return cwl

def run_sim(NUM_SESSIONS,NUM_LEVELS,NUM_CHARACTERS,CWL_LEVEL,PERFORMANCE_LEVEL,SIM_LEVEL):
    global N_LEVELS,N_CHARACTERS
    N_LEVELS = NUM_LEVELS
    N_CHARACTERS = NUM_CHARACTERS
    for sess in range(NUM_SESSIONS):
        # pass the "players" properties and return data
        cwl_next, level_next, perf_next = simulate_session(sess,CWL_LEVEL,SIM_LEVEL,PERFORMANCE_LEVEL)
        CWL_LEVEL = cwl_next
        SIM_LEVEL = level_next
        PERFORMANCE_LEVEL = perf_next
    return CWL_LEVEL,SIM_LEVEL,PERFORMANCE_LEVEL

if __name__ == "__main__":
    print('exc code')

