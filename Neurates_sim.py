import numpy as np


def simulate_session(sess, cwl, level, perf):
    # add random noise that represents factors other than CWL
    rand_noise = np.random.normal(-0.05, 0.1, N_CHARACTERS)  # same for both protocols in each session
    for protocol in range(level.shape[2]):  # per protocol
        # adjust CWL based on previous levels
        if sess != 0:
            # change cognitive workload based on previous session
            cwl = change_cwl(cwl, level, sess, protocol)
        # get performance level from each player
        level_per = calc_performance(cwl[:, sess, protocol])
        level_per = level_per + rand_noise
        level_per_ad = adjust_perf(level_per)
        perf[:, sess, protocol] = level_per_ad
        # get performance level from each player and adjust next level
        if sess < N_SESSIONS - 1:
            if protocol == 0:  # default
                # adjust level based on performance
                level = change_level(level, level_per_ad, sess, protocol)
            elif protocol == 1:  # neuro-adaptive
                # adjust level based on performance
                level = change_level(level, level_per_ad, sess, protocol)
                # adjust level based on CWL. this step can cause a player to jump 2 levels eventually, or alternatively
                # cancel the rise in level from the default protocol
                UP_BOUNDRY = [0.45, 0.55]  # between values
                DOWN_BOUNDRY = [0, 0.8]  # above/below values
                up_na = np.logical_and((cwl[:, sess, protocol] > UP_BOUNDRY[0]),
                                       (cwl[:, sess, protocol] < UP_BOUNDRY[1]))
                down_na = np.logical_or((cwl[:, sess, protocol] < DOWN_BOUNDRY[0]),
                                        (cwl[:, sess, protocol] > DOWN_BOUNDRY[1]))
                level[up_na, sess + 1, protocol] = level[up_na, sess + 1, protocol] + 1
                level[down_na, sess + 1, protocol] = level[down_na, sess + 1, protocol] - 1
                level[~up_na & ~down_na, sess + 1, protocol] = level[~up_na & ~down_na, sess + 1, protocol]
                # check not exceeding level boundaries [1 20]
            level[:, :, protocol] = adjust_level(level[:, :, protocol])
    return cwl, level, perf  # returns output of session


def calc_performance(cwl_per):
    # relation between performance and cwl = inverted U shape
    num_xpoints = 1000
    xvals = np.arange(1, num_xpoints + 1)
    # formula for inverted U shape relationship
    cwl_perf_relation = ((-1 / 250) * (xvals - 500) ** 2) + num_xpoints
    # get current session cwl
    cwl_sess = cwl_per
    indices = np.floor(cwl_sess * (num_xpoints - 1)).astype('int')
    # get performance based on cwl
    perf_in_session = cwl_perf_relation[indices] / num_xpoints
    return perf_in_session


def adjust_perf(perf):
    fix_top = np.where(perf > 1)
    if np.any(fix_top):
        perf[fix_top] = 1
    fix_bottom = np.where(perf < 0)
    if np.any(fix_bottom):
        perf[fix_bottom] = 0
    return perf


def adjust_level(level):
    fix_top = np.where(level > N_LEVELS)
    if np.any(fix_top):
        level[fix_top] = N_LEVELS
    fix_bottom = np.where(level < 1)
    if np.any(fix_bottom):
        level[fix_bottom] = 1
    return level


def change_level(level_array, performance, sess, protocol):
    PER_BOUNDRY = [0.55, 0.9]  # change level below/above performance boundary
    ind_down = performance <= PER_BOUNDRY[0]
    ind_up = performance >= PER_BOUNDRY[1]
    level_array[ind_up, sess + 1, protocol] = level_array[ind_up, sess, protocol] + 1
    level_array[ind_down, sess + 1, protocol] = level_array[ind_down, sess, protocol] - 1
    level_array[~ind_up & ~ind_down, sess + 1, protocol] = level_array[~ind_up & ~ind_down, sess, protocol]
    # check not exceeding level boundaries [1 20]
    level_array[:, :, protocol] = adjust_level(level_array[:, :, protocol])
    return level_array


def change_cwl(cwl, level, sess, protocol):
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


def run_sim(NUM_SESSIONS, NUM_LEVELS, NUM_CHARACTERS, CWL_LEV, PERFORMANCE_LEV, SIM_LEV):
    global N_SESSIONS, N_LEVELS, N_CHARACTERS
    N_LEVELS = NUM_LEVELS
    N_CHARACTERS = NUM_CHARACTERS
    N_SESSIONS = NUM_SESSIONS
    for sess in range(NUM_SESSIONS):
        # pass the "players" properties and return data
        cwl_next, level_next, perf_next = simulate_session(sess, CWL_LEV, SIM_LEV, PERFORMANCE_LEV)
        CWL_LEV = cwl_next
        SIM_LEV = level_next
        PERFORMANCE_LEV = perf_next
    return CWL_LEV, SIM_LEV, PERFORMANCE_LEV


if __name__ == "__main__":
    print('exc code')
