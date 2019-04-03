import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange

@njit
def gibbs_sample(curr_state, T=100000, grid_len=10, seed=None):
    states = []
    
    for i in range(T):
        is_valid = True
        for i in range(grid_len):
            for j in range(grid_len):
                if curr_state[i,j]==0:
                    if i-1>= 0 and curr_state[i-1,j]==1: is_valid=False
                    if i+1<grid_len and curr_state[i+1,j] == 1: is_valid=False
                    if j-1>= 0 and curr_state[i,j-1]==1: is_valid=False
                    if j+1<grid_len and curr_state[i,j+1] == 1: is_valid=False
                if is_valid: curr_state[i,j] = np.random.rand()<0.5
                states.append(curr_state.copy())
    
    return states


if __name__ == "__main__":
    grid_len = 10
    states = gibbs_sample(curr_state = np.zeros((grid_len, grid_len), dtype=np.bool), grid_len=grid_len)