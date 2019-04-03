import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit

@njit
def mp_gamma_sample(k=1.,theta=2., T=10000, sigma=1., init = 1.):
    if init <= 0:
        init = np.random.rand() *5.
    curr_state = init
    
    states = []
    accept_r = []
    for t in range(T):
        gap = np.random.normal(0.0,sigma)
        next_state = curr_state + gap
        if next_state <= 0:
            states.append(curr_state)
            accept_r.append(0)
            continue
        accept_prob = ((next_state/curr_state)**(k-1))*np.exp((curr_state-next_state)/theta)
        accept_r.append(int(np.random.rand() < accept_prob))
        curr_state = next_state if accept_r[-1] else curr_state
        states.append(curr_state)
    return states, accept_r
        

if __name__ == "__main__":
    states, accept_r = mp_gamma_sample(T=10000000, sigma=1, k=4, theta=1)
    print("Done")
    plt.ion()
    plt.figure(1)
    plt.hist(states, 100, density=True)
    plt.figure(2)
    plt.plot(np.arange(len(states)), states)
    accept = []
    lag = 5000
    for i in range(lag, len(states)):
        accept.append(sum(accept_r[i-lag:i])/lag)
    plt.figure(3)
    plt.plot(np.arange(len(accept)), accept)
    plt.pause(0.1)