import numpy as np
from numba import jit, njit
from scipy.stats import gamma

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


@njit
def mp_gamma_sample(k=1.,theta=2., T=10000, sigma=1., init = 1.):
    if init <= 0:
        init = np.random.rand() *5.
    curr_state = init
    
    states = []
    states.append(curr_state)
    accept_r1 = []
    accept_r = []

    #For each iteration of sampling
    for _ in range(T):
        #calculate the displacement to next proposed state X'
        gap = np.random.normal(0.0,sigma)
        next_state = curr_state + gap
        #Check if P(X')>0 else remain in X
        if next_state <= 0:
            states.append(curr_state)
            accept_r.append(0)
            accept_r1.append(0)
            continue
        
        #Calculate the accepatance proabbility
        accept_prob = ((next_state/curr_state)**(k-1))*np.exp((curr_state-next_state)/theta)
        #Sample from [0,1] uniformly and check if Acceptance probability is grater than that number
        accept_r1.append(int(np.random.rand() < accept_prob))
        accept_r.append(min(1,accept_prob))

        #Change next state to X' if accepted
        curr_state = next_state if accept_r1[-1] else curr_state
        states.append(curr_state)
    return states, accept_r

'''
A smoothing function for plots
'''
@njit
def get_accept(accept_r, states, lag=5000):
    accept = np.copy(accept_r)
    st = np.copy(states[:])
    for i in prange(lag):
        accept[i] = np.mean(accept_r[:i+1])
        st[i] = np.mean(states[:i+1])
    for i in prange(lag, len(states)-1):
        accept[i] = np.mean(accept_r[i-lag:i])
        st[i] = np.mean(states[i-lag:i])
    return accept, st

if __name__ == "__main__":
    T=10000
    sigma = 25
    lag = 100
    last = 1000
    k=5.5
    theta = 1

    print('Proposal Distribution variance',sigma)
    print('Rounds:',T)
    print("\ngamma distribution parameters")
    print('k:',k)
    print('Theta:', theta)


    #Sample T times
    states, accept_r = mp_gamma_sample(T=T, sigma=sigma, k=k, theta=theta)

    #We sample last 1000 from states
    states_samples = states[-last:]
    print("Done")

    #Calculate acceptance rate
    accept, st = get_accept(np.array(accept_r, dtype = float), np.array(states), lag)
    #accept1 = np.cumsum(accept_r)/np.arange(1, len(accept_r)+1)
    accept1 = accept_r
    print("Done")

    #Calculate Gamma pdf function
    vals = np.arange(1,20,0.01)
    gvals = gamma.pdf(vals, 5.5)
    print('Mean accept:',np.mean(accept_r[-last:]))

    #Plot
    trace_gamma = go.Scatter(
        x=vals,
        y=gvals,
        mode='lines',
        name='Gamma(5.5,1)'
    )

    hist_full = go.Histogram(
        x=states,
        histnorm='probability density',
        name='histogram of '+str(T)+' runs',
        xaxis='x1',
        yaxis='y1'
    )

    hist_first = go.Histogram(
        x=states[:1000],
        histnorm='probability density',
        name='histogram of first 1000 samples',
        xaxis='x2',
        yaxis='y2'
    )

    hist_last = go.Histogram(
        x=states[-1000:],
        histnorm='probability density',
        name='histogram of last 1000 samples',
        xaxis='x2',
        yaxis='y2'
    )

    trace_plot = go.Scatter(
        x=np.arange(len(states)),
        y=states,
        mode='lines',
        name='Traceplot',
        xaxis='x3',
        yaxis='y3'
    )

    trace_plot_mean = go.Scatter(
        x=np.arange(len(st)),
        y=st,
        mode='lines',
        name='Traceplot smoothened (average of '+str(lag)+' samples)',
        xaxis='x3',
        yaxis='y3'
    )


    accept_plot = go.Scatter(
        x=np.arange(len(accept1)),
        y=accept1,
        mode='lines',
        name='Acceptance rate',
        xaxis='x4',
        yaxis='y4'
    )

    accept_plot_mean = go.Scatter(
        x=np.arange(len(accept)),
        y=accept,
        mode='lines',
        name='Acceptance rate over next '+str(lag)+' samples',
        xaxis='x4',
        yaxis='y4'
    )
    
    fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('Histogram of all points', 'Histogram of last 1000 points', 'Traceplot', 'Acceptance Rate'))
    fig.append_trace(hist_full, 1, 1)
    fig.append_trace(trace_gamma, 1, 1)
    fig.append_trace(hist_last, 2, 1)
    fig.append_trace(trace_gamma, 2, 1)
    fig.append_trace(trace_plot, 3, 1)
    fig.append_trace(trace_plot_mean, 3, 1)
    fig.append_trace(accept_plot, 4, 1)
    fig.append_trace(accept_plot_mean, 4, 1)

    fig['layout']['xaxis1'].update(title='Values')
    fig['layout']['xaxis2'].update(title='Values')
    fig['layout']['xaxis3'].update(title='Time')
    fig['layout']['xaxis4'].update(title='Time')

    fig['layout']['yaxis1'].update(title='Density')
    fig['layout']['yaxis2'].update(title='Density')
    fig['layout']['yaxis3'].update(title='Value')
    fig['layout']['yaxis4'].update(title='Acceptance Rate')
    fig['layout'].update(title='Metropolis hasting for Gamma distribution', height=2000)

    plot(fig, filename='plots/MH_Summary.html')
