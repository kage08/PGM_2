import numpy as np
from numba import jit, njit, prange
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


@njit
def gibbs_sample(curr_state, T=100000, grid_len=10, rand = False):
    states = []
    states.append(curr_state)
    t=0
    while t<T:
        #For each i,j
        for i_ in range(grid_len):
            for j_ in range(grid_len):
                #If random scan choos i,j at random
                if rand:
                    i = np.random.randint(grid_len)
                    j = np.random.randint(grid_len)
                else:
                    i,j = i_, j_
                toss = np.random.randint(2)
                is_valid = True
                #Check if neighbors have value 1
                if toss==1:
                    if i-1>= 0 and curr_state[i-1,j]==1: is_valid=False
                    if i+1<grid_len and curr_state[i+1,j] == 1: is_valid=False
                    if j-1>= 0 and curr_state[i,j-1]==1: is_valid=False
                    if j+1<grid_len and curr_state[i,j+1] == 1: is_valid=False
                if is_valid: curr_state[i,j] = toss
                if rand:
                    states.append(np.copy(curr_state))
                    t+= 1
                    if t>=T: break
        if not rand:
            states.append(np.copy(curr_state))
            t+= 1
    
    return states
'''
A smoothing function for plots
'''
@njit
def get_smooth(states, lag):
    st=[]
    for i in prange(lag):
        st.append(np.mean(states[:i+1]))
    for i in prange(lag, len(states)):
        st.append(np.mean(states[i-lag:i]))
    return st

if __name__ == "__main__":
    grid_len = 50
    T=800
    print('grid length:',grid_len)
    print('Rounds:',T)

    #Get all states
    states_ = gibbs_sample(curr_state = np.eye(grid_len, dtype=np.bool), grid_len=grid_len, T=T)
    states1_ = gibbs_sample(curr_state = np.eye(grid_len, dtype=np.bool), grid_len=grid_len, T=T, rand=True)
    print("Done")

    #Number of ones in each state
    states = np.sum(states_, axis=(1,2))
    states1 = np.sum(states1_, axis=(1,2))

    
    lag = 100
    #We choose last 1000
    last = 5000
    last1 = 1000
    st = get_smooth(states, lag)
    st1 = get_smooth(states, lag)
    print("Done")

    hist_full = go.Histogram(
        x=states,
        histnorm='probability density',
        name='histogram of '+str(T)+' runs',
        xaxis='x1',
        yaxis='y1'
    )

    hist_last = go.Histogram(
        x=states[-last:],
        histnorm='probability density',
        name='histogram of last '+str(last)+' samples',
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

    hist_full1 = go.Histogram(
        x=states1,
        histnorm='probability density',
        name='histogram of '+str(T)+' runs',
        xaxis='x1',
        yaxis='y1'
    )

    hist_last1 = go.Histogram(
        x=states1[last1:],
        histnorm='probability density',
        name='histogram of last '+str(last1)+' samples',
        xaxis='x2',
        yaxis='y2'
    )

    trace_plot1 = go.Scatter(
        x=np.arange(len(states1)),
        y=states1,
        mode='lines',
        name='Traceplot',
        xaxis='x3',
        yaxis='y3'
    )

    trace_plot_mean1 = go.Scatter(
        x=np.arange(len(st1)),
        y=st,
        mode='lines',
        name='Traceplot smoothened (average of '+str(lag)+' samples)',
        xaxis='x3',
        yaxis='y3'
    )

    fig = tools.make_subplots(rows=6, cols=2, subplot_titles=('Histogram of all points', 'Histogram of last '+str(last)+' points','Trace Plot',
                                                                'Histogram of all points (RandomScan)', 'Histogram of leaving first '+str(last1)+' points(RandomScan)','Trace Plot(RandomScan)'),
                          specs=[
                                 [{'rowspan': 2}, {'rowspan':2}],
                                 [None, None],
                                 [{'colspan':2}, None],
                                 [{'rowspan': 2}, {'rowspan':2}],
                                 [None, None],
                                 [{'colspan':2}, None]],
                          print_grid=True)

    fig.append_trace(hist_full, 1, 1)
    fig.append_trace(hist_last, 1, 2)
    fig.append_trace(trace_plot, 3, 1)
    fig.append_trace(trace_plot_mean, 3, 1)
    fig.append_trace(hist_full1, 4, 1)
    fig.append_trace(hist_last1, 4, 2)
    fig.append_trace(trace_plot1, 6, 1)
    fig.append_trace(trace_plot_mean1, 6, 1)
    

    fig['layout']['xaxis1'].update(title='Values')
    fig['layout']['xaxis2'].update(title='Values')
    fig['layout']['xaxis3'].update(title='Time')
    

    fig['layout']['yaxis1'].update(title='Density')
    fig['layout']['yaxis2'].update(title='Density')
    fig['layout']['yaxis3'].update(title='Value')

    fig['layout']['xaxis4'].update(title='Values')
    fig['layout']['xaxis5'].update(title='Values')
    fig['layout']['xaxis6'].update(title='Time')
    

    fig['layout']['yaxis4'].update(title='Density')
    fig['layout']['yaxis5'].update(title='Density')
    fig['layout']['yaxis6'].update(title='Value')
    
    fig['layout'].update(title='Gibbs Sampling on Hardcore model on grid graph n='+str(grid_len), height=2000)

    plot(fig, filename='plots/Hardcore_Summary.html')