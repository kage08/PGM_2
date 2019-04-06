import numpy as np
from numba import jit, njit, prange
import time
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

@njit
def get_unnormalized(x=[0,0,0,0]):
    ans = 1.0
    ans *= 100.0 if x[0] == x[1] else 1.0
    ans *= 100.0 if x[2] == x[3] else 1.0
    ans *= 2.0 if x[1] == x[2] else 1.0
    return ans

@njit
def gibbs_sample(T=100000):
    current_state = np.array([0,0,0,0])
    states = []
    states.append(current_state)

    for _ in range(T):
        for i in range(4):
            current_state[i] = 0
            p0 = get_unnormalized(current_state)
            current_state[i] = 1
            p1 = get_unnormalized(current_state)
            current_state[i] = 0 if np.random.rand() < (p0/(p0+p1)) else 1
        states.append(current_state.copy())
    
    return states

@njit
def block_gibbs(T=100000):
    current_state = np.array([0,0,0,0])
    states = []
    states.append(current_state)
    p = np.array([0.,0.,0.,0.])

    for _ in range(T):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    current_state[i*2], current_state[i*2+1] = j,k
                    p[2*j+k] = get_unnormalized(current_state)
            p = p/np.sum(p)
            p = np.cumsum(p)
            r = np.random.rand()
            if r < p[0]: current_state[i*2], current_state[i*2+1] = 0, 0
            elif r < p[1]: current_state[i*2], current_state[i*2+1] = 0, 1
            elif r < p[2]: current_state[i*2], current_state[i*2+1] = 1, 0
            else: current_state[i*2], current_state[i*2+1] = 1, 1
            
        states.append(current_state.copy())
    
    return states
'''
Concert binary to decimal
'''
def dec(x):
    ans = 0
    for i in range(4):
        ans+= x[i]*(2**i)
    return ans

def convert_states(s):
    return [dec(x) for x in s]

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
    T1=10000
    T2 = 1000
    start = time.time()
    states1 = gibbs_sample(T1)
    stop = time.time()
    print('GS Took', stop-start,'seconds')
    start = time.time()
    states2 = block_gibbs(T2)
    stop = time.time()
    print('BGS Took', stop-start,'seconds')
    s1 = convert_states(states1)
    s2 = convert_states(states2)
    print("Converted")

    lag=500
    last1 = 40000
    last2 = 10000

    st1 = get_smooth(np.array(s1, dtype=float), lag)
    st2 = get_smooth(np.array(s2, dtype=float), lag)

    print("Done")
    

    hist_full1 = go.Histogram(
        x=s1,
        histnorm='probability density',
        name='histogram of '+str(T1)+' runs',
        xaxis='x1',
        yaxis='y1'
    )

    hist_last1 = go.Histogram(
        x=s1[-last1:],
        histnorm='probability density',
        name='histogram of last '+str(last1)+' samples',
        xaxis='x2',
        yaxis='y2'
    )

    trace_plot1 = go.Scatter(
        x=np.arange(len(s1)),
        y=s1,
        mode='lines',
        name='Traceplot',
        xaxis='x3',
        yaxis='y3'
    )

    trace_plot_mean1 = go.Scatter(
        x=np.arange(len(st1)),
        y=st1,
        mode='lines',
        name='Traceplot smoothened (average of '+str(lag)+' samples)',
        xaxis='x3',
        yaxis='y3'
    )

    hist_full2 = go.Histogram(
        x=s2,
        histnorm='probability density',
        name='histogram of '+str(T2)+' runs',
        xaxis='x1',
        yaxis='y1'
    )

    hist_last2 = go.Histogram(
        x=s2[-last2:],
        histnorm='probability density',
        name='histogram of last '+str(last2)+' samples',
        xaxis='x2',
        yaxis='y2'
    )

    trace_plot2 = go.Scatter(
        x=np.arange(len(s2)),
        y=s2,
        mode='lines',
        name='Traceplot',
        xaxis='x3',
        yaxis='y3'
    )

    trace_plot_mean2 = go.Scatter(
        x=np.arange(len(st2)),
        y=st2,
        mode='lines',
        name='Traceplot smoothened (average of '+str(lag)+' samples)',
        xaxis='x3',
        yaxis='y3'
    )

    fig = tools.make_subplots(rows=6, cols=2, subplot_titles=('Histogram of all points', 'Histogram of last '+str(last1)+' points','Trace Plot',
                                                                'Histogram of all points (BlockGibbs)', 'Histogram of last '+str(last2)+' points(BlockGibbs)','Trace Plot(BlockGibbs)'),
                          specs=[
                                 [{'rowspan': 2}, {'rowspan':2}],
                                 [None, None],
                                 [{'colspan':2}, None],
                                 [{'rowspan': 2}, {'rowspan':2}],
                                 [None, None],
                                 [{'colspan':2}, None]],
                          print_grid=True)

    #fig = tools.make_subplots(rows=2, cols=1, subplot_titles=('Histogram of all points', 'Histogram of last 1000 points'))
    fig.append_trace(hist_full1, 1, 1)
    fig.append_trace(hist_last1, 1, 2)
    fig.append_trace(trace_plot1, 3, 1)
    fig.append_trace(trace_plot_mean1, 3, 1)
    fig.append_trace(hist_full2, 4, 1)
    fig.append_trace(hist_last2, 4, 2)
    fig.append_trace(trace_plot2, 6, 1)
    fig.append_trace(trace_plot_mean2, 6, 1)
    

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
    
    fig['layout'].update(title='Gibbs Sampling', height=3000)

    #plot(fig, filename='plots/Q3_Summary.html')
