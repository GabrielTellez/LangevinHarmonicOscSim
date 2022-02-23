#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb


# In[20]:


# If the force is changed, all code has to be recompiled
@nb.njit
def k(t):
    """
    Optimal protocol from
    Schmield and Seifert PRL 98 108301 (2007), Eq. (19)
    """
    ki=0.2
    kf=1.0
    lambdai=ki
    lambdaf=kf
    tf=5.0
    if t==0.0:
        return ki
    else: 
        if t<tf:
            c2 = (-1-lambdaf*tf+np.sqrt(1+2*lambdai*tf+lambdai*lambdaf*tf*tf))/(tf*(2+lambdaf*tf))
            return (lambdai-c2*(1+c2*t))/(1.0+c2*t)**2
        else:
            return kf
@nb.njit
def center(t):
    return 0.0
@nb.njit
def f(x,t):
  return -k(t)*(x-center(t))
@nb.njit
def one_simulation(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    # times = np.zeros(tot_snapshots)
    x = np.zeros(tot_snapshots, dtype=np.float64)
    # print(shape(x))
    xold=xinit
    x[0]=xinit
    step=0
    snapshot_index=0
    while snapshot_index <= tot_snapshots:
        step=step+1
        t=step*dt
        xnew = xold + f(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
        if step % snapshot_step == 0:
            snapshot_index = snapshot_index + 1
            # times[snapshot_index] = t
            x[snapshot_index] = xnew
        xold=xnew
    return x
@nb.jit(parallel=True)
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x_sims=np.zeros((tot_sims, tot_snapshots))
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        # initial position taken from equilibrium distribution at t=0
        xinit = np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
        x_sims[sim_num]=one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x_sims


# In[3]:


def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300):
    xx=np.linspace(*x_range, 1000)
    histos=[np.histogram(xst[:,ti], density=True, range=x_range, bins=bins) for ti in range(0,len(times))]
    b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
    fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                      name='t = 0.00'),
                         go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
                  frames = [go.Frame(
                              data=[
                                  go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                     name=f't = {round(times[time_index],2)}'),
                                    go.Scatter(x=xx,y=b[time_index])
                                   ],
                              name=f"t={round(times[time_index],3)}",
                              traces=[0, 1]) for time_index in range(1, len(times))],
                layout = go.Layout(
                    xaxis=dict(range=x_range, autorange=False),
                    yaxis=dict(range=y_range, autorange=False),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                      method="animate",
                                      args=[None])])],
                        xaxis_title="x",
                        yaxis_title="P(x,t)")
                 )
    fig.update_layout(bargap=0)
    return fig


# In[21]:


times, xst = many_sims_parallel(tot_sims=1000000, snapshot_step = 100)


# In[22]:


animate_simulation(times, xst, x_range=[-4.0,4.0], y_range=[0.0, 0.6], bins=200)


# In[16]:


times


# Moving the center of the harmonic potential

# In[23]:


# If the force is changed, all code has to be recompiled
@nb.njit
def k(t):
    return 1.0
@nb.njit
def center(t):
    """
    Optimal protocol from
    Schmield and Seifert PRL 98 108301 (2007) Eq. (9)
    """
    lambdaf=1.0
    tf=5.0
    if t==0.0:
        return 0.0
    else:
        if t<tf:
            return lambdaf*(t+1)/(tf+2)
        else:
            return lambdaf 
@nb.njit
def f(x,t):
    return -k(t)*(x-center(t))
@nb.njit
def one_simulation(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    # times = np.zeros(tot_snapshots)
    x = np.zeros(tot_snapshots, dtype=np.float64)
    # print(shape(x))
    xold=xinit
    x[0]=xinit
    step=0
    snapshot_index=0
    while snapshot_index <= tot_snapshots:
        step=step+1
        t=step*dt
        xnew = xold + f(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
        if step % snapshot_step == 0:
            snapshot_index = snapshot_index + 1
            # times[snapshot_index] = t
            x[snapshot_index] = xnew
        xold=xnew
    return x
@nb.jit(parallel=True)
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x_sims=np.zeros((tot_sims, tot_snapshots))
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        # initial position taken from equilibrium distribution at t=0
        xinit = np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
        x_sims[sim_num]=one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x_sims


# In[24]:


times, xst = many_sims_parallel(tot_sims=1000000, snapshot_step = 100)


# In[25]:


animate_simulation(times, xst, x_range=[-4.0,4.0], y_range=[0.0, 0.6], bins=200)


# In[ ]:




