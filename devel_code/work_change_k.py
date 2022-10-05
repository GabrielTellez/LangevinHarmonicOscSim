#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb


# In[2]:


def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300):
    xx=np.linspace(*x_range, 1000)
    histos=[np.histogram(xst[:,ti], density=True, range=x_range, bins=bins) for ti in range(0,len(times))]
    b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"] = go.Layout(
                            xaxis=dict(range=x_range, autorange=False),
                            yaxis=dict(range=y_range, autorange=False),
                            xaxis_title="x",
                            yaxis_title="P(x,t)")
    fig_dict["layout"]["updatemenus"] = [
        {
        "type": "buttons",
        "buttons": [{
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
        }]
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 14},
            "prefix": "t = ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    fig_dict["data"] = [
        go.Bar(x=histos[0][1], y=histos[0][0], name='P(x,t)'),
        go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")]
    # make frames
    for time_index in range(0, len(times)):
        frame = go.Frame(data=[go.Bar(x=histos[time_index][1], y=histos[time_index][0]),
                               go.Scatter(x=xx,y=b[time_index])
                              ],
                              name=time_index,
                              traces=[0, 1])
        fig_dict["frames"].append(frame)
        slider_step = {
            "args": [
                [time_index],
                {"frame": {"duration": 300, "redraw": False},
                 "mode": "immediate",
                 "transition": {"duration": 300}}
                ],
                "label": round(times[time_index],3),
                "method": "animate"}
        sliders_dict["steps"].append(slider_step)
    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig=go.Figure(fig_dict)
    fig.update_layout(bargap=0)
    return fig


# In[13]:


@nb.njit
def k(t):
    tf=5.0
    ki=1.0
    kf=2.0
    if t<tf:
        return ki+(kf-ki)*t/tf
    else:
        return kf
@nb.njit
def center(t):
    return 0.0
@nb.njit
def f(x,t):
  return -k(t)*(x-center(t))
@nb.njit
def U(x,t):
    return 0.5*k(t)*(x-center(t))*(x-center(t))
@nb.njit
def one_simulation(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    x = np.zeros(tot_snapshots, dtype=np.float64)
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    xold=xinit
    x[0]=xinit
    w = 0.0
    q = 0.0
    p = 0.0
    step=0
    snapshot_index=0
    while snapshot_index <= tot_snapshots:
        step=step+1
        t=step*dt
        xnew = xold + f(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
        p = U(xold, t)-U(xold,t-dt)
        w = w + p
        q = q + U(xnew,t)-U(xold,t)
        if step % snapshot_step == 0:
            snapshot_index = snapshot_index + 1
            x[snapshot_index] = xnew
            power[snapshot_index] = p
            work[snapshot_index] = w
            heat[snapshot_index] = q
            delta_U[snapshot_index] = U(xnew,t)-U(x[0],0)
        xold=xnew
    return x, power, work, heat, delta_U
@nb.jit(parallel=True)
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x = np.zeros((tot_sims, tot_snapshots))
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        # initial position taken from equilibrium distribution at t=0
        xinit = np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
        x[sim_num], power[sim_num], work[sim_num], heat[sim_num], delta_U[sim_num] = one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x, power, work, heat, delta_U


# In[14]:


times, x, power, work, heat, delta_U = many_sims_parallel(tot_sims=1000000)


# By Jarzynski equality : < exp(-work) > = exp (- Delta F) = sqrt(ki/kf)

# In[22]:


px.line(x=times, y=np.average(np.exp(-work), axis=0))


# In[21]:


np.sqrt(1/2.0)


# In[28]:


ks=np.array([k(t) for t in times])


# In[31]:


px.line(x=times, y=np.sqrt(ks[0]/ks))


# In[32]:


px.line(x=times, y=[np.sqrt(ks[0]/ks), np.average(np.exp(-work), axis=0)])


# In[33]:


px.line(x=times, y=np.sqrt(ks[0]/ks)-np.average(np.exp(-work), axis=0))


# In[16]:


work


# In[17]:


x


# In[18]:


delta_U


# In[19]:


heat


# In[12]:


animate_simulation(times, x)


# In[25]:


animate_simulation(times, work,x_range=[-1,1], y_range=[0,4])


# In[ ]:




