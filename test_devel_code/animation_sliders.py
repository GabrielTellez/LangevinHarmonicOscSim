#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb


# In[2]:


# If the force is changed, all code has to be recompiled
@nb.njit
def k(t):
    return 1.0
@nb.njit
def center(t):
    xi = -0.5
    xf = 0.5
    tf = 5.0
    v = (xf - xi)/tf
    if t<tf:
        return xi + v*t
    else:
        return xf
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


# In[52]:


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


# In[44]:


times, xst = many_sims_parallel(tot_sims=1000000, snapshot_step = 100)


# In[54]:


anim_sliders=animate_simulation(times, xst, x_range=[-4.0,4.0], y_range=[0.0, 0.6], bins=200)


# In[55]:


anim_sliders.show()


# In[56]:


anim_sliders.write_html("anim_sliders.html")


# In[ ]:




