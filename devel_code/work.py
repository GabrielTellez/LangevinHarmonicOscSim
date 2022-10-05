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


# In[59]:


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


# In[9]:


times, x, power, work, heat, delta_U = many_sims_parallel()


# In[11]:


delta_U-work-heat


# In[12]:


work


# In[13]:


animate_simulation(times, work)


# In[26]:


np.average(work[:,10])


# In[15]:


np.average(heat[:,10])


# In[16]:


np.average(work[:,10])+np.average(heat[:,10])


# In[17]:


np.average(delta_U[:,10])


# In[18]:


np.average(work[:,10])+np.average(heat[:,10])-np.average(delta_U[:,10])


# In[20]:


np.average(power[:,10])


# In[21]:


np.average(np.exp(-work[:,10]))


# In[23]:


np.average(work, axis=0)


# In[27]:


px.line(x=times, y=np.average(work, axis=0))


# In[29]:


px.line(x=times, y=np.average(np.exp(-work), axis=0))


# In[30]:


times, x, power, work, heat, delta_U = many_sims_parallel(tot_sims=1000000)


# By Jarzynski equality : < exp(-work) > = exp (- Delta F) = 1

# In[31]:


px.line(x=times, y=np.average(np.exp(-work), axis=0))


# In[33]:


1/np.sqrt(1000)


# In[34]:


1/np.sqrt(1000000)


# In[35]:


animate_simulation(times, work)


# In[36]:


np.average(work[:,100])+np.average(heat[:,100])-np.average(delta_U[:,100])


# In[39]:


np.average(work[:,100])


# In[41]:


animate_simulation(times, np.exp(-work))


# In[42]:


px.line(x=times, y=np.average(power, axis=0))


# In[50]:


animate_simulation(times, power, x_range = [-0.001, 0.001], y_range=[0,2500])


# In[51]:


animate_simulation(times, heat)


# In[52]:


px.line(x=times, y=np.average(heat, axis=0))


# In[53]:


px.line(x=times, y=np.average(delta_U, axis=0))


# In[54]:


px.line(x=times, y=np.average(delta_U-heat-work, axis=0))


# In[55]:


animate_simulation(times, delta_U)


# Checking 1st law

# In[58]:


animate_simulation(times, delta_U-work-heat,x_range=[-0.000001,0.000001], y_range=[0,10000000])


# In[ ]:




