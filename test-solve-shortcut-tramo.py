#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb

def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300, x_label='x', y_label='P(x,t)', show_x_eq_distrib=True):
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
                            xaxis_title=x_label,
                            yaxis_title=y_label )
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
    fig_dict["data"] = [go.Bar(x=histos[0][1], y=histos[0][0], name=y_label)]
    if show_x_eq_distrib:
        fig_dict["data"].append(go.Scatter(x=xx,y=b[0], name = f"Eq. distr."))
    # make frames
    for time_index in range(0, len(times)):
        frame_data = [go.Bar(x=histos[time_index][1], y=histos[time_index][0])]
        if show_x_eq_distrib:
            frame_data.append(go.Scatter(x=xx,y=b[time_index]))
        frame = go.Frame(data=frame_data,
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


# In[2]:


from scipy.optimize import fsolve


# In[3]:


def eq_to_solve(k,ki,kf):
    return np.exp(-2.0*k)-(ki/kf)*(k-kf)/(k-ki)


# In[4]:


fsolve(eq_to_solve,1.0,args=(0.5,1.0))


# In[6]:


[ksol]=fsolve(eq_to_solve,1.0,args=(0.5,1.0))


# In[7]:


ksol


# In[5]:


@nb.njit
def k1(t):
    """ Original ESE from Trizac et al """
    gamma=1.0
    ki=0.5/gamma
    kf=1.0/gamma
    deltak=kf-ki
    tf=1.0/(kf*30) # 30 es la proporción entre trelax/tf en el articulo de Trizac
    s=t/tf
    if t<tf:
        return (3.0*deltak*s*(1-s)/tf)/(ki+deltak*(3.0*s**2-2.0*s**3)) + ki + deltak*(3.0*s**2-2.0*s**3)
    else:
        return kf
@nb.njit
def center1(t):
    return 0.0
@nb.njit
def f1(x,t):
    return -k1(t)*(x-center1(t))
@nb.njit
def U1(x,t):
    return 0.5*k1(t)*(x-center1(t))*(x-center1(t))
@nb.njit
def one_simulation1(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    x = np.zeros(tot_snapshots, dtype=np.float64)
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    energy = np.zeros_like(x)
    xold=xinit
    x[0]=xinit
    energy[0] = U1(x[0],0)
    w = 0.0
    q = 0.0
    p = 0.0
    step=0
    snapshot_index=0
    while snapshot_index <= tot_snapshots:
        t=step*dt
        xnew = xold + f1(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
        p = U1(xnew, t+dt)-U1(xnew,t)
        w = w + p
        q = q + U1(xnew,t)-U1(xold,t)
        step=step+1
        if step % snapshot_step == 0:
            snapshot_index = snapshot_index + 1
            x[snapshot_index] = xnew
            power[snapshot_index] = p/dt
            work[snapshot_index] = w
            heat[snapshot_index] = q
            delta_U[snapshot_index] = U1(xnew,t+dt)-U1(x[0],0)
            energy[snapshot_index] = U1(xnew,t+dt)
        xold=xnew
    return x, power, work, heat, delta_U, energy
@nb.jit(parallel=True)
def many_sims_parallel1(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x = np.zeros((tot_sims, tot_snapshots))
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    energy = np.zeros_like(x)
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        # initial position taken from equilibrium distribution at t=0
        xinit = np.random.normal(center1(0.0), scale=np.sqrt(1.0/k1(0.0)))
        x[sim_num], power[sim_num], work[sim_num], heat[sim_num], delta_U[sim_num], energy[sim_num] = one_simulation1(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x, power, work, heat, delta_U, energy


# In[16]:


gamma=1.0
ki=0.5/gamma
kf=1.0/gamma
deltak=kf-ki
tf=1.0/(kf*30) # 30 es la proporción entre trelax/tf en el articulo de Trizac
[ksoltf]=fsolve(eq_to_solve,1.0,args=(ki*tf,kf*tf))
ko=ksoltf/tf
@nb.njit
def k2(t):
    """ Protocolo por tramo recto horizontal """
    if t<=0.0:
        return ki
    if t<=tf:
        return ko
    else:
        return kf
@nb.njit
def center2(t):
    return 0.0
@nb.njit
def f2(x,t):
    return -k2(t)*(x-center2(t))
@nb.njit
def U2(x,t):
    return 0.5*k2(t)*(x-center2(t))*(x-center2(t))
@nb.njit
def one_simulation2(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    x = np.zeros(tot_snapshots, dtype=np.float64)
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    energy = np.zeros_like(x)
    xold=xinit
    x[0]=xinit
    energy[0] = U2(x[0],0)
    w = 0.0
    q = 0.0
    p = 0.0
    step=0
    snapshot_index=0
    while snapshot_index <= tot_snapshots:
        t=step*dt
        xnew = xold + f2(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
        p = U2(xnew, t+dt)-U2(xnew,t)
        w = w + p
        q = q + U2(xnew,t)-U2(xold,t)
        step=step+1
        if step % snapshot_step == 0:
            snapshot_index = snapshot_index + 1
            x[snapshot_index] = xnew
            power[snapshot_index] = p/dt
            work[snapshot_index] = w
            heat[snapshot_index] = q
            delta_U[snapshot_index] = U2(xnew,t+dt)-U2(x[0],0)
            energy[snapshot_index] = U2(xnew,t+dt)
        xold=xnew
    return x, power, work, heat, delta_U, energy
@nb.jit(parallel=True)
def many_sims_parallel2(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x = np.zeros((tot_sims, tot_snapshots))
    work = np.zeros_like(x)
    power = np.zeros_like(x)
    heat = np.zeros_like(x)
    delta_U = np.zeros_like(x)
    energy = np.zeros_like(x)
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        # initial position taken from equilibrium distribution at t=0
        xinit = np.random.normal(center2(0.0), scale=np.sqrt(1.0/k2(0.0)))
        x[sim_num], power[sim_num], work[sim_num], heat[sim_num], delta_U[sim_num], energy[sim_num] = one_simulation2(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x, power, work, heat, delta_U, energy


# In[17]:


times1, x1, power1, work1, heat1, delta_U1, energy1 = many_sims_parallel1(tot_sims=10000, dt=0.00001, tot_steps=10000)
times2, x2, power2, work2, heat2, delta_U2, energy2 = many_sims_parallel2(tot_sims=10000, dt=0.00001, tot_steps=10000)


# In[18]:


ks1=np.array([k1(t) for t in times1])
ks2=np.array([k2(t) for t in times2])


# In[19]:


px.line(x=times1, y=[ks1, ks2])


# In[20]:


def k(t):
    return k2(t)
def center(t):
    return 0.0


# In[21]:


sim_anim2=animate_simulation(times2,x2,x_range=[-3,3],y_range=[0,1.8])


# In[22]:


sim_anim2


# In[ ]:




