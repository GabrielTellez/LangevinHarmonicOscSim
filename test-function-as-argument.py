#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb


# In[46]:


def k(t):
    return 1.0
def center(t):
    return 0.0
def make_simulator(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100, k=k, center=center):
    k = nb.njit(k)
    center = nb.njit(center)
    @nb.njit
    def f(x,t):
        return -k(t)*(x-center(t))
    @nb.njit
    def U(x,t):
        return 0.5*k(t)*(x-center(t))*(x-center(t))
    @nb.njit
    def one_simulation(dt=dt, tot_steps=tot_steps, xinit=0.0, noise_scaler=noise_scaler, snapshot_step=snapshot_step):
        tot_snapshots = int(tot_steps/snapshot_step) + 1
        x = np.zeros(tot_snapshots, dtype=np.float64)
        work = np.zeros_like(x)
        power = np.zeros_like(x)
        heat = np.zeros_like(x)
        delta_U = np.zeros_like(x)
        energy = np.zeros_like(x)
        xold=xinit
        x[0]=xinit
        energy[0] = U(x[0],0)
        w = 0.0
        q = 0.0
        p = 0.0
        step=0
        snapshot_index=0
        while snapshot_index <= tot_snapshots:
            t=step*dt
            xnew = xold + f(xold,t)*dt + np.random.normal()*np.sqrt(2.0*dt*noise_scaler)
            p = U(xnew, t+dt)-U(xnew,t)
            w = w + p
            q = q + U(xnew,t)-U(xold,t)
            step=step+1
            if step % snapshot_step == 0:
                snapshot_index = snapshot_index + 1
                x[snapshot_index] = xnew
                power[snapshot_index] = p/dt
                work[snapshot_index] = w
                heat[snapshot_index] = q
                delta_U[snapshot_index] = U(xnew,t+dt)-U(x[0],0)
                energy[snapshot_index] = U(xnew,t+dt)
            xold=xnew
        return x, power, work, heat, delta_U, energy
    @nb.jit(parallel=True)
    def many_sims_parallel(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler = noise_scaler, snapshot_step = snapshot_step):
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
            xinit = np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
            x[sim_num], power[sim_num], work[sim_num], heat[sim_num], delta_U[sim_num], energy[sim_num] = one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
        return times, x, power, work, heat, delta_U, energy
    return many_sims_parallel


# In[47]:


simulator1 = make_simulator(tot_sims=1000000, dt=0.00001, tot_steps=10000)


# In[48]:


times, x, power, work, heat, delta_U, energy = simulator1()


# In[49]:


len(x)


# In[35]:


x


# In[54]:


def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300, x_label='x', y_label='P(x,t)', show_x_eq_distrib=True, k=k, center=center):
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


# In[50]:


animate_simulation(times,x,x_range=[-3,3],y_range=[0,1.8])


# In[52]:


simulator2 = make_simulator(tot_sims=1000000, dt=0.00001, tot_steps=10000, k=lambda t: 2.0, center=lambda t:  1.0)


# In[53]:


times, x, power, work, heat, delta_U, energy = simulator2()


# In[55]:


animate_simulation(times,x,x_range=[-3,3],y_range=[0,1.8])


# In[56]:


class trapezoidal():
    """This define processes with trapezoidal form , free parameters are t_a,t_b,k_m,t_f,k_i,k_f"""
    def __init__(self,t_a,t_b,k_m,t_f,k_i,k_f):
        self.t_a, self.t_b, self.k_m, self.t_f, self.k_i, self.k_f = t_a, t_b, k_m, t_f, k_i, k_f
    
    def k(self,t):
        t_a, t_b, k_m, t_f, k_i, k_f = self.t_a, self.t_b, self.k_m, self.t_f, self.k_i, self.k_f
        
        m_1 = (k_m-k_i)/t_a    #slope first part
        b_1 = k_i              #intercept first part
        c = k_m                # constant 
        m_2 = -(k_m-k_f)/(t_f-t_b) #slope second part
        b_2 = k_m-m_2*t_b      #intercept second part
        if t<=0.0:
            return k_i
        if 0<t<t_a:
            return m_1*t+b_1
        if t_a<t<t_b:
            return c
        if t_b<t<t_f:
            return m_2*t+b_2
        else: # t >= t_f
            return k_f        


# In[62]:


k_i=0.5
k_f=1
k_m=3.0
t_f=1.0/(k_f*30)
t_a=0.2*t_f
t_b=0.7*t_f
def k_trapez(t):        
        m_1 = (k_m-k_i)/t_a    #slope first part
        b_1 = k_i              #intercept first part
        c = k_m                # constant 
        m_2 = -(k_m-k_f)/(t_f-t_b) #slope second part
        b_2 = k_m-m_2*t_b      #intercept second part
        if t<=0.0:
            return k_i
        if 0<t<t_a:
            return m_1*t+b_1
        if t_a<t<t_b:
            return c
        if t_b<t<t_f:
            return m_2*t+b_2
        else: # t >= t_f
            return k_f        


# In[63]:


simulator3 = make_simulator(tot_sims=1000000, dt=0.00001, tot_steps=10000, k=k_trapez)


# In[64]:


times, x, power, work, heat, delta_U, energy = simulator3()


# In[65]:


animate_simulation(times,x,x_range=[-3,3],y_range=[0,1.8],k=k_trapez)


# In[ ]:




