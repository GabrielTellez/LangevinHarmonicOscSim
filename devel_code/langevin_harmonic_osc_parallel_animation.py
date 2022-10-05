#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/GabrielTellez/LangevinHarmonicOscSim/blob/main/langevin_harmonic_osc_parallel_optimized2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb


# In[3]:


@nb.njit
def one_simulation(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    def k(t):
      return 1.0+1.0*t
    def center(t):
      return 0.5*t
    def f(x,t):
      return -k(t)*(x-center(t))
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


# In[27]:


x = one_simulation()


# In[28]:


len(x)


# In[29]:


shape(x)


# In[30]:


plot(x)


# In[30]:


@nb.jit(parallel=True)
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x_sims=np.zeros((tot_sims, tot_snapshots))
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        x_sims[sim_num]=one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x_sims


# In[31]:


times, xs = many_sims_parallel()


# In[32]:


len(times), shape(xs)


# In[33]:


times


# In[34]:


times, xs = many_sims_parallel(tot_sims=1, dt=0.001, tot_steps=10010, snapshot_step=100)
print(len(times), shape(xs))
times


# In[36]:


get_ipython().run_line_magic('time', 'many_sims_parallel()')


# In[10]:


@nb.njit
def one_simulation_w_times(dt=0.001, tot_steps=10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    def k(t):
      return 1.0+1.0*t
    def center(t):
      return 0.5*t
    def f(x,t):
      return -k(t)*(x-center(t))
    tot_snapshots = int(tot_steps/snapshot_step) + 1
    times = np.zeros(tot_snapshots)
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
            times[snapshot_index] = t
            x[snapshot_index] = xnew
        xold=xnew
    return times, x


# In[12]:


times, x = one_simulation_w_times()


# In[13]:


times


# In[36]:


times, xs = one_simulation_w_times( dt=0.001, tot_steps=10010, snapshot_step=100)
print(len(times), shape(xs))
times


# In[37]:


get_ipython().run_line_magic('time', 'many_sims_parallel(tot_sims=100000)')


# In[38]:


get_ipython().run_line_magic('time', 'many_sims_parallel(tot_sims=100000, snapshot_step = 1000)')


# In[39]:


get_ipython().run_line_magic('time', 'many_sims_parallel(tot_sims=100000, snapshot_step = 1000)')


# In[40]:





# In[43]:


dt = 0.001
snapshot_step = 1000
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
times, xst = many_sims_parallel(tot_sims=100000, dt = dt, snapshot_step = snapshot_step)
fig=go.Figure()
for time_index in range(1,11):
    # t=dt*snapshot_step*time_index
    t=times[time_index]
    b=np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t))
    fig.add_trace(go.Histogram(x=xst[:,time_index], histnorm='probability density', name=f"t = {round(t,1)}"))
    fig.add_trace(go.Scatter(x=xx,y=b, name = f"Eq. distrib."))
fig.update_layout(barmode='overlay',
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
fig.update_traces(opacity=0.75)
fig.show()


# In[44]:


times, xst = many_sims_parallel(tot_sims=1000000, dt = dt, snapshot_step = snapshot_step)


# In[45]:


dt = 0.001
snapshot_step = 1000
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
fig=go.Figure()
for time_index in range(1,11):
    # t=dt*snapshot_step*time_index
    t=times[time_index]
    b=np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t))
    fig.add_trace(go.Histogram(x=xst[:,time_index], histnorm='probability density', name=f"t = {round(t,1)}"))
    fig.add_trace(go.Scatter(x=xx,y=b, name = f"Eq. distr."))
fig.update_layout(barmode='overlay',
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
fig.update_traces(opacity=0.75)
fig.show()


# In[58]:



fig=go.Figure(frames = [go.Frame(
                data=go.Histogram(x=xst[:,time_index], 
                                  histnorm='probability density',
                                  name=f't = {round(times[time_index])}')) 
                            for time_index in range(1,11)],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.add_trace(go.Histogram(x=xst[:,0], 
                                  histnorm='probability density',
                                  name='t = 0.0'))
fig.show()


# In[ ]:





# In[87]:


dt = 0.001
snapshot_step = 1000
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Histogram(x=xst[:,0], 
                                  histnorm='probability density',
                                  name='t = 0.0'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[go.Histogram(x=xst[:,time_index], 
                                 histnorm='probability density',
                                 name=f't={times[time_index]}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index])}",
                          traces=[0, 1]) for time_index in range(1,11)],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.show()


# In[92]:


xst[:,2]


# In[93]:


np.histogram(xst[:,2])


# In[96]:


np.histogram(xst[:,2], density=True, range=[-3.0,6.0])


# In[106]:


histo=np.histogram(xst[:,2], density=True, range=[-3.0,6.0], bins=50)
fig=go.Figure(data=[go.Bar(x=histo[1], y=histo[0])])
fig.update_layout(bargap=0)


# In[117]:


dt = 0.001
snapshot_step = 1000
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
histos=[np.histogram(xst[:,ti], density=True, range=[-3.0,6.0], bins=50) for ti in range(0,11)]
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                  name='t = 0.0'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[
                              go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                 name=f't = {round(times[time_index],1)}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index])}",
                          traces=[0, 1]) for time_index in range(1,11)],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.update_layout(bargap=0)
fig.show()


# In[118]:


dt = 0.001
snapshot_step = 1000
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
histos=[np.histogram(xst[:,ti], density=True, range=[-3.0,6.0], bins=100) for ti in range(0,11)]
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                  name='t = 0.0'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[
                              go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                 name=f't = {round(times[time_index],1)}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index])}",
                          traces=[0, 1]) for time_index in range(1,11)],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.update_layout(bargap=0)
fig.show()


# In[124]:


times, xst = many_sims_parallel(tot_sims=1000000, dt = dt, snapshot_step = 100)


# In[125]:


dt = 0.001
snapshot_step = 100
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
histos=[np.histogram(xst[:,ti], density=True, range=[-3.0,6.0], bins=100) for ti in range(0,len(times))]
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                  name='t = 0.00'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[
                              go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                 name=f't = {round(times[time_index],1)}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index],3)}",
                          traces=[0, 1]) for time_index in range(1, len(times))],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.update_layout(bargap=0)
fig.show()


# In[129]:


dt = 0.001
snapshot_step = 100
def k(t):
  return 1.0+1.0*t
def center(t):
  return 0.5*t
def f(x,t):
  return -k(t)*(x-center(t))
xx=np.linspace(-3.0, 6.0, 1000)
histos=[np.histogram(xst[:,ti], density=True, range=[-3.0,6.0], bins=300) for ti in range(0,len(times))]
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                  name='t = 0.00'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[
                              go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                 name=f't = {round(times[time_index],1)}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index],3)}",
                          traces=[0, 1]) for time_index in range(1, len(times))],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.update_layout(bargap=0)
fig.show()


# In[131]:


# Now the force is a global function
@nb.njit
def k(t):
    if t<8.0:
        return 1.0+1.0*t
    else:
        return 1.0+1.0*8.0
@nb.njit
def center(t):
    if t<8.0:
        return 0.5*t
    else:
        return 0.5*8.0
@nb.njit
def f(x,t):
  return -k(t)*(x-center(t))


# In[132]:


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


# In[136]:


@nb.jit(parallel=True)
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x_sims=np.zeros((tot_sims, tot_snapshots))
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        x_sims[sim_num]=one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x_sims


# In[137]:


times, xst = many_sims_parallel(tot_sims=1000000, dt = dt, snapshot_step = 100)


# In[138]:


xx=np.linspace(-3.0, 6.0, 1000)
histos=[np.histogram(xst[:,ti], density=True, range=[-3.0,6.0], bins=300) for ti in range(0,len(times))]
b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                  name='t = 0.00'),
                     go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
              frames = [go.Frame(
                          data=[
                              go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                 name=f't = {round(times[time_index],1)}'),
                                go.Scatter(x=xx,y=b[time_index])
                               ],
                          name=f"t={round(times[time_index],3)}",
                          traces=[0, 1]) for time_index in range(1, len(times))],
            layout = go.Layout(
                xaxis=dict(range=[-3, 6], autorange=False),
                yaxis=dict(range=[-0, 1.4], autorange=False),
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(label="Play",
                                  method="animate",
                                  args=[None])])],
                    xaxis_title="x",
                    yaxis_title="P(x,t)")
             )
fig.update_layout(bargap=0)
fig.show()


# In[145]:


# If the force is changed, all code has to be recompiled
@nb.njit
def k(t):
    if t<5.0:
        return 1.0+1.0*t
    else:
        return 1.0+1.0*5.0
@nb.njit
def center(t):
    if t<5.0:
        return 0.5*t
    else:
        return 0.5*5.0
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
def many_sims_parallel(tot_sims = 1000, dt = 0.001, tot_steps =10000, xinit=0.0, noise_scaler=1.0, snapshot_step=100):
    tot_snapshots = int(tot_steps/snapshot_step)+1
    x_sims=np.zeros((tot_sims, tot_snapshots))
    times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
    for sim_num in nb.prange(tot_sims):
        x_sims[sim_num]=one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x_sims


# In[146]:


times, xst = many_sims_parallel(tot_sims=1000000, dt = dt, snapshot_step = 100)


# In[141]:


def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300):
    xx=np.linspace(-3.0, 6.0, 1000)
    histos=[np.histogram(xst[:,ti], density=True, range=x_range, bins=bins) for ti in range(0,len(times))]
    b=[np.exp(-0.5*k(t)*(xx-center(t))**2)/np.sqrt(2*np.pi/k(t)) for t in times]
    fig=go.Figure(data = [go.Bar(x=histos[0][1], y=histos[0][0],
                                      name='t = 0.00'),
                         go.Scatter(x=xx,y=b[0], name = f"Eq. distr.")],
                  frames = [go.Frame(
                              data=[
                                  go.Bar(x=histos[time_index][1], y=histos[time_index][0],
                                     name=f't = {round(times[time_index],1)}'),
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


# In[148]:


anim=animate_simulation(times, xst)


# In[149]:


anim


# In[150]:


anim.write_html("chasing.html")


# In[ ]:




