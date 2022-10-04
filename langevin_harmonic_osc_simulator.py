import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb

def k(t):
  """Default stiffness for the harmonic potential"""
  return 1.0
def center(t):
  """Default center for the harmonic potential"""
  return 0.0
def make_simulator(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100, k=k, center=center):
  """Makes a numba compiled njit langevin simulator of a brownian
  particle in a harmonic potential with a given stiffness function k and center

  Args:
      tot_sims (int, optional): default total number of simulations. Defaults to 1000.
      dt (float, optional): default time step. Defaults to 0.001.
      tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
      noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
      snapshot_step (int, optional): save a snapshot of simulation at
      each snapshot_step time. Defaults to 100.
      k (float function, optional): stiffness function k(t) of the potential. Defaults to k(t)=1.0.
      center (float function, optional): center function of the potential. Defaults to center(t)=0.0.

  Returns:
      njitted function: numba compiled function that performs simulation 
  """
  k = nb.njit(k)
  center = nb.njit(center)
  @nb.njit
  def f(x,t):
    """ Force on the particle"""
    return -k(t)*(x-center(t))
  @nb.njit
  def U(x,t):
    """ Harmonic potential energy"""
    return 0.5*k(t)*(x-center(t))*(x-center(t))
  @nb.njit
  def one_simulation(dt=dt, tot_steps=tot_steps, xinit=0.0, noise_scaler=noise_scaler, snapshot_step=snapshot_step):
    """Function that performs one simulation

    Args:
        tot_sims (int, optional): default total number of simulations. Defaults to 1000.
        dt (float, optional): default time step. Defaults to 0.001.
        tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
        noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
        snapshot_step (int, optional): save a snapshot of simulation at each snapshot_step time. Defaults to 100.
        

    Returns:
        x, power, work, heat, delta_U, energy (tuple): array of
          snapshots of the simulation
          x = position, power, work, heat, delta_U=energy difference between
           current state and initial state, energy

    """
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
    """Function that performs many simulations with initial condition at
    thermal equilibrium at t=0

    Args:
      tot_sims (int, optional): default total number of simulations. Defaults to 1000.
      dt (float, optional): default time step. Defaults to 0.001.
      tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
      noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
      snapshot_step (int, optional): save a snapshot of simulation at
      each snapshot_step time. Defaults to 100.

    Returns:
        times, x, power, work, heat, delta_U, energy: 
          times: list of times of the snapshots
          x, power, work, heat, delta_U, energy = list of positions,
          power, ..., snapshots for each simulation
          ie. x[sim] = [x(0), x(snapshot_step*dt), x(2*snapshot_step*dt), .... ] for simulation number sim.
    """
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

def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300, x_label='x', y_label='P(x,t)', show_x_eq_distrib=True, k=k, center=center):
  """Plot and animates a simulation data results

  Args:
      times (list of float): list of times where snapshots where taken
      xst (list of list of float): list of snapshots of many
        simulations. Should have shape (tot_sims, tot_snapshots)  
      x_range (list, optional): range of the data to plot. Defaults to [-3.0, 6.0].
      y_range (list, optional): range of the histogram of xst. Defaults to [0, 1.5].
      bins (int, optional): bins to compute histogram of xst. Defaults to 300.
      x_label (str, optional): label for xst in the plot. Defaults to 'x'.
      y_label (str, optional): label for the probability density of xst. Defaults to 'P(x,t)'.
      show_x_eq_distrib (bool, optional): show the equilibrium
        distribution corresponding to a harmonic oscilator with center(t)
        and stiffness k(t). Defaults to True.
      k (float function, optional): stiffness function of the potential. Defaults to k(t)=1.0.
      center (float function, optional): center function of the potential. Defaults to center(t)=0.0.

  Returns:
      Plotly graphics object: animation of the simulation data
  """
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

class Simulation:
  """Stores simulation parameters and results. 
  Analyses the results: builds PDF of the simulation results (position,
  work, etc..)
  """
  result_labels = ["times", "x", "power", "work", "heat", "delta_U", "energy"]
  def __init__(self, tot_sims, dt, tot_steps, noise_scaler, snapshot_step, k, center, results):
    """Initializes the Simulation class with parameters and raw results

    Args:
        tot_sims (int): total number of simulations.
        dt (float): time step.
        tot_steps (int): number of steps of each simulation.
        noise_scaler (float): brownian noise scale k_B T. Defaults to 1.0.
        snapshot_step (int): a snapshot of simulation has been saved each snapshot_step time.
        k (float function): stiffness of the potential
        center (float function): center of the potential
        results (tuple): results in the form (times, x, power, work, heat, delta_U, energy) where
          times (ndarray): ndarray of times where snapshot where taken
          x (ndarray of shape (tot_sims, tot_snapshots)): x[sim][ts] = position of
            the brownian particle in simulation number num and snapshot ts
          power (ndarray of shape (tot_sims, tot_snapshots)): power[sim][ts] = power into
            the system at snapshot ts and simulation sim
          work (ndarray of shape (tot_sims, tot_snapshots)): work[sim][ts] perfomed into
            the system in simulation sim up to snapshot ts
          heat (ndarray of shape (tot_sims, tot_snapshots)): heat[sim][ts] into
            the system in simulation sim up to snapshot ts
          delta_U (ndarray of shape (tot_sims, tot_snapshots)): energy[sim][ts]
            difference between snapshot = 0 and current snapshot ts in
            simulation sim 
          energy (ndarray of shape (tot_sims, tot_snapshots)):
          energy[sim][ts] in simulation sim at snapshot ts 
    """
    self.tot_sims = tot_sims
    self.dt = dt
    self.tot_steps = tot_steps
    self.noise_scaler = noise_scaler
    self.snapshot_step = snapshot_step
    self.k = k 
    self.center = center 

    (times, x, power, work, heat, delta_U, energy) = results
    self.results = {
      'times': times,
      'x': x,
      'power': power,
      'work': work,
      'heat': heat,
      'delta_U': delta_U,
      'energy': energy
    }


class Simulator:
  """Simulator class for Langevin dynamics of a harmonic oscillator with
  variable potential. Encapsulates the simulator, perform
  simulations, analyses them and store results
  of simulation 
  """
  def __init__(self, tot_sims = 1000, dt = 0.001, tot_steps = 10000, noise_scaler=1.0, snapshot_step=100, k=k, center=center):
    """Initializes the Simulator

    Args:
        tot_sims (int, optional): total number of simulations. Defaults to 1000.
        dt (float, optional): time step. Defaults to 0.001.
        tot_steps (int, optional): total steps of each simulation. Defaults to 10000.
        noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
        snapshot_step (int, optional): save a snapshot of simulation at
        each snapshot_step time. Defaults to 100.
        k (float function, optional): stiffness function k(t) of the potential. Defaults to k(t)=1.0.
        center (float function, optional): center function of the potential. Defaults to center(t)=0.0.
    """

    # store the default parameters for simulations
    self.tot_sims = tot_sims
    self.dt = dt
    self.tot_steps = tot_steps
    self.noise_scaler = noise_scaler
    self.snapshot_step = snapshot_step
    self.k = k
    self.center = center
    self.simulator = make_simulator(tot_sims=tot_sims, dt=dt, tot_steps=tot_steps, noise_scaler=noise_scaler, snapshot_step=snapshot_step, k=k, center=center)

    self.simulations_performed = 0
    # list of Simulations classes to store results of simulations
    self.simulation = []

  def run(self, tot_sims, dt, tot_steps, noise_scaler, snapshot_step):
    """Runs a simulation and store the results

    Args:
      tot_sims (int, optional): total number of simulations. 
      dt (float, optional): time step. 
      tot_steps (int, optional): total steps of each simulation. 
      noise_scaler (float, optional): brownian noise scale k_B T. 
      snapshot_step (int, optional): save a snapshot of simulation at
        each snapshot_step time. 
    """

    results = self.simulator(tot_sims, dt, tot_steps, noise_scaler, snapshot_step)
    sim = Simulation(tot_sims, dt, tot_steps, noise_scaler, snapshot_step, self.k, self.center, results)
    self.simulation.append(sim)
    self.simulations_performed += 1
