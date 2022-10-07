import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import numba as nb
import cloudpickle
import pickle

def make_simulator(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100,
                  k=None, center=None,
                  harmonic_potential=True,
                  force=None, potential=None,
                  initial_distribution = None):
  """Makes a numba compiled njit langevin simulator of a brownian
  particle in an external time variable potential or force 

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
  if harmonic_potential:
    if k == None:
      def k(t):
        """Default stiffness for the harmonic potential
          t |--> 1.0
        """
        return 1.0
    if center == None:
      def center(t):
        """Default center for the harmonic potential
          t |--> 0.0
        """
        return 0.0
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
    if initial_distribution == None:
      @nb.njit
      def initial_distribution():
        """Samples initial position according to the equilibrium distribution

        Returns:
            float: x random sample distributed according to exp(-k(0)*(x-center(0)**2/2))
        """
        return np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
        # To be implemented for non harmonic potential,
        # see inverse transform sampling
        # Challenge: it has to be numba njittable.
    else:
      initial_distribution=nb.njit(initial_distribution)

  else:
    # In general force mode
    # check that the force or the potential are defined
    if force == None and potential == None:
      raise ValueError("In general force mode, the force or the potential have to be provided. Both cannot be None")    
    if force == None:
      # We need to compute the force from the potential
      U=nb.njit(potential)
      def force(x,t):
        dx=1E-9
        return -(U(x+dx,t)-U(x,t))/dt
    if potential == None:
      raise ValueError("In general force mode, the potential have to be provided. It is too slow to compute it from the force.")    
      # We need the potential from the force
      # this will never run:
      f=nb.njit(force)
      def potential(x,t):
        # Integral by basic numerical quadrature trapezoidal rule
        # This is way too slow
        dx=1E-6
        x0=0.0
        xs = np.arange(x0+dx, x-dx, dx)
        integral=0.5*(f(x0,t)+f(x,t))
        for xp in xs:
          integral += f(xp,t)
        integral = integral*dx 
        return integral
    if initial_distribution == None:
      raise ValueError('In general force mode the initial distribution has to be provided')

    # To do : test if potential and force are coherent
    ####
    f=nb.njit(force)
    U=nb.njit(potential)
    initial_distribution=nb.njit(initial_distribution)

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
        # initial position taken from a given initial_distribution
        xinit = initial_distribution()
        x[sim_num], power[sim_num], work[sim_num], heat[sim_num], delta_U[sim_num], energy[sim_num] = one_simulation(dt=dt, tot_steps=tot_steps, xinit=xinit, noise_scaler=noise_scaler, snapshot_step=snapshot_step)
    return times, x, power, work, heat, delta_U, energy
  return many_sims_parallel

################################################################################

def animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300, x_label='x', y_label='P(x,t)', show_x_eq_distrib=True, k=None, center=None):
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
  if k == None:
    def k(t):
      """Default stiffness for the harmonic potential
        t |--> 1.0
      """
      return 1.0
  if center == None:
    def center(t):
      """Default center for the harmonic potential
        t |--> 0.0
      """
      return 0.0
  xx=np.linspace(*x_range, 1000)
  histos=[np.histogram(xst[:,ti], density=True, range=x_range, bins=bins) for ti in range(0,len(times))]
  # To do: plot general PDF exp(-U(x,t))
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

################################################################################

def plot_quantity(t_array, y_array,
                  t_range=None, y_range=None,
                  t_label ='t', y_label=''):
  """Plots y_array as function of t_array

  Args:
      t_array (np.array): time axis array of
      y_array (np.array): quantity to plot array
      t_range (list, optional): t range. Defaults to Autoscale.
      y_range (list, optional): y range. Defaults to Autoscale.
      t_label (str, optional): label for t axis. Defaults to 't'.
      y_label (str, optional): label for y axis. Defaults to ''.

  Returns:
      Plotly graphic object: the plot of the quantity
  """
  # make figure
  fig_dict = {
    "data": [],
    "layout": {}
  }

  if t_range == None:
    xaxis_dict = dict(autorange=True)
  else:
    xaxis_dict = dict(range=t_range, autorange=False)
  if y_range == None:
    yaxis_dict = dict(autorange=True)
  else:
    yaxis_dict = dict(range=y_range, autorange=False)
  fig_dict["layout"] = go.Layout(
                        xaxis=xaxis_dict,
                        yaxis=yaxis_dict,
                        xaxis_title=t_label,
                        yaxis_title=y_label )
  fig_dict["data"].append(
    go.Scatter(x=t_array, y=y_array, name=y_label)
  ) 
  fig=go.Figure(fig_dict)
  return fig

################################################################################

class Simulation:
  """Stores simulation parameters and results. 
  Analyses the results: builds PDF of the simulation results (position,
  work, etc..)
  """
  result_labels = ["x", "power", "work", "heat", "delta_U", "energy"]
  def __init__(self, tot_sims, dt, tot_steps, noise_scaler, snapshot_step, k, center, results, name=""):
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
        name (string, optional): name of the simulation
    """
    self.tot_sims = tot_sims
    self.dt = dt
    self.tot_steps = tot_steps
    self.noise_scaler = noise_scaler
    self.snapshot_step = snapshot_step
    self.name = name
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
    self.histogram = {}
    self.pdf = {}
    self.averages = {} 
    self.average_func = {}
    self.variances = {}
    self.variance_func = {}
    
  def __str__(self):
    return f'Simulation "{self.name}"'

  def build_histogram(self, quantity, bins = 300, q_range = None):
    """Builds the histogram of a quantity

    Args:
        quantity (string): quantity to build its histogram. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
        bins (int, optional): bins for the histogram. Defaults to 300.
        q_range (list, optional): range for the quantity. Defaults to
        None for automatic range.
    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    self.histogram[quantity]=np.array([np.histogram(self.results[quantity][:,ti], density=True, range=q_range, bins=bins) for ti in range(0,len(self.results["times"]))], dtype=object)
 
  def build_pdf(self, quantity):
    """Builds the probability density function (PDF) for a quantity.
    The PDF is build and function is defined to access it in self.pdf(quantity)

    Args:
        quantity (string): quantity to build its pdf. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]
    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    if quantity not in self.histogram.keys():
      # Build the histogram if not previously build
      self.build_histogram(quantity)
    def pdf(x, t):
      # time t to snapshot index ti
      bins_t = self.results['times']
      if t < np.min(bins_t) or t > np.max(bins_t):
        raise ValueError(f"In PDF of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]")
      ti = np.digitize(t, bins_t) - 1
      if ti < 0: 
        ti=0
      
      # self.histogram[quantity][ti, 0] # contains P(x)
      # self.histogram[quantity][ti, 1] # contains x
      # get the index corresponding to value x in the bins
      bins_x = self.histogram[quantity][ti, 1]
      if x < np.min(bins_x) or x > np.max(bins_x):
        raise ValueError(f"{quantity}={x} is out of bounds [{np.min(bins_x)}, {np.max(bins_x)}")
    
      index_x = np.digitize(x, bins_x) - 1
      if index_x < 0: 
        index_x=0
      return self.histogram[quantity][ti, 0][index_x]
    self.pdf[quantity] = pdf    

  def build_averages(self, quantity):
    """Computes the average of a quantity. 
    The average at time t (with corresponding time_index of the snapshot)
    is stored in averages[quantity][time_index]
    A function giving the average as a function of time is created and
    stored in average_func(quantity)

    Args:
        quantity (string): quantity to build its averages. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

    Raises:
        ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    self.averages[quantity] = np.average(self.results[quantity], axis=0)
    def av_fnct(t):
      # time t to snapshot index ti
      bins_t = self.results['times']
      if t < np.min(bins_t) or t > np.max(bins_t):
        raise ValueError(f"In average of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]")
      ti = np.digitize(t, bins_t) - 1
      if ti < 0: 
        ti=0
      return self.averages[quantity][ti]
    self.average_func[quantity] = av_fnct 

  def build_variances(self, quantity):
    """Computes the variance of a quantity. 
    The variance at time t (with corresponding time_index of the snapshot)
    is stored in variances[quantity][time_index]
    A function giving the variance as a function of time is created and
    stored in variance_func(quantity)

    Args:
        quantity (string): quantity to build its variances. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

    Raises:
        ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    self.variances[quantity] = np.var(self.results[quantity], axis=0)
    def var_fnct(t):
      # time t to snapshot index ti
      bins_t = self.results['times']
      if t < np.min(bins_t) or t > np.max(bins_t):
        raise ValueError(f"In average of {quantity}: time={t} is out of bounds [{np.min(bins_t)}, {np.max(bins_t)}]")
      ti = np.digitize(t, bins_t) - 1
      if ti < 0: 
        ti=0
      return self.variances[quantity][ti]
    self.variance_func[quantity] = var_fnct 

  def animate_pdf(self,quantity, x_range=[-3.0, 3.0], y_range=[0, 1.5], bins=300, show_x_eq_distrib=None):
    """Shows an animation of the evolution of the PDF of a quantity

    Args:
        quantity (string): quantity to animate its PDF. Must be in ["x", "power", "work", "heat", "delta_U", "energy"]
        x_range (list, optional): range for the quantity in the PDF. Defaults to [-3.0, 3.0].
        y_range (list, optional): range for the PDF value. Defaults to [0, 1.5].
        bins (int, optional): bins for the histogram. Defaults to 300.
        show_x_eq_distrib (boolean, optional): if True the instantaneous
        equilibrium position distribution is shown. Defaults to None.

    Raises:
        ValueError: quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

    Returns:
        Plotly graphics object: animation of the PDF
    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    if show_x_eq_distrib == None:
      show_x_eq_distrib = (quantity == 'x')
    return animate_simulation(self.results['times'], self.results[quantity], 
                       x_range=x_range, y_range=y_range, 
                       bins=bins, 
                       x_label=quantity, y_label=f'P({quantity},t)', 
                       show_x_eq_distrib=show_x_eq_distrib, 
                       k=self.k, center=self.center)

  def save(self, filename):
    """Saves the simulation

    Args:
        filename (string): filename where the simulation is saved
    """
    with open(filename, 'wb') as f:
      cloudpickle.dump(self, f, pickle.DEFAULT_PROTOCOL)

  def load(filename):
    """Loads a simulation from file

    Args:
        filename (string): filename of the simulation to load the

    Returns:
        Simulation: the loaded simulation
    """
    with open(filename, 'rb') as f:
      _sim = pickle.load(f)
    return _sim

  def analyse(self):
    """Builds all histogram, PDF, averages and variances"""
    for k in self.result_labels:
      self.build_histogram(k)
      self.build_pdf(k)
      self.build_averages(k)
      self.build_variances(k)

  def plot_average(self, quantity, 
                  t_range=None, y_range=None,
                  t_label ='t', y_label=None  ):
    """Plots <quantity> as a function of time

    Args:
        quantity (string): quantity to plot. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

    Raises:
        ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

    Returns:
        Plotly graphics object: plot of the quantity
    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    if quantity not in self.averages:
      self.build_averages(quantity)
    if y_label == None:
      y_label = quantity
    # make figure
    fig = plot_quantity(self.results['times'], self.averages[quantity],
                  t_range=t_range, y_range=y_range,
                  t_label=t_range, y_label=y_label)
    return fig

  def plot_variance(self, quantity, 
                  t_range=None, y_range=None,
                  t_label ='t', y_label=None):
    """Plots the variance of quantity as a function of time

    Args:
        quantity (string): quantity to plot. Should be in ["x", "power", "work", "heat", "delta_U", "energy"]

    Raises:
        ValueError: if quantity is not in ["x", "power", "work", "heat", "delta_U", "energy"]

    Returns:
        Plotly graphics object: plot of the quantity
    """
    if quantity not in self.result_labels:
      raise ValueError(f"quantity {quantity} must be in {self.result_labels}")
    if quantity not in self.variances:
      self.build_variances(quantity)
    if y_label == None:
      y_label = f'Var({quantity})'
    # make figure
    fig = plot_quantity(self.results['times'], self.variances[quantity],
                  t_range=t_range, y_range=y_range,
                  t_label=t_range, y_label=y_label)
    return fig

##################################################################################

class Simulator:
  """Simulator class for Langevin dynamics of a harmonic oscillator with
  variable potential. Encapsulates the simulator, perform
  simulations, analyses them and store results
  of simulation 
  """
  def __init__(self, tot_sims = 1000, dt = 0.001, tot_steps = 10000, noise_scaler=1.0, snapshot_step=100,
              k=None, center=None, harmonic_potential = True,
              force = None, potential = None, initial_distribution=None):
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
        harmonic_potential (boolean, optional): it True: the external potential
          is harmonic with stiffness k(t) and center(t).
          If False the external force is given by the force argument or
          the the external potential is given by potential argument
        force (float function(x,t), optional): the external force
        potential (float function(x,t), optional): the external potential
        initial_distribution (float function(), optional): initial
          condition for x(0). Default: for harmonic oscillator, sampled
          from equilibrium distribution exp(-k(0)(x-center(0)**2/2).
          Have to be provided for general potential if harmonic_potential=False
    """
    initial_distribution_not_compiled = initial_distribution
    if harmonic_potential:
      if k == None:
        def k(t):
          """Default stiffness for the harmonic potential
            t |--> 1.0
          """
          return 1.0
      if center == None:
        def center(t):
          """Default center for the harmonic potential
            t |--> 0.0
          """
          return 0.0
      def force(x,t):
        """ Force on the particle"""
        return -k(t)*(x-center(t))

      def potential(x,t):
        """ Harmonic potential energy"""
        return 0.5*k(t)*(x-center(t))*(x-center(t))
      if initial_distribution_not_compiled == None:
        def initial_distribution_not_compiled():
          """Samples initial position according to the equilibrium distribution
          Returns:
              float: x random sample distributed according to exp(-k(0)*(x-center(0)**2/2))
          """
          return np.random.normal(center(0.0), scale=np.sqrt(1.0/k(0.0)))
     
    # store the default parameters for simulations
    self.tot_sims = tot_sims
    self.dt = dt
    self.tot_steps = tot_steps
    self.noise_scaler = noise_scaler
    self.snapshot_step = snapshot_step
    self.k = k
    self.center = center
    self.harmonic_potential = harmonic_potential
    self.force = force
    self.potential = potential
    self.initial_distribution = initial_distribution_not_compiled
    self.simulator = make_simulator(tot_sims=tot_sims, dt=dt, tot_steps=tot_steps, noise_scaler=noise_scaler, snapshot_step=snapshot_step, k=k, center=center, 
                                    harmonic_potential=harmonic_potential,
                                    force=force, potential=potential, initial_distribution=initial_distribution)
    self.simulations_performed = 0
    # list of Simulations classes to store results of simulations
    self.simulation = []

  def run(self, tot_sims=None, dt=None, tot_steps=None, noise_scaler=None, snapshot_step=None, name=""):
    """Runs a simulation and store the results

    Args:
      tot_sims (int, optional): total number of simulations. 
      dt (float, optional): time step. 
      tot_steps (int, optional): total steps of each simulation. 
      noise_scaler (float, optional): brownian noise scale k_B T. 
      snapshot_step (int, optional): save a snapshot of simulation at
        each snapshot_step time. 
      name (str, optional): name of the simulation
    """
    if tot_sims == None:
      tot_sims = self.tot_sims
    if dt == None:
      dt = self.dt
    if tot_steps == None:
      tot_steps = self.tot_steps
    if noise_scaler == None:
      noise_scaler = self.noise_scaler
    if snapshot_step == None:
      snapshot_step = self.snapshot_step

    results = self.simulator(tot_sims, dt, tot_steps, noise_scaler, snapshot_step)
    sim = Simulation(tot_sims, dt, tot_steps, noise_scaler, snapshot_step, self.k, self.center, results, name)
    self.simulation.append(sim)
    self.simulations_performed += 1

  def analyse(self, sim_num=None):
    """Performs the analysis of simulation number sim_num

    Args:
        sim_num (int, optional): simulation number. Defaults to last
        simulation preformed.
    """
    if sim_num == None:
      sim_num = self.simulations_performed - 1
    if sim_num < 0 or sim_num > self.simulations_performed - 1:
      raise ValueError(f"Simulation number {sim_num} does not exists")

    self.simulation[sim_num].analyse()
