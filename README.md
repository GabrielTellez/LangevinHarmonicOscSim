# LangevinHarmonicOscSim

Simulation of a brownian particle in a harmonic potential

## "langevin_harmonic_osc_simulator.py"

Library for simulation of a brownian particle on a time-dependent
harmonic potential

## Simulation and Simulator classes

### class Simulation

- Simulation(tot_sims, dt, tot_steps, noise_scaler, snapshot_step, k, center, results, name='')

- Stores simulation parameters and results.

- Analyses the results: builds PDF of the simulation results (position work, etc..)

### class Simulator

- Simulator(tot_sims=1000, dt=0.001, tot_steps=10000, noise_scaler=1.0, snapshot_step=100, k=<function k>, center=<function center>)
- Simulator class for Langevin dynamics of a harmonic oscillator with
  variable potential. Encapsulates the simulator, perform
  simulations, analyses them and store results
  of simulations.

### For further details, see the documentation [html](doc.html) or [plain text](doc.txt).

### make_simulator:

Creates a compiled function to simulate a brownian
particle on a time-dependent harmonic potential

make_simulator(tot_sims = 1000, dt = 0.001, tot_steps =10000, noise_scaler=1.0, snapshot_step=100, k=k, center=center):

Makes a numba compiled njit langevin simulator of a brownian
particle in a harmonic potential with a given stiffness function k and center

Args:

- tot_sims (int, optional): default total number of simulations. Defaults to 1000.
- dt (float, optional): default time step. Defaults to 0.001.
- tot_steps (int, optional): default number of steps of each simulation. Defaults to 10000.
- noise_scaler (float, optional): brownian noise scale k_B T. Defaults to 1.0.
- snapshot_step (int, optional): save a snapshot of simulation at
  each snapshot_step time. Defaults to 100.
- k (float, optional): stiffness function k(t) of the potential. Defaults to k(t)=1.0.
- center (float, optional): center function of the potential. Defaults to center(t)=0.0.

Returns:

njitted function: numba compiled function that performs simulation

Example:

    # creates a simulator with functions k_function(t) for the stiffness and center_function(t) for the center
    simulator = make_simulator(tot_sims=1000000, dt=0.00001, tot_steps=10000, k=k_function, center=center_function)
    # runs the simulator
    times, x, power, work, heat, delta_U, energy = simulator()
    """ Returns:
        times, x, power, work, heat, delta_U, energy:
          times: list of times of the snapshots
          x, power, work, heat, delta_U, energy = list of positions,
          power, ..., snapshots for each simulation
          ie. x[sim] = [x(0), x(snapshot_step*dt), x(2*snapshot_step*dt), .... ] for simulation number sim.
    """

## Graphics utilities

### animate_simulation

animate_simulation(times, xst, x_range=[-3.0, 6.0], y_range=[0, 1.5], bins=300, x_label='x', y_label='P(x,t)', show_x_eq_distrib=True, k=k, center=center):

Plot and animates a simulation data results

Args:

- times (list of float): list of times where snapshots where taken
- xst (list of list of float): list of snapshots of many
  simulations. Should have shape (tot_sims, tot_snapshots)
- x_range (list, optional): range of the data to plot. Defaults to [-3.0, 6.0].
- y_range (list, optional): range of the histogram of xst. Defaults to [0, 1.5].
- bins (int, optional): bins to compute histogram of xst. Defaults to 300.
- x_label (str, optional): label for xst in the plot. Defaults to 'x'.
- y_label (str, optional): label for the probability density of xst. Defaults to 'P(x,t)'.
- show_x_eq_distrib (bool, optional): show the equilibrium
  distribution corresponding to a harmonic oscilator with center(t)
  and stiffness k(t). Defaults to True.
- k (float, optional): stiffness function of the potential. Defaults to k(t)=1.0.
- center (float, optional): center function of the potential. Defaults to center(t)=0.0.

Returns:
Plotly graphics object: animation of the simulation data

### plot_quantity

plot_quantity(t_array, y_array, t_range=None, y_range=None t_label='t',
y_label='')

Plots y_array as function of t_array

Args:

- t_array (np.array): time axis array of
- y_array (np.array): quantity to plot array
- t_range (list, optional): t range. Defaults to Autoscale.
- y_range (list, optional): y range. Defaults to Autoscale.
- t_label (str, optional): label for t axis. Defaults to 't'.
- y_label (str, optional): label for y axis. Defaults to ''.

Returns:

- Plotly graphic object: the plot of the quantity

## "test-mylibrary.ipynb"

Jupyter notebook illustrating the use of the simulation library.

## "test-simulation-class.ipynb"

Jupyter notebook illustration the use of the classes Simulator and Simulation

## tests/

Tests for use with pytest for the simulator and simulation classes

## scripts/

Scripts to run jupyter notebook remotely on a SLURM cluster

## devel_code/

Developement and test code (undocumented)
