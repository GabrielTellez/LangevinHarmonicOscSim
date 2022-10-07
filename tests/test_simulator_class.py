from multiprocessing.sharedctypes import Value
from ..langevin_harmonic_osc_simulator import Simulator
import numpy as np
import pytest


def test_init_Simulator():
  """
  tests the initialization of a simulator class.
  """
  attributes = ["tot_sims", "dt", "tot_steps", "noise_scaler", "snapshot_step", "k", "center", "simulator"]
  simulator = Simulator()
  for attr in attributes:
    assert hasattr(simulator, attr)
  assert callable(simulator.simulator)

def test_run_simulation_store_parameters():
  """
  tests the simulator class runs a simulation and correctly store the
  simulation parameters.
  """

  tot_sims = 1000
  dt = 0.0001
  tot_steps = 1000
  noise_scaler = 1
  snapshot_step = 100

  simulator = Simulator()
  assert simulator.simulations_performed == 0
  simulator.run(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler = noise_scaler, snapshot_step = snapshot_step)
  assert simulator.simulations_performed == 1
  assert len(simulator.simulation) == 1
  assert tot_sims == simulator.simulation[0].tot_sims
  assert tot_steps == simulator.simulation[0].tot_steps
  assert dt == simulator.simulation[0].dt
  assert noise_scaler == simulator.simulation[0].noise_scaler
  assert snapshot_step == simulator.simulation[0].snapshot_step

def test_run_simulation_store_results_shape():
  """Test correct shape of results
  """
  tot_sims = 1000
  dt = 0.0001
  tot_steps = 1000
  noise_scaler = 1
  snapshot_step = 100

  simulator = Simulator()
  simulator.run(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler = noise_scaler, snapshot_step = snapshot_step)
  shape = (tot_sims, int(tot_steps/snapshot_step) + 1)
  sim_res = simulator.simulation[0].results
  assert len(sim_res['times']) == int(tot_steps/snapshot_step) + 1
  assert np.shape(sim_res['x']) == shape
  assert np.shape(sim_res['power']) == shape
  assert np.shape(sim_res['work']) == shape
  assert np.shape(sim_res['heat']) == shape
  assert np.shape(sim_res['delta_U']) == shape
  assert np.shape(sim_res['energy']) == shape


def test_run_simulation_default_parameters():
  """
  tests is run of the simulator without arguments runs with the 
  parameters provided in the constructor.
  """
  simulator = Simulator()
  assert simulator.simulations_performed == 0
  simulator.run()
  assert simulator.simulations_performed == 1
  assert len(simulator.simulation) == 1
  assert simulator.tot_sims == simulator.simulation[0].tot_sims
  assert simulator.tot_steps == simulator.simulation[0].tot_steps
  assert simulator.dt == simulator.simulation[0].dt
  assert simulator.noise_scaler == simulator.simulation[0].noise_scaler
  assert simulator.snapshot_step == simulator.simulation[0].snapshot_step


def test_run_simulation_store_name():
  """
  tests the simulator class runs a named simulation and correctly store
  its name
  """

  tot_sims = 1000
  dt = 0.0001
  tot_steps = 1000
  noise_scaler = 1
  snapshot_step = 100
  name = "test simulation"

  simulator = Simulator()
  assert simulator.simulations_performed == 0
  simulator.run(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler = noise_scaler, snapshot_step = snapshot_step, name=name)
  assert name == simulator.simulation[0].name

def assert_sim_analysis(sim):
  """Asserts if a simulation has perfomed its analysis"""
  for k in sim.result_labels:
    assert k in sim.histogram
    assert k in sim.pdf
    assert k in sim.averages
    assert k in sim.average_func
    assert k in sim.variances 
    assert k in sim.variance_func

def test_analyse_last_simulation():
  tot_sims = 1000
  dt = 0.0001
  tot_steps = 1000
  noise_scaler = 1
  snapshot_step = 100
  name = "test simulation"

  simulator = Simulator()
  simulator.run(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler = noise_scaler, snapshot_step = snapshot_step, name=name)
  simulator.analyse()
  assert_sim_analysis(simulator.simulation[0])

def test_fail_analyse_nonexistent_simulation():
  """Test failure to analyse a simulation that does not exists"""
  simulator = Simulator()
  with pytest.raises(ValueError):
    simulator.analyse()
  

def test_constant_force_without_force_or_potential():
  with pytest.raises(ValueError):
    # Should raise an error if the force and the potential are not provided
    simulator = Simulator(harmonic_potential=False)

def test_constant_force_average_x():
  """Tests that if the force f is constant then
  <x(t)> = f t + <x(0)>
  and
  Var(x(t)) = 2 t
  """
  def f(x,t):
    return 1.0
  def U(x,t):
    return -1.0*x
  tot_sims=100000
  simulator = Simulator(tot_sims=tot_sims, harmonic_potential=False, force=f, potential=U)
  simulator.run()
  sim=simulator.simulation[0]
  sim.build_averages('x')
  tol = 4.7/np.sqrt(tot_sims)
  for t in sim.results['times']:
    var = np.sqrt(2.0*t)
    assert sim.average_func['x'](t) == pytest.approx(1.0*t, abs=var*tol), f"average position not equal at time t={t}"
  
def test_constant_force_variance_x():
  """Tests the variance when the force f is constant:
  <x(t)> = f t + <x(0)>
  and
  Var(x(t)) = 2 t
  """
  def f(x,t):
    return 1.0
  def U(x,t):
    return -1.0*x
  tot_sims=100000
  simulator = Simulator(tot_sims=tot_sims, harmonic_potential=False, force=f, potential=U)
  simulator.run()
  sim=simulator.simulation[0]
  sim.build_variances('x')
  tol = 4.7/np.sqrt(tot_sims)
  for t in sim.results['times']:
    var = np.sqrt(3*4.0*t)
    assert sim.variance_func['x'](t) == pytest.approx(2.0*t, abs=var*tol), f"variance position not equal at time t={t}"
  