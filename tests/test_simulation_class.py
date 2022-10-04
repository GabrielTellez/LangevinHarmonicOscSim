from ..langevin_harmonic_osc_simulator import Simulation
import numpy as np
import pytest

@pytest.fixture
def dummy_sim():
  """Builds a simulation with dummy paramaters and data
  """
  def k(t):
    """t |--> 1.0 """
    return 1.0
  def center(t):
    """t |--> 0.0"""
    return 0.0
  tot_sims = 10
  dt = 0.01
  tot_steps = 100
  snapshot_step = 10
  noise_scaler = 1.0
  tot_snapshots = int(tot_steps/snapshot_step)+1
  x = np.random.random_sample((tot_sims, tot_snapshots))
  work = np.random.random_sample((tot_sims, tot_snapshots))
  power = np.random.random_sample((tot_sims, tot_snapshots))
  heat = np.random.random_sample((tot_sims, tot_snapshots))
  delta_U = np.random.random_sample((tot_sims, tot_snapshots))
  energy = np.random.random_sample((tot_sims, tot_snapshots))
  times=np.arange(0, (1+tot_steps)*dt, dt*snapshot_step)
  results = (times, x, power, work, heat, delta_U, energy)
  sim = Simulation(tot_sims = tot_sims, dt = dt, tot_steps = tot_steps, noise_scaler=noise_scaler, snapshot_step=snapshot_step, k=k, center=center, results=results)
  return (
    tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
    k, center, results,
    sim
  )

def test_simulation_init(dummy_sim):
  """Tests correct creation of a simulation class and store of parameters
  """
  (
    tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
    k, center, results,
    sim
  ) = dummy_sim
  assert sim.tot_sims == tot_sims
  assert sim.dt ==  dt
  assert sim.tot_steps == tot_steps 
  assert sim.noise_scaler == noise_scaler
  assert sim.snapshot_step == snapshot_step
  assert sim.k == k 
  assert sim.center == center 

def test_simulation_init_store_results(dummy_sim):
  """Tests correct creation of a simulation class and store of the results
  """
  (
    tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
    k, center, results,
    sim
  ) = dummy_sim
  (times, x, power, work, heat, delta_U, energy) = results
  labels = ["times", "x", "power", "work", "heat", "delta_U", "energy"]
  assert sim.result_labels == labels
  assert (sim.results['times'] - times).all() == 0
  assert (sim.results['x'] - x).all() == 0
  assert (sim.results['power'] - power).all() == 0
  assert (sim.results['work'] - work).all() == 0
  assert (sim.results['heat'] - heat).all() == 0
  assert (sim.results['delta_U'] - delta_U).all() == 0
  assert (sim.results['energy'] - energy).all() == 0



def test_simulation_results_shape(dummy_sim):
  """Tests if the results have the correct shape

  Args:
      dummy_sim (tuple): dummy simulation (
    tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
    k, center, results,
    Simulation
  )
  """
  (
    tot_sims, dt, tot_steps, noise_scaler, snapshot_step,
    k, center, results,
    sim
  ) = dummy_sim
  sim_res = sim.results
  tot_snapshots = int(tot_steps/snapshot_step)+1
  shape = (tot_sims, tot_snapshots)
  assert len(sim_res['times']) == tot_snapshots
  assert np.shape(sim_res['x']) == shape
  assert np.shape(sim_res['power']) == shape
  assert np.shape(sim_res['work']) == shape
  assert np.shape(sim_res['heat']) == shape
  assert np.shape(sim_res['delta_U']) == shape
  assert np.shape(sim_res['energy']) == shape