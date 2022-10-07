from ..langevin_harmonic_osc_simulator import make_simulator
import numpy as np
from pytest import approx
import pytest

def test_make_simulator():
  """
  tests if the simulator with default parameters is created correctly and can be called.
  """

  simulator = make_simulator()
  assert callable(simulator)

def test_run_simulator():
  """
  tests the simulator runs and returns correct data shape.
  """
  def k1(t):
    return 1.0
  def center1(t):
    return 0.0
  tot_sims=1000
  tot_steps=10000
  snapshot_step=100
  simulator = make_simulator(tot_sims=tot_sims, dt=0.00001, tot_steps=tot_steps, snapshot_step=snapshot_step, k=k1, center=center1)
  times, x, power, work, heat, delta_U, energy = simulator()
  shape = (tot_sims, int(tot_steps/snapshot_step) + 1)
  assert len(times) == int(tot_steps/snapshot_step) + 1
  assert np.shape(x) == shape
  assert np.shape(power) == shape
  assert np.shape(work) == shape
  assert np.shape(heat) == shape
  assert np.shape(delta_U) == shape
  assert np.shape(energy) == shape

  # Now test the run with different parameters from the default ones
  tot_sims2=500
  tot_steps2 = 1000
  snapshot_step2 = 230
  times, x, power, work, heat, delta_U, energy = simulator(tot_sims=tot_sims2, tot_steps=tot_steps2, snapshot_step=snapshot_step2)
  shape = (tot_sims2, int(tot_steps2/snapshot_step2) + 1)
  assert len(times) == int(tot_steps2/snapshot_step2) + 1
  assert np.shape(x) == shape
  assert np.shape(power) == shape
  assert np.shape(work) == shape
  assert np.shape(heat) == shape
  assert np.shape(delta_U) == shape
  assert np.shape(energy) == shape

def test_no_evolution():
  """Tests for no change in probability distribution when the potential
  does not change"""
  def k1(t):
    return 1.0
  def center1(t):
    return 0.0
  def Peq(x):
    """Equilibrium distribution"""
    return np.exp(-0.5*k1(0.0)*(x-center1(0.0))**2)/np.sqrt(2*np.pi/k1(0.0))

  tot_sims=100000
  tot_steps=1000
  snapshot_step=100
  simulator = make_simulator(tot_sims=tot_sims, dt=0.00001, tot_steps=tot_steps, snapshot_step=snapshot_step, k=k1, center=center1)
  times, x, power, work, heat, delta_U, energy = simulator()
  x_range=[-3.0, 3.0]
  bins = 300
  histos=[np.histogram(x[:,ti], density=True, range=x_range, bins=bins) for ti in range(0,len(times))]
  size_x = len(histos[0][1])
  xx=np.linspace(*x_range, size_x-1)
  Peq_array=Peq(xx)
  tol = 1.0/np.sqrt(tot_sims)
  for time_index in range(0, len(times)):
    err = np.square(histos[time_index][0]-Peq_array).mean()
    assert err < 0.1/np.sqrt(tot_sims) , f"P(x) should be the equilibrium distribution at all times. Error on time_index={time_index}"
    assert work[:,time_index] == approx(0.0), f"No work if the potential does not change. Error on time_index={time_index}"
    assert power[:,time_index] == approx(0.0), f"No power if the potential does not change. Error on time_index={time_index}"
    if time_index > 0:
      assert not heat[:,time_index] == approx(0.0), f"heat is not zero for each realization. Error on time_index={time_index}"
    assert np.average(heat[:,time_index]) == approx(0.0, rel=tol, abs=tol), f"heat should be zero on average. Error on time_index={time_index}"
    assert np.average(delta_U[:,time_index]) == approx(0.0, rel=tol, abs=tol), f"energy should not change on average. Error on time_index={time_index}"
    # test energy with a confidence interval of 1-1E-6, thus the factor 4.75 
    assert np.average(energy[:, time_index]) == approx (0.5, rel=4.75*tol), f"on average energy should be (1/2) k_B T. Error on time_index={time_index}"

def test_energy_conservation_fixed_potential():
  # test with fixed potential
  def k1(t):
    return 1.0
  def center1(t):
    return 0.0
  tot_sims=10
  tot_steps=10000
  snapshot_step=100
  simulator = make_simulator(tot_sims=tot_sims, dt=0.00001, tot_steps=tot_steps, snapshot_step=snapshot_step, k=k1, center=center1)
  times, x, power, work, heat, delta_U, energy = simulator()
  for time_index in range(0, len(times)):
    assert delta_U[:, time_index] - (work[:, time_index] + heat[:, time_index]) == approx(0.0)
    assert delta_U[:, time_index] - (energy[:, time_index] - energy[:, 0]) == approx(0.0)

def test_energy_conservation_variable_potential():
  # test with moving center and k linear
  def k2(t):
    return 1.0+0.1*t
  def center2(t):
    return 0.2*t
  tot_sims=10
  tot_steps=10000
  snapshot_step=100
  simulator = make_simulator(tot_sims=tot_sims, dt=0.00001, tot_steps=tot_steps, snapshot_step=snapshot_step, k=k2, center=center2)
  times, x, power, work, heat, delta_U, energy = simulator()
  for time_index in range(0, len(times)):
    assert delta_U[:, time_index] - (work[:, time_index] + heat[:, time_index]) == approx(0.0)
    assert delta_U[:, time_index] - (energy[:, time_index] - energy[:, 0]) == approx(0.0)

def test_general_force():
  """Tests the simulator for a non harmonic force
  """
  def f(x,t):
    return 1.0
  with pytest.raises(ValueError):
    # This will fail if the potential is not provided
    simulator = make_simulator(harmonic_potential=False, force=f)
    times, x, power, work, heat, delta_U, energy = simulator()

def test_general_potential():
  """Tests the simulator for a non harmonic potential
  """
  def U(x,t):
    return -1.0*x
  def init_cond():
    return 0.0
  simulator = make_simulator(harmonic_potential=False, potential=U, initial_distribution=init_cond)
  times, x, power, work, heat, delta_U, energy = simulator()






