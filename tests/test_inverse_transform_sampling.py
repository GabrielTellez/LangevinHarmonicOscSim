from ..langevin_harmonic_osc_simulator import Simulator, make_sampler
import numpy as np
import pytest
from scipy.special import gamma


def test_inverse_transform_sampler_single():
  """Test the inverse transform sampler
  """
  a=1.0
  def f(x):
    return np.exp(-a*np.abs(x))
  sampler, _ = make_sampler(f)
  samples = [sampler() for _ in range(1000000)]
  histo, xs = np.histogram(samples, density=True, range=(-4,4), bins=100)
  fs = f(xs[1:])
  fs = fs * a / 2.0 # normalization
  tol = 4.7*a/np.sqrt(len(samples))
  error = np.square((fs - histo)/fs).mean()
  assert error <= tol, f'error = {error} larger than tolerance {tol}'

def test_inverse_transform_sampler_multi():
  """Test the inverse transform sampler
  """
  a=3.5
  def f(x):
    return np.exp(-a*np.abs(x))
  _ , sampler = make_sampler(f)
  samples = sampler(1000000) 
  histo, xs = np.histogram(samples, density=True, range=(-2,2), bins=100)
  fs = f(xs[1:])
  fs = fs * a / 2.0 # normalization
  tol = 4.7*a/np.sqrt(len(samples))
  error = np.square((fs - histo)/fs).mean()
  assert error <= tol, f'error = {error} larger than tolerance {tol}'
  
def test_simulator_quartic_potential_initial_condition():
  """Tests the simulator with a quartic potential and 
  initial condition = equilibrium with that quartic potential
  """
  k0=1.0
  def U(x,t):
    k=k0+1.0*t
    return k*x**4/4.0

  simulator=Simulator(harmonic_potential=False, potential=U)
  simulator.run(tot_sims=100000)
  sim=simulator.simulation[0]
  xmax = np.power(4*2.0/k0, 1/4.0)
  sim.build_histogram('x', q_range=(-xmax, xmax))

  histo, xs = sim.histogram['x'][0]
  fs = np.exp(-U(xs[1:], 0))
  # normalization 
  Z=gamma(1/4)/np.sqrt(2*np.sqrt(k0))
  fs = fs / Z
  tol = 4.7/np.sqrt(sim.tot_sims)
  # tol = 1E-2
  error = np.square((fs - histo)/fs).mean()
  assert error <= tol, f'error = {error} larger than tolerance {tol}'

