import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numba as nb

def make_sampler(pdf,range=(-25,25), bins=10000001):
  """Generates a sampler of random samples distributed with pdf using the inverse
  transform sampling method
  Adapted from:
  https://towardsdatascience.com/random-sampling-using-scipy-and-numpy-part-i-f3ce8c78812e
  """

  def normalisation(x):
    return simps(pdf(x), x)
  xs = np.linspace(*range, bins)
  # define function to normalise our pdf to sum to 1 so it satisfies a distribution
  norm_constant = normalisation(xs)
  # create pdf
  my_pdfs = pdf(xs) / norm_constant
  # create cdf then ensure it is bounded at [0,1]
  my_cdf = np.cumsum(my_pdfs)
  my_cdf = my_cdf / my_cdf[-1]

  def sampler_single():
    rand = np.random.random_sample()
    return np.interp(rand, my_cdf, xs)
  def sampler_multi(size: int):
    rand = np.random.random_sample(size)
    return np.interp(rand, my_cdf, xs)

  return sampler_single, sampler_multi

def f(x):
  return np.exp(-np.abs(x))

samplerf_single, samplerf_multi = make_sampler(f)
jit_samplerf_single=nb.njit(samplerf_single)
jit_samplerf_multi=nb.njit(samplerf_multi)

for i in range(10):
  print(jit_samplerf_single())
print(jit_samplerf_multi(20))
