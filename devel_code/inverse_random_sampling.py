import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps

def inverse_cdf(pdf,range=(-25,25), bins=10000001):
  """Generates random samples distributed with pdf using the inverse
  transform sampling method"""

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
  generate the inverse cdf
  func_ppf = interp1d(my_cdf, xs, fill_value='extrapolate')
  return func_ppf


def f(x):
  return np.exp(-np.abs(x))

inv_cdf_f=inverse_cdf(f)

newxs=np.linspace(0,1,10000)
invcdfs=inv_cdf_f(newxs)

# This works and numba compiles for generating arrays
#def samplerf(size):
#  return np.interp(np.random.random_sample(size=size), newxs, invcdfs)

def samplerf_single():
  rand = np.random.random_sample()
  return np.interp(rand, newxs, invcdfs)

def samplerf_multi(size):
  rand = np.random.random_sample(size)
  return np.interp(rand, newxs, invcdfs)

import numba as nb

jit_samplerf_single=nb.njit(samplerf_single)
jit_samplerf_multi=nb.njit(samplerf_multi)

for i in range(10):
  print(jit_samplerf_single())
print(jit_samplerf_multi(20))



