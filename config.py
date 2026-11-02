import numpy as np
import scipy as s
from math import factorial
import matplotlib.pyplot as plt
from scipy.special import comb, eval_genlaguerre

#config
epsilon = 0.1
d = 2 # dimension of system

# matrix for initial gaussian wavepacket
d1 = 1.0
d2 = 1.0
D1 = np.diag([d1,d2])
D2 = s.linalg.block_diag(D1, np.linalg.inv(D1))

# center of gaussian
q10 = 0.1
q20 = 0.0
p10 = 0.3
p20 = 0.0

lamb = 1 # lambda for Henon-Heiles system

N = 10000 # number of samples
dt = 0.01
final_time = 1.0
target_time = 1.0
n_steps = int(round(final_time / dt))

basis_indices = [
    ((1, 0), (1, 0)),
    ((0, 1), (0, 1)),
    ((2, 0), (2, 0)),
    ((1, 1), (1, 1)),
    ((0, 2), (0, 2)),
    ((1, 0), (0, 0)),
    ((0, 0), (1, 0)),
    ((0, 1), (0, 0)),
    ((0, 0), (0, 1)),
    ((1, 0), (0, 1)),
    ((0, 1), (1, 0)),
]

rng = np.random.default_rng(seed=42) # MC sampling seed
z0 = np.array([q10,q20,p10,p20]) # gaussian wavepacket centre
covariance = (epsilon / 2.0) * np.linalg.inv(D2) # Covariance corresponding exactly to the Gaussian Wigner density
number_of_trials = 20
