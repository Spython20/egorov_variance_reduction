from config import *

def position_1_observable(q1, q2, p1, p2):
    return q1

def position_2_observable(q1, q2, p1, p2):
    return q2

def momentum_1_observable(q1, q2, p1, p2):
    return p1

def momentum_2_observable(q1, q2, p1, p2):
    return p2

def kinetic_energy_observable(q1, q2, p1, p2):
    return 0.5 * (p1**2 + p2**2)

def potential_energy_observable(q1, q2, p1, p2):
    return 0.5 * (q1**2 + q2**2) + lamb * (q1**2 * q2 - (1.0 / 3.0) * q2**3)

def grad_V(q1,q2): # take in sampled position arrays
    grad_q1_values = []
    grad_q2_values = []
    for i in range(len(q1)):
        grad_q1_values.append(q1[i] + 2* lamb*(q1[i] * q2[i]))
        grad_q2_values.append(q2[i]+ lamb * (q1[i]**2 - q2[i]**2))

    return np.array(grad_q1_values), np.array(grad_q2_values)


# takes in array of positions and momentums of each sampled point
def flow_vel_verlet(q10,q20,p10, p20, dt, n_steps):
    q1 = []
    q2 = []
    p1 = []
    p2 = []

    q1.append(q10)
    q2.append(q20)
    p1.append(p10)
    p2.append(p20)

    # velocity verlet algorithm
    for n in range(n_steps):
        grad_q1, grad_q2 = grad_V(q1[n], q2[n])
        p1_half = p1[n] - 0.5 * dt * grad_q1
        p2_half = p2[n] - 0.5 * dt * grad_q2
        q1.append(q1[n] + dt * p1_half)
        q2.append(q2[n] + dt * p2_half)

        grad_q1, grad_q2 = grad_V(q1[n+1], q2[n+1]) # recompute grad after new position computed

        p1.append(p1_half- 0.5 * dt * grad_q1)
        p2.append(p2_half- 0.5 * dt * grad_q2)
    return np.array(q1), np.array(q2), np.array(p1), np.array(p2)
'''
these arrays are of the form
[
  [sample1, sample2, sample3, ...]   <- time step 0
  [sample1, sample2, sample3, ...]   <- time step 1
  [sample1, sample2, sample3, ...]   <- time step 2
  ...
]
'''

def MC_sample(sample_size,rng):
    samples = rng.multivariate_normal(mean=z0, cov=covariance, size=sample_size) # returns matrix of shape (N,4) since the mean is (4,) which is 1dim array with 4 entries
    q1_samples = samples[:,0]
    q2_samples = samples[:,1]
    p1_samples = samples[:,2]
    p2_samples = samples[:,3]

    return q1_samples, q2_samples, p1_samples, p2_samples

def compute_observable(observable, q1, q2, p1, p2):
    observable_values = observable(q1, q2, p1, p2)
    return observable_values

def compute_expectation(observable, q0, p0, dt, n_steps):
    times = []
    expectation_values = []
    observable_values_stored = []
    q_stored = []
    p_stored = []
    for t, q_t, p_t in flow_vel_verlet(q0=q0, p0=p0, dt=dt, n_steps=n_steps): # loops for each time step
        observable_values = observable(q_t, p_t) # composes points with observable
        times.append(t)

        expectation_values.append(np.mean(observable_values)) # Monte Carlo approximation of the phase-space integral, which is just a mean since we have even-weight quadrature
        observable_values_stored.append(observable_values.copy())
        q_stored.append(q_t.copy())
        p_stored.append(p_t.copy())

    return (np.asarray(times), np.asarray(expectation_values), np.asarray(observable_values_stored), np.asarray(q_stored), np.asarray(p_stored))

# 2d phase space origin shifted for Hagedorn formula
def complex_z_1(q1, p1):
    return np.sqrt(d1) * (q1-q10) + 1j * (p1-p10) / np.sqrt(d1)

def complex_z_2(q2, p2):
    return np.sqrt(d2) * (q2-q20) + 1j * (p2-p20) / np.sqrt(d2)

# computes laguerre coefficients for polynomial
def laguerre_mn(m, n, zeta):
    """L_{m,n}(zeta) from LaTr Theorem 1 (Eq. 20)."""
    if m <= n:
        return 2**n * factorial(m) * np.conjugate(zeta)**(n-m) * eval_genlaguerre(m, n-m, 2*abs(zeta)**2)
    return 2**m * factorial(n) * (-zeta)**(m-n) * eval_genlaguerre(n, m-n, 2*abs(zeta)**2)

# orthogonal polynomial
def H(index, q1, q2, p1, p2):
    k, l = index
    zeta1 = complex_z_1(q1, p1) / np.sqrt(2*epsilon)
    zeta2 = complex_z_2(q2, p2) / np.sqrt(2*epsilon)
    norm = np.sqrt(2.0**(sum(k)+sum(l)) * factorial(k[0])*factorial(k[1])*factorial(l[0])*factorial(l[1]))
    value = ((-1)**sum(l) / norm * laguerre_mn(k[0], l[0], zeta1) * laguerre_mn(k[1], l[1], zeta2))
    return np.real_if_close(value)

def chorin_step_one(observable_flow_composed, q1_samples, q2_samples, p1_samples, p2_samples):
    c = []
    for index in basis_indices:
        c.append(np.mean(observable_flow_composed * np.conjugate(H(index, q1_samples, q2_samples, p1_samples, p2_samples)), axis=1))
    return c

def chorin_step_two(observable_flow_composed, c_alpha, q1_samples, q2_samples, p1_samples, p2_samples):
    correction = np.zeros_like(observable_flow_composed, dtype=complex)
    for i in range(len(c_alpha)):
        index = basis_indices[i]
        k, l = index
        if k == l == (0, 0):
            continue

        H_values = H(index, q1_samples, q2_samples, p1_samples, p2_samples)
        correction += np.asarray(c_alpha[i])[:, None] * H_values[None, :]
    corrected_values = observable_flow_composed - correction
    estimate = np.mean(corrected_values, axis=1)
    return np.real(estimate) if np.isrealobj(observable_flow_composed) else estimate
