from config import *
from functions import *

def run_main():
    egorov_q1_trials = []
    egorov_q2_trials = []
    egorov_p1_trials = []
    egorov_p2_trials = []

    chorin_q1_trials = []
    chorin_q2_trials = []
    chorin_p1_trials = []
    chorin_p2_trials = []

    for trial in range(number_of_trials):
        trial_num = trial + 1
        print("begin trial #" + str(trial_num))

        # initial monte carlo sampling
        initial_rng = np.random.default_rng(1000 + 2 * trial)
        q1_initial, q2_initial, p1_initial, p2_initial = MC_sample(N, initial_rng)

        print("computing initial sampling coefficients #" + str(trial_num))
        sample_1_q1, sample_1_q2, sample_1_p1, sample_1_p2 = flow_vel_verlet(q1_initial, q2_initial, p1_initial, p2_initial,dt, n_steps)
        q1_composed = compute_observable(position_1_observable, sample_1_q1, sample_1_q2, sample_1_p1, sample_1_p2)
        q2_composed = compute_observable(position_2_observable, sample_1_q1, sample_1_q2, sample_1_p1, sample_1_p2)
        p1_composed = compute_observable(momentum_1_observable, sample_1_q1, sample_1_q2, sample_1_p1, sample_1_p2)
        p2_composed = compute_observable(momentum_2_observable, sample_1_q1, sample_1_q2, sample_1_p1, sample_1_p2)
        print("computing initial chorin coefficients #" + str(trial_num))
        q1_coefficients = chorin_step_one(q1_composed, q1_initial, q2_initial, p1_initial, p2_initial)
        q2_coefficients = chorin_step_one(q2_composed, q1_initial, q2_initial, p1_initial, p2_initial)
        p1_coefficients = chorin_step_one(p1_composed, q1_initial, q2_initial, p1_initial, p2_initial)
        p2_coefficients = chorin_step_one(p2_composed, q1_initial, q2_initial, p1_initial, p2_initial)

        # second independent sampling
        second_rng = np.random.default_rng(1001 + 2 * trial)
        q1_second, q2_second, p1_second, p2_second = MC_sample(N, second_rng)

        print("computing second sampling flow #" + str(trial_num))
        sample_2_q1, sample_2_q2, sample_2_p1, sample_2_p2 = flow_vel_verlet(q1_second, q2_second, p1_second, p2_second,dt, n_steps)
        q1_second_composed = compute_observable(position_1_observable, sample_2_q1, sample_2_q2, sample_2_p1, sample_2_p2)
        q2_second_composed = compute_observable(position_2_observable, sample_2_q1, sample_2_q2, sample_2_p1, sample_2_p2)
        p1_second_composed = compute_observable(momentum_1_observable, sample_2_q1, sample_2_q2, sample_2_p1, sample_2_p2)
        p2_second_composed = compute_observable(momentum_2_observable, sample_2_q1, sample_2_q2, sample_2_p1, sample_2_p2)
        egorov_q1 = np.mean(q1_second_composed, axis=1) # take mean of each row, see vel_verlet.py for array shapes
        egorov_q2 = np.mean(q2_second_composed, axis=1)
        egorov_p1 = np.mean(p1_second_composed, axis=1)
        egorov_p2 = np.mean(p2_second_composed, axis=1)

        print("computing corrected chorin estimate #" + str(trial_num))
        chorin_q1 = chorin_step_two(q1_second_composed, q1_coefficients, q1_second, q2_second, p1_second, p2_second)
        chorin_q2  = chorin_step_two(q2_second_composed, q2_coefficients, q1_second, q2_second, p1_second, p2_second)
        chorin_p1  = chorin_step_two(p1_second_composed, p1_coefficients, q1_second, q2_second, p1_second, p2_second)
        chorin_p2  = chorin_step_two(p2_second_composed, p2_coefficients, q1_second, q2_second, p1_second, p2_second)
        egorov_q1_trials.append(egorov_q1)
        egorov_q2_trials.append(egorov_q2)
        egorov_p1_trials.append(egorov_p1)
        egorov_p2_trials.append(egorov_p2)

        chorin_q1_trials.append(chorin_q1)
        chorin_q2_trials.append(chorin_q2)
        chorin_p1_trials.append(chorin_p1)
        chorin_p2_trials.append(chorin_p2)

    # convert trial lists to arrays
    egorov_q1_trials = np.array(egorov_q1_trials)
    egorov_q2_trials = np.array(egorov_q2_trials)
    egorov_p1_trials = np.array(egorov_p1_trials)
    egorov_p2_trials = np.array(egorov_p2_trials)

    chorin_q1_trials = np.array(chorin_q1_trials)
    chorin_q2_trials = np.array(chorin_q2_trials)
    chorin_p1_trials = np.array(chorin_p1_trials)
    chorin_p2_trials = np.array(chorin_p2_trials)

    # timestep corresponding to target_time
    target_index = int(round(target_time / dt))

    # values from every trial at target_time
    egorov_q1_target = egorov_q1_trials[:, target_index]
    egorov_q2_target = egorov_q2_trials[:, target_index]
    egorov_p1_target = egorov_p1_trials[:, target_index]
    egorov_p2_target = egorov_p2_trials[:, target_index]

    chorin_q1_target = chorin_q1_trials[:, target_index]
    chorin_q2_target = chorin_q2_trials[:, target_index]
    chorin_p1_target = chorin_p1_trials[:, target_index]
    chorin_p2_target = chorin_p2_trials[:, target_index]
    
    # put the four phase-space coordinates together
    egorov_trials = np.column_stack((egorov_q1_target,egorov_q2_target,egorov_p1_target,egorov_p2_target))
    chorin_trials = np.column_stack((chorin_q1_target,chorin_q2_target,chorin_p1_target,chorin_p2_target))

    # covariance
    egorov_covariance = np.cov(egorov_trials, rowvar=False, ddof=1)
    chorin_covariance = np.cov(chorin_trials, rowvar=False, ddof=1)
    # extract the varaince from the covariance matrix; the diagonal is the variance
    egorov_coordinate_variances = np.diag(egorov_covariance)
    chorin_coordinate_variances = np.diag(chorin_covariance)

    coordinate_names = ["q1", "q2", "p1", "p2"]
    for i, name in enumerate(coordinate_names):
        print(name)
        print("Egorov variance:", egorov_coordinate_variances[i])
        print("Chorin variance:", chorin_coordinate_variances[i])
        print("Variance ratio:",chorin_coordinate_variances[i] / egorov_coordinate_variances[i])
    
    # q1-p1 phase space plot
    plt.figure(figsize=(8, 6))
    plt.scatter(egorov_q1_target,egorov_p1_target,color="blue",marker="o",s=60,alpha=0.75,label="Base Egorov trials")
    plt.scatter(chorin_q1_target,chorin_p1_target,color="red",marker="s",s=60,alpha=0.75,label="Chorin trials")
    plt.xlabel(r"$\langle q_1 \rangle$")
    plt.ylabel(r"$\langle p_1 \rangle$")
    plt.title(f"Phase Space $(q_1,p_1)$ at t={target_time}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

    # q2-p2 phase space plot
    plt.figure(figsize=(8, 6))

    plt.scatter(egorov_q2_target, egorov_p2_target, color="blue", marker="o", s=60, alpha=0.75, label="Base Egorov trials")
    plt.scatter(chorin_q2_target, chorin_p2_target, color="red", marker="s", s=60, alpha=0.75, label="Chorin trials")
    plt.xlabel(r"$\langle q_2 \rangle$")
    plt.ylabel(r"$\langle p_2 \rangle$")
    plt.title(f"Phase Space $(q_2,p_2)$ at t={target_time}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


run_main()
