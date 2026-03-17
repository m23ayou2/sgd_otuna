import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
import mlflow

def plot_observations(observations, dt=None, var_names=None, log_to_mlflow=False):

    if var_names is None:
        var_names = ['$X_0$', '$X_1$', '$X_2$']

    time = np.arange(observations.shape[0])

    fig = plt.figure(figsize=(15,4))

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.plot(time, observations[:,i])
        plt.title(f'Variable {var_names[i]}')
        plt.xlabel('Time step')
        plt.ylabel('Value')

    plt.tight_layout()

    if log_to_mlflow:
        mlflow.log_figure(fig, "observations.png")

    # plt.show()


def plot_estim_evolution(theta_estim_vec, theta, dt=None, var_names=None, log_to_mlflow=False, artifact_name="theta_evolution.png"):
    fig = plt.figure(figsize=(15,4))

    # Assuming epoch_hist, sigma_hist, rho_hist, beta_hist are defined
    plt.plot(theta_estim_vec[:,0], label=f'σ (vrai={theta[0]:.2f})', lw=1.5)
    plt.plot(theta_estim_vec[:,1], label=f'ρ (vrai={theta[1]:.2f})', lw=1.5)
    plt.plot(theta_estim_vec[:,2], label=f'β (vrai={theta[2]:.2f})', lw=1.5)

    # Horizontal lines for true values
    plt.axhline(y=theta[0], color='C0', ls=':', alpha=0.5)
    plt.axhline(y=theta[1], color='C1', ls=':', alpha=0.5)
    plt.axhline(y=theta[2], color='C2', ls=':', alpha=0.5)

    # Labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Valeur')
    plt.title('Evolution des paramètres physiques')

    # Legend and grid
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Layout and display
    plt.tight_layout()
    if log_to_mlflow:
        mlflow.log_figure(fig, artifact_name)

    # plt.show()


def plot_3d_trajectory(observations, var_names=None):
    var_names = ['$X_0$', '$X_1$', '$X_2$']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory as a line
    ax.plot(observations[:, 0], observations[:, 1], observations[:, 2],
            color='blue', linewidth=1)

    # Optionally mark the start point
    ax.scatter(*observations[0], color='green', s=50, label='Start')

    ax.set_xlabel(var_names[0])
    ax.set_ylabel(var_names[1])
    ax.set_zlabel(var_names[2])
    ax.set_title('3D Phase Space Trajectory')
    ax.legend()

    plt.tight_layout()
    plt.show()

def lorenz(t, x, sigma=10, beta=8/3, rho=28):
    return [sigma * (x[1] - x[0]), 
            x[0] * (rho - x[2]) - x[1], 
            x[0] * x[1] - beta * x[2]]


def solve_lorenz(t_0 = 0, t_f=10, dt=0.001, X_0 = [-8, 8, 27]): 
    t_train = np.arange(t_0, t_f, dt)
    sol = solve_ivp(lorenz, (t_train[0], t_train[-1]), X_0, t_eval=t_train)
    observations = sol['y'].transpose(1,0)

    return observations

def evaluate(theta_estim, theta):
    erreur_rel = np.abs((theta_estim - theta) / theta) * 100
    print(f"L'erreur relative sur σ, ρ, β est de : {erreur_rel[0]:.6f}%, {erreur_rel[1]:.6f}%, {erreur_rel[2]:.6f}%")
    return erreur_rel


def sgd_update(theta, grad, step_size):
    return theta - step_size * grad


def compute_snr(signal, noise_level=1.0, std=1.0):
    signal_power = np.var(signal, axis=0)
    actual_noise_variance = (noise_level * std) ** 2
    snr = 10 * np.log10(signal_power / actual_noise_variance)
    return snr
