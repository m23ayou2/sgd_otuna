import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from tqdm import tqdm
from utils import plot_3d_trajectory, plot_observations
from utils import solve_lorenz
from utils import evaluate
from utils import plot_estim_evolution
from utils import sgd_update
import optuna
import configparser
from utils import compute_snr
import argparse
from functools import partial
import os
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
from mlflow.tracking import MlflowClient



def f(X, theta):
    return np.array([theta[0]*X[1] - theta[0]*X[0],
                     theta[1]*X[0] - X[0]*X[2] - X[1],
                     X[0]*X[1] - theta[2]*X[2]])

def runge_kutta4(f, X_n, theta, h):
    k1 = f(X_n, theta)
    k2 = f(X_n + h*k1/2, theta)
    k3 = f(X_n + h*k2/2, theta)
    k4 = f(X_n + h*k3, theta)
    return X_n + h*(k1 + 2*k2 + 2*k3 + k4)/6


def compute_grad(f, methode, theta_estim, A, estim_obs, n, start = 5):
    
    if start == None:
        sum_term = np.dot(estim_obs[:n], A).sum(axis=0)
    else :
        sum_term = np.dot(estim_obs[n-start:n], A).sum(axis=0)   

    X_nm1 = estim_obs[n-1]
    X_n   = estim_obs[n] 
    
    error = methode(f, X_nm1, theta_estim, h) -  X_n
    # error = h * f(X_nm1, theta_estim) + X_nm1 -  X_n

    grad = sum_term * error
    return grad

def decay_lr(lr, epoch, decay = 1e-4): 
    return lr

def l_estimators(window_grads, trim_ratio=0.2):

    x = np.sort(window_grads)
    n = len(x)

    # 1. Mean
    mean_est = np.sum(window_grads,  axis=0)

    # 2. Median
    median_est = np.median(window_grads,  axis=0) * n

    # 3. Trimmed mean
    k = int(trim_ratio * n)
    trimmed = x[k:n-k] if n > 2*k else x
    trimmed_mean = np.mean(trimmed, axis=0)

    # 4. Winsorized mean
    winsor = x.copy()
    if n > 2*k:
        winsor[:k] = x[k]
        winsor[n-k:] = x[n-k-1]
    winsor_mean = np.mean(winsor,  axis=0)

    return {
        "mean": mean_est,
        "median": median_est,
        "trimmed_mean": trimmed_mean,
        "winsor_mean": winsor_mean
    }

def geometric_median(points, max_iter=50, tol=1e-5):

    y = np.mean(points, axis=0)

    for _ in range(max_iter):

        diff = points - y
        dist = np.linalg.norm(diff, axis=1)

        mask = dist > 1e-10
        if not np.any(mask):
            return y

        weights = 1 / dist[mask]

        y_new = np.sum(points[mask] * weights[:, None], axis=0) / np.sum(weights)

        if np.linalg.norm(y - y_new) < tol:
            break

        y = y_new

    return y


def huber_gradient_estimator(window_grads, c=1.5, max_iter=20):

    theta = np.median(window_grads, axis=0)

    for _ in range(max_iter):

        r = window_grads - theta

        weights = np.ones_like(r)
        mask = np.abs(r) > c
        weights[mask] = c / np.abs(r[mask])

        theta = np.sum(weights * window_grads, axis=0) / np.sum(weights, axis=0)

    return theta

def decay_grad(grads, current_grad_idx, clip_value=np.array([25,10,10]), window=40, decay=0.8):

    grad = grads[current_grad_idx]

    start = max(0, current_grad_idx - window + 1)
    window_grads = grads[start:current_grad_idx + 1]

    # --- normalize each gradient BEFORE smoothing ---
    norms = np.linalg.norm(window_grads, axis=1, keepdims=True) + 1e-8
    window_grads_norm = window_grads / norms

    # compute L-estimators
    grad_estim = l_estimators(window_grads_norm)["mean"]
    # grad_estim = geometric_median(window_grads_norm)
    # grad_estim = huber_gradient_estimator(window_grads_norm)

    return grad_estim




def main(window = 40, 
         decay = 0.8, 
         epochs = 2,
         std = 0.01,
         noise_level = 0.5,
         lr = np.array([1e-3, 1e-3, 1e-3]), 
         clip_value = np.array([25,10,10]),
         start = None): 
    
    theta_estim = np.array([-5.0, -3.0, 6.0])

    nbr_epochs = int(epochs * N)
    A = np.array([[-1, 1, 0],
                  [1 , 0, 0],
                  [0 , 0, -1]])

    estim_obs = observations.copy() + noise_level*np.random.normal(0, std, (N, 3))

    theta_estim_vec = np.zeros((nbr_epochs, 3))
    
    grads = np.zeros((nbr_epochs, 3))

    if start ==None: 
        beg_tqdm = 0
    else: 
        beg_tqdm = start
    
    theta_estim_vec[0:beg_tqdm] = theta_estim

    for epoch in tqdm(range(beg_tqdm, nbr_epochs)):
        n = epoch % N
        grad = compute_grad(f, runge_kutta4, theta_estim, A, estim_obs, n, start = start)
        grads[epoch, :] = grad
        grad = decay_grad(grads, epoch, clip_value, window, decay)
        theta_estim = sgd_update(theta_estim, grad, lr)
        lr = decay_lr(lr, epoch)
        theta_estim_vec[epoch] = theta_estim
    
    return theta_estim, theta_estim_vec, grads


def objective(trial, noise_level, std, client):

    window = trial.suggest_int('window', 10, 100)  
    decay = trial.suggest_float('decay', 0.1, 0.99) 
    nbr_epochs = trial.suggest_float('nbr_epochs', 0.5, 6) 

    lr = [
        trial.suggest_float('lr_0', 1e-4, 1e-3, log=True),
        trial.suggest_float('lr_1', 1e-4, 1e-3, log=True),
        trial.suggest_float('lr_2', 1e-4, 1e-3, log=True)
    ]
    start = trial.suggest_int("start", 1, 150)

#    clip_value = [
#         trial.suggest_int('clip_0', 1, 40),
#         trial.suggest_int('clip_1', 1, 30),
#         trial.suggest_int('clip_2', 1, 30)
#     ]
    with mlflow.start_run(nested=True) as run:
        # theta_estim, theta_estim_vec, grads = main(window, decay, nbr_epochs, noise_var, np.array(lr), np.array(clip_value))
        theta_estim, theta_estim_vec, grads = main(window = window, 
                                                 decay = decay, 
                                                epochs = nbr_epochs,
                                                std = std,
                                                noise_level = noise_level,
                                                lr = lr,
                                                start = start,
                                           #     clip_value = np.array([25,10,10])
                                                )

        evaluate(theta_estim, theta)

        # result = np.mean((theta_estim - theta)**2)
        # result = np.mean(((theta_estim - theta)/theta)**2)*100
        eqm = ((theta_estim - theta)/theta)**2
        result = (eqm[0]+eqm[1]+eqm[2])/3

        # result = np.var(grads)
        # result = np.mean(np.abs(grads))

        # # this dosn't work worked the better model does not have the min of this
        # result = np.mean(np.abs(np.diff(theta_estim_vec, axis=0))) 

        # result = np.mean(np.abs(grads[-50:]))
        #result = np.mean(np.linalg.norm(grads, axis=1))
        # print(np.mean(((theta_estim - theta)/theta)**2)*100)
        err = np.abs(theta_estim/theta)
        mlflow.log_metric("accuracy", np.mean(np.abs(theta_estim - theta)/theta)*100)
        mlflow.log_metric("err sigma", err[0]*100)
        mlflow.log_metric("err rho", err[1]*100)
        mlflow.log_metric("err beta", err[2]*100)

        mlflow.log_metric("sigma", theta_estim[0])
        mlflow.log_metric("rho", theta_estim[1])
        mlflow.log_metric("beta", theta_estim[2])

        mlflow.log_param("window", window)
        mlflow.log_param("nbr_epochs", nbr_epochs)
        mlflow.log_param("lr_0", lr[0])
        mlflow.log_param("lr_1", lr[1])
        mlflow.log_param("lr_2", lr[2])
        mlflow.log_param("start", start)
        mlflow.log_param("std", std)
        mlflow.log_param("noise_level", noise_level)

        plot_estim_evolution(theta_estim_vec, theta, log_to_mlflow=True, artifact_name= f"theta_evolution_trial_{trial.number}.png")
        # Add a description for this run
        mlflow.set_tag("mlflow.note.content",
                       f"Testing start = 5 and (eqm[0]+eqm[1]+eqm[2])/3")


    return result



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is the parsing for args")

    parser.add_argument("--std", type=float, help="The standard deviation to be used")
    parser.add_argument("--nbr_trials", type=int, help="The number of trials to be done")
    parser.add_argument("--noise_level", type=float, help="The noise level to be used")
    args = parser.parse_args()

    nbr_trails = args.nbr_trials
    std = args.std
    noise_level = args.noise_level
    

    config = configparser.ConfigParser()
    config_path = 'config.ini'
    files_read = config.read(config_path)


    X_0 = np.array([-8.0, 2.0, 27.0], dtype=np.float64)
    theta = np.array([10.0, 28.0, 8.0/3.0], dtype=np.float64)

    t_0 = config['Optimizer'].getfloat('t_0')
    t_f = config['Optimizer'].getfloat('t_1')
    h = config['Optimizer'].getfloat('h')
    N = int((t_f-t_0)/h)

    observations = np.zeros((N, 3))
    observations[0] = np.array(X_0)

    for n in range(N-1):
        observations[n+1] = runge_kutta4(f, observations[n], theta, h)

    observations_solver = solve_lorenz(t_0=t_0, t_f=t_f, dt=h, X_0=X_0)



    snr = compute_snr(observations, noise_level=noise_level, std=std)
    print(f"Noise stadard diviation is {std} and noise level {noise_level}")
    print(f"SNR (dB) for each dimension: {snr}")
    print(f"Average SNR (dB): {np.mean(snr):.2f}")
    
    client = MlflowClient()
    
    obj = lambda trial: objective(trial, noise_level=noise_level, std=std, client=client)

    MLFLOW_TRACKING_URI = "http://ec2-16-171-0-26.eu-north-1.compute.amazonaws.com:5000/"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"optuna_ex{np.mean(snr):.3}")

    mlflow_callback = MLflowCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        metric_name="accuracy"
    )



    study = optuna.create_study(direction='minimize')
    study.optimize(obj, n_trials = nbr_trails)


    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    theta_estim, theta_estim_vec, grads = main(window = trial.params["window"], 
                                               decay = trial.params["decay"], 
                                               epochs = trial.params["nbr_epochs"],
                                               std = std,
                                               noise_level = noise_level,
                                               lr = np.array([trial.params["lr_0"], trial.params["lr_1"], trial.params["lr_2"]]), 
                                               clip_value = np.array([trial.params["clip_0"], trial.params["clip_1"], trial.params["clip_2"]]))

    sigma, rho, beta = evaluate(theta_estim, theta)

