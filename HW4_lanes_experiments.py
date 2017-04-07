"""
WIP 4 "lanes" experiments
"""

#----------------------------------------
# imports

import numpy
np = numpy
import numpy.random

from pylab import *

import time

from doina_class.algorithms import iterative_policy_evaluation, EVMC
from doina_class.environments import env_dict
from doina_class.policies import EpsilonGreedy
from doina_class.utilities import err_plot, sample


#----------------------------------------
# hparams

# these are all fixed in my experiments, except size and target_policy
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--P_terminate', type=float, default=.001) # *innate* probability of termination 
parser.add_argument('--gamma', type=int, default=.9)
parser.add_argument('--size', type=int, default=20)
parser.add_argument('--num_trials', type=int, default=30)
parser.add_argument('--points_per_trial', type=int, default=5000)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--target_policy', type=str, default='evaluation')
parser.add_argument('--time_scale', type=float, default=.0001)
args = parser.parse_args()
args_dict = args.__dict__
locals().update(args_dict)

if seed is None:
    seed = np.random.randint(2**32 - 1)


#----------------------------------------
# RUN 

# we run all combinations of environments and termination probabilities in this script
env_names = ['lanes1', 'lanes2', 'lanes3', 'lanes4']
envs = [env_dict[env](size, P_terminate=P_terminate) for env in env_names]
term_probs = [0,.5,1] # aka p^~_bot in the write-up

# results as RMS
RMSs = np.zeros((len(envs), len(term_probs), num_trials, points_per_trial))
figure()

for env_n, env in enumerate(envs):
    env.gamma = gamma
    mu = np.ones((env.nS, env.nA)) * 1./env.nA
    Q_pi = iterative_policy_evaluation(mu, env, return_Q=1)

    if target_policy == 'evaluation':
        pi = mu
    elif target_policy == 'control':
        pi = 'greedy'
    else:
        assert False, "not implemented!"

    for trial in range(num_trials):
        for nn, stochastic_termination in enumerate(term_probs):

            Q = np.zeros((env.nS, env.nA))
            C = np.zeros((env.nS, env.nA))
            t0 = time.time()
            for step in range(points_per_trial):
                while time.time() < t0 + time_scale * step:
                    C, Q, T = EVMC(env, pi=mu, mu=mu, num_episodes=1, C=C, Q=Q, stochastic_termination=stochastic_termination)
                RMSs[env_n, nn, trial, step] = (np.mean((Q - Q_pi)**2))**.5

        print trial 

    # TODO: save properly
    np.save("HW4_lanes__RMSs.npy", RMSs)


    if 1: # plotting 
        subplot(4,1, env_n+1)
        title(env_names[env_n])
        for nn, stochastic_termination in enumerate(term_probs):
            err_plot(RMSs[env_n, nn], label=r"$\tilde{p}_\bot=" + str(stochastic_termination) + r"\gamma$")
            ylabel('RMS of Q')
            if env_n == 3:
                legend()
                xlabel('seconds / 10^4)')
            show()





