import os
import numpy as np
import pandas as pd
from   tqdm.notebook import trange
from   tti_explorer import config, utils
from   tti_explorer.case import simulate_case, CaseFactors
from   tti_explorer.contacts import EmpiricalContactsSimulator
from   tti_explorer.strategies import TTIFlowModel, RETURN_KEYS
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from   matplotlib import colors as mcolors
from   matplotlib import cm
import numpy as np
import GPy
from   emukit.core import ContinuousParameter, ParameterSpace
from   emukit.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity
from   emukit.core.initial_designs import RandomDesign
from   GPy.models import GPRegression
from   emukit.model_wrappers import GPyModelWrapper
from   emukit.sensitivity.monte_carlo import MonteCarloSensitivity
from   emukit.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from   emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
import matplotlib.pyplot as plt
import teaching_plots as plot
import mlai
import pandas as pd
from   tti_explorer.strategies import TTIFlowModel
def print_doc(func):
    print(func.__doc__)
rng = np.random.RandomState(0)

def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
path_to_bbc_data = os.path.join("../../data", "bbc-pandemic")


def update_policy_config(app_uptake, go_to_school_prob, wfh):
    policy_config['app_cov'] = app_uptake
    policy_config['compliance'] = 0.5
    policy_config['wfh_prob'] = wfh
    policy_config['go_to_school_prob']  = go_to_school_prob
    
    return policy_config


# policy_config = update_policy_config(0.5, 0.05)
# Separating this because it is built from the ammended policy_config







"""
Runs TTI simulator as many times as the different input initialisations.
The main reason we need this is to put in to EmuKit for the Experimental
Design Loop.

Args:
    pol_configs (list): Inputs are [app_cov, compliance]

Returns:
    effective_rs (np.expand_dims(np.array(to_return), 1)): For every simulation run, return
    the effective r, as was plotted form Bryn and Andrei previously.

"""
#simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)


def run_tti_sim(pol_configs):
    to_return=[]
    
    for vals in pol_configs:
        policy_config = update_policy_config(vals[0], vals[1], vals[2])
        factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
        strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
        rng = np.random.RandomState(42)
        tti_model = TTIFlowModel(rng, **strategy_config)
        n_cases = 10000
        outputs = list()
        temp =[]
        for _ in trange(n_cases):
            case = simulate_case(rng, **case_config)
            case_factors = CaseFactors.simulate_from(rng, case, **factor_config)
            contacts = simulate_contacts(case, **contacts_config)
            res = tti_model(case, contacts, case_factors)
            outputs.append(res)
            if res["Effective R"] >= 0:
                temp.append(res["Effective R"]) # Get effective R. Only non-nan vals are kept
        to_return.append(np.mean(temp))
    return np.expand_dims(np.array(to_return), 1)

v1 = [0.05, .99]
v2 = [0.2,.95]
v3 = [0.05, .99]
v4 = [0.05, .99]
v5 = [0.05, .99]
# v6 = [0.05, .4]
from GPyOpt.methods import BayesianOptimization


def run_bo(strigency):

    # Fitting emulator to data, x: (app_cov, compliance), y: effective_r
    kern_eq = GPy.kern.RBF(input_dim=3, ARD = True)
    kern_bias = GPy.kern.Bias(input_dim=3)
    kern = kern_eq + kern_bias
#    model_gpy = GPRegression(x,y, kern)
#    model_gpy.kern.variance = 1**2
#    # model_gpy.likelihood.variance.fix(1e-5)
#    model_emukit = GPyModelWrapper(model_gpy)
#    model_emukit.optimize() # optimise (max log-lik)
    domain = [{'name': 'app_cov', 'type': 'continuous', 'domain': (0,1)},{'name': 'go_to_school_prob', 'type': 'continuous', 'domain': (0,1)},{'name': 'wfh_prob', 'type': 'continuous', 'domain': (0,.8)}]
    opt = BayesianOptimization(f=run_tti_sim, domain=domain,model_type='GP', initial_design_numdata = 10,
    kernel=kern, acquisition_type='EI')
    opt.run_optimization(max_iter=6, report_file='bo.txt')


s_levels=['S1_symptom_based_TTI','S2_symptom_based_TTI','S3_symptom_based_TTI','S4_symptom_based_TTI','S5_symptom_based_TTI',]

over18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_o18.csv"))
under18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_u18.csv"))
simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
cidx=0
cols = ['red', 'tomato', 'orange', 'deepskyblue', 'green']
for strigency in s_levels:
    case_config = config.get_case_config("delve")
    contacts_config = config.get_contacts_config("delve")
    policy_config = config.get_strategy_configs("delve", strigency)[strigency]
    factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
    strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
    rng = np.random.RandomState(42)

    tti_model = TTIFlowModel(rng, **strategy_config)

    run_bo(strigency)
    cidx+=1
