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
from GPyOpt.methods import BayesianOptimization
from scipy.stats import gamma

def print_doc(func):
    print(func.__doc__)
rng = np.random.RandomState(0)

def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")
path_to_bbc_data = os.path.join("../../data", "bbc-pandemic")


def update_policy_config(go_to_school_prob, met_before_w, met_before_s, met_before_o, wfh_prob, asym_fac):
    policy_config['go_to_school_prob'] = go_to_school_prob
    policy_config['met_before_w'] = met_before_w
    policy_config['met_before_s'] = met_before_s
    policy_config['met_before_o']  = met_before_o
    policy_config['wfh_prob']=wfh_prob
    contacts_config['asymp_factor']=asym_fac


    return policy_config, contacts_config
    
    
def update_policy_config_for_bo(go_to_school_prob, met_before_w, met_before_s, met_before_o, wfh_prob, app_cov):
    policy_config['go_to_school_prob'] = go_to_school_prob
    policy_config['met_before_w'] = met_before_w
    policy_config['met_before_s'] = met_before_s
    policy_config['met_before_o']  = met_before_o
    policy_config['wfh_prob']=wfh_prob
    policy_config['app_cov']=app_cov

    policy_config['compliance']=0.5


    return policy_config
def update_inf_profile(alpha, beta):
    case_config['inf_profile']=he_infection_profile(10, gamma_params={"a": alpha, "scale": 1 / beta})
    return case_config
    
    
def he_infection_profile(period, gamma_params):
    """he_infection_profile

    Args:
        period (int): length of infectious period
        gamma_params (dict): shape and scale gamma parameters
        of infection profile

    Returns:
        infection_profile (np.array[float]): discretised and
        truncated gamma cdf, modelling the infection profile
        over period
    """
    inf_days = np.arange(period)
    mass = gamma.cdf(inf_days + 1, **gamma_params) - gamma.cdf(inf_days, **gamma_params)
    return mass / np.sum(mass)

# policy_config = update_policy_config(0.5, 0.05)
# Separating this because it is built from the ammended policy_config

#{'isolate_individual_on_symptoms': True, 'isolate_individual_on_positive': True, 'isolate_household_on_symptoms': True, 'isolate_household_on_positive': True, 'isolate_contacts_on_symptoms': False, 'isolate_contacts_on_positive': True, 'test_contacts_on_positive': True, 'do_symptom_testing': True, 'do_manual_tracing': True, 'do_app_tracing': True, 'fractional_infections': True, 'testing_delay': 2, 'app_trace_delay': 0, 'manual_trace_delay': 1, 'manual_home_trace_prob': 1.0, 'manual_work_trace_prob': 1.0, 'manual_othr_trace_prob': 1.0, 'met_before_w': 0.79, 'met_before_s': 0.9, 'met_before_o': 0.75, 'max_contacts': 20, 'quarantine_length': 14, 'latent_period': 3, 'app_cov': 0.35, 'compliance': 0.8, 'go_to_school_prob': 1.0, 'wfh_prob': 0.25}
#
#
#


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
        case_config = update_inf_profile(vals[0], vals[1])
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

v1 = [0.00, .99]
v2 = [0.05,.99]
v3 = [0.05, .99]
v4 = [0.05, .99]
v5 = [0.05, .99]
v6 = [0.05, .7]

s_levels=['S1_symptom_based_TTI','S2_symptom_based_TTI','S3_symptom_based_TTI','S4_symptom_based_TTI','S5_symptom_based_TTI',]

over18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_o18.csv"))
under18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_u18.csv"))
simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
cidx=0
cols = ['red', 'tomato', 'orange', 'deepskyblue', 'green']


def optimise_it(strig):
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias
    domain = [
    {'name': 'alpha', 'type': 'continuous', 'domain': (0.1,5)},
    {'name': 'beta', 'type': 'continuous', 'domain': (0.1,5)}]
    opt = BayesianOptimization(f=run_tti_sim, maximize=True, domain=domain,model_type='GP', initial_design_numdata = 20,
    kernel=kern, acquisition_type='EI')
    out_name="bo-infectious-max-{}.txt".format(strig)
    opt.run_optimization(max_iter=20, report_file=out_name)

for strigency in s_levels:
    case_config = config.get_case_config("delve")
    contacts_config = config.get_contacts_config("delve")
    policy_config = config.get_strategy_configs("delve", strigency)[strigency]
    factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
    strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
    rng = np.random.RandomState(42)

    tti_model = TTIFlowModel(rng, **strategy_config)

#    run_sensitivity(strigency, 100, 10, 10000, cols[cidx])
    optimise_it(strigency)
    cidx+=1
