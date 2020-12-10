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


def update_policy_config(app_uptake, pol_compl, wfh, go_to_school_prob, asym_fac):
    policy_config['app_cov'] = app_uptake
    policy_config['compliance'] = pol_compl
    policy_config['wfh_prob'] = wfh
    policy_config['go_to_school_prob']  = go_to_school_prob
    contacts_config['asymp_factor']=asym_fac
    return policy_config, contacts_config


#     contacts_config['work_sar']=work_sar
def update_policy_config(app_uptake, pol_compl):
    policy_config['app_cov'] = app_uptake
    policy_config['compliance'] = pol_compl
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
        policy_config = update_policy_config(vals[0], vals[1])
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



def run_sensitivity(strigency, initial_samples, bo_samples, mc_samples, col):

    space = ParameterSpace([ContinuousParameter('app_cov', *v1),ContinuousParameter('compliance', *v2)]) # init param space for experimental design
    design = RandomDesign(space)
    x = design.get_samples(initial_samples) # get 5 random combinations of initialisations
    y = run_tti_sim(x) # obtain corresponding effective_rs

    # Fitting emulator to data, x: (app_cov, compliance), y: effective_r
    kern_eq = GPy.kern.RBF(input_dim=2, ARD = True) + GPy.kern.White(input_dim=2, variance=1)
    kern_bias = GPy.kern.Bias(input_dim=2)
    kern = kern_eq + kern_bias
    model_gpy = GPRegression(x,y, kern)
    model_gpy.kern.variance = 2**2
    # model_gpy.likelihood.variance.fix(1e-5)
    model_emukit = GPyModelWrapper(model_gpy)
    model_emukit.optimize() # optimise (max log-lik)

    # Initialise experimental design loop. Using integrated variance as acquisition
    # to "query" the input space aiming to reduce uncertainty over the func we're approx. i.e. effective_r
    num_of_loops = bo_samples
    integrated_variance = IntegratedVarianceReduction(space=space, model=model_emukit)
    ed = ExperimentalDesignLoop(space=space, model=model_emukit, acquisition = integrated_variance)
    ed.run_loop(run_tti_sim, num_of_loops)
    # Plot Main Effects
    num_mc = mc_samples
    senstivity = MonteCarloSensitivity(model = model_emukit, input_domain = space)
    main_effects_gp, total_effects_gp, p = senstivity.compute_effects(num_monte_carlo_points = num_mc)
    print("main_effects: ", main_effects_gp)
    print("total_effects: ", total_effects_gp)
    fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
    main_effects_gp_plot = {ivar: main_effects_gp[ivar][0] for ivar in main_effects_gp}

    d = {'App Uptake - Compliance Sensitivity Analysis':main_effects_gp_plot}

    pd.DataFrame(d).plot(kind='bar', ax=ax, color=col, alpha=0.65)
    plt.ylabel('% of explained output variance')
    out_name = "main-effects-{}.pdf".format(strigency)

    mlai.write_figure(filename=out_name, directory='./uq')

    # Plot Total Effects
    fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
    total_effects_gp_plot = {ivar: total_effects_gp[ivar][0] for ivar in total_effects_gp}
    d = {'App Uptake - Compliance Sensitivity Analysis':total_effects_gp_plot}
    pd.DataFrame(d).plot(kind='bar', ax=ax, color=col, alpha=0.65)
    ax.set_ylabel('% of explained output variance')
    out_name = "total-effects-{}.pdf".format(strigency)
    mlai.write_figure(filename=out_name, directory='./uq')

    app_cov_eval = np.linspace(0.05, 1, 100)
    app_com_eval = np.linspace(0.05, 1, 100)
    points=[]
    for i in app_cov_eval:
        for j in app_com_eval:
            points.append([i, j])
    points = np.asarray(points)

    X = points[:,0].reshape((100,100))
    Y = points[:,1].reshape((100,100))
    Z, _ = model_gpy.predict(points)
    Z = Z.reshape((100,100))

    print(X.shape)
    print(Y.shape)
    # print(Z[Z<1])


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7, color='red')
    # ax.scatter(x[:,0], x[:,1], y, color='red', alpha=0.99)
    ax.set_xlabel('App Uptake')
    ax.set_ylabel('Compliance')
    ax.set_zlabel('Effective R')
    out_name = "func-plot-{}.pdf".format(strigency)

    plt.savefig(out_name, format='pdf', bbox_inches='tight')


s_levels=['S1_test_based_TTI_test_contacts','S2_test_based_TTI_test_contacts','S3_test_based_TTI_test_contacts','S4_test_based_TTI_test_contacts','S5_test_based_TTI_test_contacts',]

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

    run_sensitivity(strigency, 20, 0, 10000, cols[cidx])
    cidx+=1
