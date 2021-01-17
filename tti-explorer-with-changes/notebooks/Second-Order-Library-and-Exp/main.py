from emukit_new.core import DiscreteParameter, ContinuousParameter, ParameterSpace
from emukit_new.core.initial_designs import RandomDesign
from emukit_new.model_wrappers import GPyModelWrapper
from emukit_new.experimental_design.acquisitions import IntegratedVarianceReduction, ModelVariance
from emukit_new.experimental_design.experimental_design_loop import ExperimentalDesignLoop
import os
import numpy as np
import pandas as pd
from tqdm.notebook import trange
from tti_explorer import config, utils
from tti_explorer.case import simulate_case, CaseFactors
from tti_explorer.contacts import EmpiricalContactsSimulator
from tti_explorer.strategies import TTIFlowModel, RETURN_KEYS
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
urllib.request.urlretrieve('https://raw.githubusercontent.com/lawrennd/talks/gh-pages/teaching_plots.py','teaching_plots.py')
urllib.request.urlretrieve('https://raw.githubusercontent.com/lawrennd/talks/gh-pages/mlai.py','mlai.py')
urllib.request.urlretrieve('https://raw.githubusercontent.com/lawrennd/talks/gh-pages/gp_tutorial.py','gp_tutorial.py')
from matplotlib import colors as mcolors
from matplotlib import cm
import numpy as np
import GPy
from GPy.models import GPRegression
import matplotlib.pyplot as pltb
import teaching_plots as plot
import mlai
import pandas as pd
from scipy.stats import gamma
from SALib.sample import saltelli
from SALib.analyze import sobol
from emukit_new.sensitivity.monte_carlo import ModelFreeMonteCarloSensitivity
from emukit_new.sensitivity.monte_carlo import MonteCarloSensitivity
from emukit_new.sensitivity.monte_carlo import MonteCarloSecondOrderSensitivity


def load_csv(pth):
    return np.loadtxt(pth, dtype=int, skiprows=1, delimiter=",")


name = 'S3_test_based_TTI'
case_config = config.get_case_config("delve")
contacts_config = config.get_contacts_config("delve")
policy_config = config.get_strategy_configs("delve", name)[name]
factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)

path_to_bbc_data = os.path.join("../../data", "bbc-pandemic")
over18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_o18.csv"))
under18 = load_csv(os.path.join(path_to_bbc_data, "contact_distributions_u18.csv"))
rng = np.random.RandomState(42)
simulate_contacts = EmpiricalContactsSimulator(over18, under18, rng)
tti_model = TTIFlowModel(rng, **strategy_config)

def update_case_config(p_day_noticed_symptoms_a):
    alpha = p_day_noticed_symptoms_a

    x = np.linspace(gamma.pdf(0.01, a=alpha),
                    gamma.pdf(0.99, a=alpha), 10)
    #apply softmax
    norm = np.exp(x)
    s = sum(norm)
    norm = np.round(norm/s, decimals=2)
    norm[-1] = 1- sum(norm[:-1])
    
    case_config['p_day_noticed_symptoms'] = norm
    return case_config

def update_contacts_config(asymp):
    contacts_config['asymp_factor'] = asymp
    return contacts_config

def update_policy_config(quarantine, latent, testing):
    policy_config['quarantine_length'] = int(round(quarantine))
    policy_config['latent_period'] = int(round(latent))
    policy_config['testing_delay'] = int(round(testing))
    return policy_config

def run_tti_sim(configs):
    to_return=[]
    for vals in configs:
#         contact_config = update_case_config(vals[0])
#         case_config = update_case_config(vals[1])
        policy_config = update_policy_config(vals[0],vals[1], vals[2])
        factor_config = utils.get_sub_dictionary(policy_config, config.DELVE_CASE_FACTOR_KEYS)
        strategy_config = utils.get_sub_dictionary(policy_config, config.DELVE_STRATEGY_FACTOR_KEYS)
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



def main():


    variable_domain3 = [10,11,12,13,14] 
    variable_domain4 = [3,4,5,6,7,8] 
    variable_domain5 = [1,2,3,4,5] 
    initial_samples = 10 # number of random runs to perform before starting experimental design 
    space = ParameterSpace([
    #     ContinuousParameter('asymp_factor', *variable_domain1),
    #                         ContinuousParameter('p_day_noticed_symptoms', *variable_domain2),
                            DiscreteParameter('quarantine_length', variable_domain3),
                            DiscreteParameter('latent_period', variable_domain4),
                            DiscreteParameter('testing_delay', variable_domain5)]) # init param space for experimental design
    design = RandomDesign(space)
    x = design.get_samples(initial_samples) # get 5 random combinations of initialisations
    y = run_tti_sim(x) # obtain corresponding effective_rs


    # Fitting emulator to data, x: (app_cov, compliance), y: effective_r
    Num_of_param = 3
    kern_eq = GPy.kern.RBF(input_dim=Num_of_param, ARD = True)
    kern_bias = GPy.kern.Bias(input_dim=Num_of_param)
    kern = kern_eq + kern_bias
    model_gpy = GPRegression(x,y, kern)
    model_gpy.kern.variance = 1**2
    model_gpy.likelihood.variance.fix(1e-5)
    model_emukit = GPyModelWrapper(model_gpy) 
    model_emukit.optimize() # optimise (max log-lik)
    # display(model_gpy)

    # Initialise experimental design loop. Using integrated variance as acquisition
    # to "query" the input space aiming to reduce uncertainty over the func we're approx. i.e. effective_r
    num_of_loops = 5
    integrated_variance = IntegratedVarianceReduction(space=space, model=model_emukit)
    ed = ExperimentalDesignLoop(space=space, model=model_emukit, acquisition = integrated_variance)
    ed.run_loop(run_tti_sim, num_of_loops)


    #Calculate Sobol indices with SALib
    problem = {
    'num_vars': 3,
    'names': ['quarantine_length','latent_period','testing_delay'],
    'bounds': [
                [10,14], 
                [3,8],
                [1,5]]
    }
    xt = saltelli.sample(problem, 11000)
    Y,_= model_gpy.predict(xt)
    Y = np.squeeze(Y)
    Si = sobol.analyze(problem, Y)

    # Calculate Sobol indices with Emukit_new
    senstivity = MonteCarloSecondOrderSensitivity(model = model_emukit, input_domain = space)
    main_effects_gp, secondary_effects_gp, total_effects_gp, _ = senstivity.compute_effects(num_monte_carlo_points = 1000)


    # Plot Main Effects
    fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
    main_effects_gp_plot = {ivar: main_effects_gp[ivar][0] for ivar in main_effects_gp}
    d = {'GP Monte Carlo':main_effects_gp_plot}
    pd.DataFrame(d).plot(kind='bar', ax=ax)
    plt.ylabel('% of explained output variance')
    ax.set_title("First Order Sobol Indices")
    out_name = "First-Order-Effects-{}.pdf".format(strigency)
    mlai.write_figure(filename=out_name, directory='./plots')


    # Plot Second Order Effects
    fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
    secondary_effects_gp_plot = {ivar: secondary_effects_gp[ivar] for ivar in secondary_effects_gp}
    for ivar in secondary_effects_gp:
        ax.bar(ivar,secondary_effects_gp[ivar], color = 'grey')
    plt.ylabel('% of explained output variance')   
    ax.set_title("Second Order Sobol Indices")
    out_name = "Second-Order-Effects-{}.pdf".format(strigency)
    mlai.write_figure(filename=out_name, directory='./plots')

    # Plot Total Effects
    fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
    total_effects_gp_plot = {ivar: total_effects_gp[ivar][0] for ivar in main_effects_gp}
    d = {'GP Monte Carlo':total_effects_gp_plot}
    pd.DataFrame(d).plot(kind='bar', ax=ax)
    plt.ylabel('% of explained output variance')
    ax.set_title("Total Effects")
    out_name = "Total-Effects-{}.pdf".format(strigency)
    mlai.write_figure(filename=out_name, directory='./plots')



if __name__ == "__main__":
    main()