# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import Tuple, Callable

from ...core import ParameterSpace
from ...core.loop import UserFunctionWrapper
from ...core.interfaces.models import IModel
from itertools import combinations 


class ModelFreeMonteCarloSensitivity(object):
    """
    Class to do sensitivity analysis of a function. It computes Monte Carlo approximations to
    the Sobol indexes and the total variance components of each input variable of some objective
    of interest.
    """

    def __init__(self, objective: Callable, input_domain: ParameterSpace) -> None:
        """
        :param objective: python function in which the sensitivity analysis will be performed.
        :param input_domain: parameter space.
        """
        self.objective = UserFunctionWrapper(objective)
        self.input_domain = input_domain

    def _generate_samples(self, num_monte_carlo_points: int=int(1e5)) -> None:
        """
        Generates the two samples that are used to compute the main and total indices

        :param num_monte_carlo_points: number of samples to generate
        """
        self.main_sample = self.input_domain.sample_uniform(num_monte_carlo_points)
        self.fixing_sample = self.input_domain.sample_uniform(num_monte_carlo_points)

    def saltelli_estimators(self,
                            f_main_sample: np.ndarray, f_fixing_sample: np.ndarray,
                            f_new_fixing_sample: np.ndarray, num_monte_carlo_points: int,
                            total_mean: np.float64, total_variance: np.float64) -> Tuple:
        """
        Saltelli estimators of the total mean and variance
        """

        variable_main_variance = sum(f_main_sample * f_new_fixing_sample) / (num_monte_carlo_points - 1) - total_mean**2
        variable_total_variance = total_variance \
                                  - sum(f_fixing_sample * f_new_fixing_sample) / (num_monte_carlo_points - 1) \
                                  + total_mean**2
        return variable_main_variance, variable_total_variance

    def compute_statistics(self, sample: np.ndarray) -> Tuple:
        """
        Computes mean and variance of a sample

        :param sample: A sample to compute statistics for.
        :return: A tuple (mean, variance).
        """
        return sample.mean(), sample.var()

    def compute_effects(self,
                        main_sample: np.ndarray=None, fixing_sample: np.ndarray=None,
                        num_monte_carlo_points: int=int(1e5)) -> Tuple:
        """
        Computes the main and total effects using Monte Carlo and a give number of samples.
        - Main effects: contribution of x_j alone to the variance of f.
        - Total effects: contribution to all Sobol terms in which x_j is involved to the variance of f.

        The (unbiased) Monte Carlo estimates are computed using:

        "A. Saltelli, Making best use of model evaluations to compute sensitivity indices, Computer Physics Com.
        608 munications, 145 (2002), pp. 280-297"

        :param main_sample: original sample that is used in the Monte Carlo computations.
        :param fixing_sample: supplementary sample that is used in the Monte Carlo computations.
        :param num_monte_carlo_points: number of points used to compute the effects.

        :return: A tuple (main effects, total effects, total variance).
        """
        if main_sample is None or fixing_sample is None:
            self.num_monte_carlo_points = num_monte_carlo_points
            self._generate_samples(self.num_monte_carlo_points)
        else:
            self.main_sample = main_sample
            self.fixing_sample = fixing_sample
            self.num_monte_carlo_points = self.main_sample.shape[0]

        f_main_sample = self.objective.f(self.main_sample)
        f_fixing_sample = self.objective.f(self.fixing_sample)

        total_mean, total_variance = self.compute_statistics(f_main_sample)
        variable_names = self.input_domain.parameter_names

        main_effects = {}
        total_effects = {}
        var_index = 0

        for variable in variable_names:
            # --- All columns are the same but the one of interest that is replaced by the original sample
            self.new_fixing_sample = self.fixing_sample.copy()
            self.new_fixing_sample[:, var_index] = self.main_sample[:, var_index]

            # --- Evaluate the objective at the new fixing sample
            f_new_fixing_sample = self.objective.f(self.new_fixing_sample)

            # --- Compute the main and total variances
            variable_main_variance, variable_total_variance = \
                self.saltelli_estimators(
                    f_main_sample, f_fixing_sample,
                    f_new_fixing_sample, self.num_monte_carlo_points,
                    total_mean, total_variance)

            # --- Compute the effects
            main_effects[variable] = variable_main_variance / total_variance
            total_effects[variable] = variable_total_variance / total_variance

            var_index += 1

        return main_effects, total_effects, total_variance


class MonteCarloSensitivity(ModelFreeMonteCarloSensitivity):
    """
    Class to compute the sensitivity coefficients of given model. This class wraps the model and calls the mean
    predictions that are used to compute the sensitivity inputs using Monte Carlo.
    """

    def __init__(self, model: IModel, input_domain: ParameterSpace)-> None:
        """
        :param model: model wrapper with the interface IModel.
        :param input_domain: space class.
        """

        self.model = model
        self.model_objective = lambda x: self.model.predict(x)[0]

        super().__init__(self.model_objective, input_domain)



class MonteCarloSecondOrderSensitivity(ModelFreeMonteCarloSensitivity):
    """
    Class to compute the sensitivity coefficients of given model. This class wraps the model and calls the mean
    predictions that are used to compute the sensitivity inputs using Monte Carlo.
    """

    def __init__(self, model: IModel, input_domain: ParameterSpace)-> None:
        """
        :param model: model wrapper with the interface IModel.
        :param input_domain: space class.
        """

        self.model = model
        self.model_objective = lambda x: self.model.predict(x)[0]

        super().__init__(self.model_objective, input_domain)

    def _generate_samples(self, num_monte_carlo_points: int=int(1e5)) -> None:
        """
        Generates the two samples that are used to compute the main and total indices

        :param num_monte_carlo_points: number of samples to generate
        """
        self.main_sample = self.input_domain.sample_uniform(num_monte_carlo_points)
        self.main_sample2 = self.input_domain.sample_uniform(num_monte_carlo_points)
        self.fixing_sample = self.input_domain.sample_uniform(num_monte_carlo_points)

    def saltelli_estimators(self,
                            f_main_sample: np.ndarray, f_fixing_sample: np.ndarray,
                            f_new_fixing_sample: np.ndarray, num_monte_carlo_points: int,
                            total_mean: np.float64, total_variance: np.float64) -> Tuple:
        """
        Saltelli estimators of the total mean and variance
        """

        variable_main_variance = sum(f_main_sample * f_new_fixing_sample) / (num_monte_carlo_points - 1) - total_mean**2
        variable_total_variance = total_variance \
                                  - sum(f_fixing_sample * f_new_fixing_sample) / (num_monte_carlo_points - 1) \
                                  + total_mean**2
        return variable_main_variance, variable_total_variance

    def saltelli_estimators_second(self,
                            f_main_sample: np.ndarray, f_main_sample2: np.ndarray, f_fixing_sample: np.ndarray,
                            f_new_fixing_sample: np.ndarray, num_monte_carlo_points: int,
                            total_mean: np.float64, total_variance: np.float64,
                            first_order_sobol_of_pair0: np.float64, first_order_sobol_of_pair1: np.float64) -> Tuple:
        """
        Saltelli estimators of the total mean and variance
        """

        variable_second_variance = sum(f_main_sample * f_main_sample2 * f_new_fixing_sample) / (num_monte_carlo_points - 1) - first_order_sobol_of_pair0 - first_order_sobol_of_pair1 - total_mean**2

        return variable_second_variance

    def compute_effects(self,
                        main_sample: np.ndarray=None, main_sample2: np.ndarray=None, fixing_sample: np.ndarray=None,
                        num_monte_carlo_points: int=int(1e5)) -> Tuple:

        if main_sample is None or main_sample2 is None or fixing_sample is None:
            self.num_monte_carlo_points = num_monte_carlo_points
            self._generate_samples(self.num_monte_carlo_points)
        else:
            self.main_sample = main_sample
            self.main_sample2 = main_sample2
            self.fixing_sample = fixing_sample
            self.num_monte_carlo_points = self.main_sample.shape[0]

        f_main_sample = self.objective.f(self.main_sample)
        f_main_sample2 = self.objective.f(self.main_sample2)
        f_fixing_sample = self.objective.f(self.fixing_sample)

        total_mean, total_variance = self.compute_statistics(f_main_sample)
        variable_names = self.input_domain.parameter_names

        main_effects = {}
        secondary_effect = {}
        total_effects = {}
        var_index = 0

        for variable in variable_names:
            # --- All columns are the same but the one of interest that is replaced by the original sample
            self.new_fixing_sample = self.fixing_sample.copy()
            self.new_fixing_sample[:, var_index] = self.main_sample[:, var_index]

            # --- Evaluate the objective at the new fixing sample
            f_new_fixing_sample = self.objective.f(self.new_fixing_sample)

            # --- Compute the main and total variances
            variable_main_variance, variable_total_variance = \
                self.saltelli_estimators(
                    f_main_sample, f_fixing_sample,
                    f_new_fixing_sample, self.num_monte_carlo_points,
                    total_mean, total_variance)

            # --- Compute the effects
            main_effects[variable] = variable_main_variance / total_variance
            total_effects[variable] = variable_total_variance / total_variance
            var_index += 1

        combs = combinations(range(len(variable_names)), 2)
        for pair in combs:
            # --- All columns are the same but two are replaced by the original sample
            self.new_fixing_sample = self.fixing_sample.copy()
            self.new_fixing_sample[:, pair[0]] = self.main_sample[:, pair[0]]
            self.new_fixing_sample[:, pair[1]] = self.main_sample2[:, pair[1]]

            # --- Evaluate the objective at the new fixing sample
            f_new_fixing_sample = self.objective.f(self.new_fixing_sample)

            first_order_sobol_of_pair0 = main_effects[variable_names[pair[0]]]*total_variance
            first_order_sobol_of_pair1 = main_effects[variable_names[pair[1]]]*total_variance

            # --- Compute the second order Sobol indices
            variable_secondary_variance = \
                self.saltelli_estimators_second(
                    f_main_sample, f_main_sample2, f_fixing_sample,
                    f_new_fixing_sample, self.num_monte_carlo_points,
                    total_mean, total_variance,
                    first_order_sobol_of_pair0, first_order_sobol_of_pair1
                )

            # --- Compute the effects
            pair_names = variable_names[pair[0]] + '+' + variable_names[pair[1]]
            secondary_effect[pair_names] = variable_secondary_variance / total_variance

        # return f_main_sample, f_main_sample2, f_fixing_sample, f_new_fixing_sample
        return main_effects, secondary_effect, total_effects, total_variance

