import numpy as np
from sklearn.model_selection import ParameterGrid
import warnings

with warnings.catch_warnings():
    # to hide the UserWarning occuring during the import because of geopandas not being available
    warnings.filterwarnings(action="ignore", category=UserWarning)
    from mgwr.gwr import Gaussian

with warnings.catch_warnings():
    # to hide the FutureWarning occuring during the import
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    import gpflow


from poi.project_config import BaselinesConfig, SVGPConfig
from poi.helpers import rmse_gaussian_lik_evaluation
from poi.baselines import (
    gwr,
    linear_kriging,
    rf_kriging,
    neural_network,
)
from poi.svgp import SlicedLinear, SlicedNN


# =====================================================================
# FILL IN START
# =====================================================================

from poi.chargingpoints import PROJECT_NAME
import poi.chargingpoints.data_pipeline

# General

# PROJECT_NAME
project_data_pipeline = poi.chargingpoints.data_pipeline.project_data_pipeline
evaluation_function = rmse_gaussian_lik_evaluation

# Baselines
baseline_project_data_pipeline_kwargs_options = [
    {
        "feature_engineering": False,
        "feature_type": None,
        "knot_points": 80,
    },
]
for count_feature_d_max in [
    0.5,
    1,
    1.25,
    1.5,
    1.75,
    2,
    2.25,
    2.5,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12.5,
    15,
]:
    baseline_project_data_pipeline_kwargs_options.append(
        {
            "feature_engineering": True,
            "feature_type": "both",
            "knot_points": 80,
            "count_feature_d_max": count_feature_d_max,
        }
    )
baseline_models = [
    {
        "model_base_name": "GWR",
        "model_run_function": gwr,
        "model_run_kwargs": {
            "family": Gaussian(),
            "kernel": "gaussian",
            "fixed": False,
            "constant": False,
            "bw": None,
        },
    },
    {
        "model_base_name": "Linear kriging",
        "model_run_function": linear_kriging,
        "model_run_kwargs": {},
    },
    {
        "model_base_name": "RF kriging",
        "model_run_function": rf_kriging,
        "model_run_kwargs": {},
    },
    {
        "model_base_name": "Neural network",
        "model_run_function": neural_network,
        "model_run_kwargs": {
            "verbose": False,
        },
    },
]

# SVGP

p = 5  # only chargingpoint covariates  # was X_train.shape[1] - 2

svgp_param_grid = ParameterGrid(
    param_grid=[
        {
            "project_data_pipeline_kwargs": [
                {
                    "feature_engineering": False,
                    "feature_type": None,
                    "knot_points": 0.5,
                },
            ],
            "mean_fct_dict": [
                {
                    "class": SlicedLinear,
                    "kwargs": {
                        "A": np.zeros((p, 1)).tolist(),
                        # list instead of numpy array so it is JSON serializable out of the box
                        "p": p,
                    },
                },
                {
                    "class": SlicedNN,
                    "kwargs": {"p": p},
                },
            ],
            "poi_kernel_type": [
                "Linear",
                "Gaussian",
            ],
            "svgp_likelihood_class_kwargs": [
                {"variance": 0.01},
                # {"variance": 0.1},
                # {"variance": 0.3},
                # {"variance": 0.5},
                # {"variance": 1.0},
            ]
        },
        # {
        #     "project_data_pipeline_kwargs": [
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 0.5,
        #         },
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 0.6,
        #         },
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 0.7,
        #         },
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 0.8,
        #         },
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 0.9,
        #         },
        #         {
        #             "feature_engineering": False,
        #             "feature_type": None,
        #             "knot_points": 1,
        #         },
        #     ],
        #     "mean_fct_dict": [
        #         {
        #             "class": SlicedLinear,
        #             "kwargs": {
        #                 "A": np.zeros((p, 1)).tolist(),
        #                 # list instead of numpy array so it is JSON serializable out of the box
        #                 "p": p,
        #             },
        #         },
        #     ],
        #     "poi_kernel_type": [
        #         "Linear",
        #     ],
        # },
    ]
)

svgp_likelihood_class = gpflow.likelihoods.Gaussian
# svgp_likelihood_class_kwargs = {"variance": 0.01}

svgp_D = 2  # Input dimensions for kernel
svgp_spatial_kernel_lengthscales = 50
svgp_iterations = 3000
svgp_logging_iteration = 20
svgp_monitor_iteration = 10
svgp_evaluation_iteration = 20

# =====================================================================
# FILL IN END
# =====================================================================

baselines_config = BaselinesConfig(
    PROJECT_NAME=PROJECT_NAME,
    project_data_pipeline=project_data_pipeline,
    evaluation_function=evaluation_function,
    project_data_pipeline_kwargs_options=baseline_project_data_pipeline_kwargs_options,
    models=baseline_models,
)

svgp_configs = []
for svgp_params in svgp_param_grid:
    svgp_configs.append(
        SVGPConfig(
            PROJECT_NAME=PROJECT_NAME,
            project_data_pipeline=project_data_pipeline,
            evaluation_function=evaluation_function,
            project_data_pipeline_kwargs=svgp_params["project_data_pipeline_kwargs"],
            mean_fct_class=svgp_params["mean_fct_dict"]["class"],
            mean_fct_kwargs=svgp_params["mean_fct_dict"]["kwargs"],
            likelihood_class=svgp_likelihood_class,
            # likelihood_class_kwargs=svgp_likelihood_class_kwargs,
            likelihood_class_kwargs=svgp_params["svgp_likelihood_class_kwargs"],
            poi_kernel_type=svgp_params["poi_kernel_type"],
            D=svgp_D,
            spatial_kernel_lengthscales=svgp_spatial_kernel_lengthscales,
            iterations=svgp_iterations,
            logging_iteration=svgp_logging_iteration,
            monitor_iteration=svgp_monitor_iteration,
            evaluation_iteration=svgp_evaluation_iteration,
        )
    )
