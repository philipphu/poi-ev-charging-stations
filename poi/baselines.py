"""
Defines functions to test the predictive power of several baseline models.
"""
from contextlib import redirect_stderr
import json
from loguru import logger
import os
import time
import warnings

with warnings.catch_warnings():
    # to hide the UserWarning occuring during the import because of geopandas not being available
    warnings.filterwarnings(action="ignore", category=UserWarning)
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from sklearn.preprocessing import StandardScaler, MinMaxScaler

with warnings.catch_warnings():
    # to hide the FutureWarning occuring during the import
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    with redirect_stderr(open(os.devnull, "w")):
        # to hide the "Using TensorFlow backend." message occuring during the import
        import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.backend as K

import poi
from poi.helpers import (
    get_information_dict,
    save_df_as_latex,
    get_model_name,
    test_json_serializable,
)


def gwr(
    X_train,
    X_test,
    y_train,
    family,
    kernel,
    fixed=False,
    constant=True,
    spherical=False,
    bw=None,
):
    s = time.time()

    coords_train = X_train[:, 0:2]
    covariates_train = X_train[:, 2:]

    coords_test = X_test[:, 0:2]
    covariates_test = X_test[:, 2:]

    if bw is None:
        sel_bw = Sel_BW(
            y=y_train,
            X_loc=covariates_train,
            coords=coords_train,
            family=family,
            kernel=kernel,
            fixed=fixed,
            constant=constant,
            spherical=spherical,
        )
        bw = sel_bw.search(criterion="CV")
        logger.info(f"Found BW value of {bw}.")

    model = GWR(
        coords=coords_train,
        y=y_train,
        X=covariates_train,
        bw=bw,
        family=family,
        kernel=kernel,
        fixed=fixed,
        constant=constant,
        spherical=spherical,
    )

    model.fit()

    e = time.time()
    model_training_time = e - s

    results_is = model.predict(
        points=coords_train,
        P=covariates_train,
        exog_scale=model.exog_scale,
        exog_resid=model.exog_resid,
    )
    y_pred_is = results_is.predictions

    s = time.time()

    results_oos = model.predict(
        points=coords_test,
        P=covariates_test,
        exog_scale=model.exog_scale,
        exog_resid=model.exog_resid,
    )
    y_pred_oos = results_oos.predictions

    e = time.time()
    model_oos_prediction_time = e - s

    return y_pred_is, y_pred_oos, model_training_time, model_oos_prediction_time


def linear_kriging(
    X_train,
    X_test,
    y_train,
):
    s = time.time()

    coords_train = X_train[:, 0:2]
    covariates_train = X_train[:, 2:]

    coords_test = X_test[:, 0:2]
    covariates_test = X_test[:, 2:]

    m = LinearRegression(normalize=True, copy_X=True, fit_intercept=False, n_jobs=-1)
    m.fit(covariates_train, y_train)
    m_pred = m.predict(covariates_train)
    res = y_train - m_pred

    kernel = ConstantKernel() * RBF(10, (1e-2, 1e2)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    gp.fit(coords_train, res)

    e = time.time()
    model_training_time = e - s

    y_pred_is = m.predict(covariates_train) + gp.predict(coords_train)

    s = time.time()
    y_pred_oos = m.predict(covariates_test) + gp.predict(coords_test)
    e = time.time()
    model_oos_prediction_time = e - s

    return y_pred_is, y_pred_oos, model_training_time, model_oos_prediction_time


def rf_kriging(
    X_train,
    X_test,
    y_train,
):
    s = time.time()

    coords_train = X_train[:, 0:2]
    covariates_train = X_train[:, 2:]

    coords_test = X_test[:, 0:2]
    covariates_test = X_test[:, 2:]

    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=50, n_jobs=-1)
    m.fit(covariates_train, y_train.squeeze())
    m_pred = m.predict(covariates_train)
    res = y_train.squeeze() - m_pred

    kernel = ConstantKernel() * RBF(10, (1e-2, 1e2)) + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1)
    gp.fit(coords_train, res)

    e = time.time()
    model_training_time = e - s

    y_pred_is = m.predict(covariates_train) + gp.predict(coords_train)
    y_pred_is = y_pred_is.reshape(-1, 1)

    s = time.time()
    y_pred_oos = m.predict(covariates_test) + gp.predict(coords_test)
    y_pred_oos = y_pred_oos.reshape(-1, 1)
    e = time.time()
    model_oos_prediction_time = e - s

    return y_pred_is, y_pred_oos, model_training_time, model_oos_prediction_time


def neural_network(
    X_train,
    X_test,
    y_train,
    verbose=False,
):
    s = time.time()

    K.clear_session()  # just in case, clear Keras session/TF graph

    # Scale data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_standardized = scaler.transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # Build model
    model = Sequential()

    model.add(
        Dense(
            64,
            kernel_initializer="normal",
            input_dim=X_train.shape[1],
            activation="relu",
        )
    )

    model.add(Dense(256, kernel_initializer="normal", activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(256, kernel_initializer="normal", activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(64, kernel_initializer="normal", activation="relu"))
    model.add(Dropout(rate=0.5))

    # Ouput Layer
    model.add(Dense(1, kernel_initializer="normal", activation="linear"))

    # Compile the network
    model.compile(loss="mse", optimizer="adam", metrics=["mse"])
    if verbose:
        model.summary()

    # Train the network
    model.fit(
        X_train_standardized,
        y_train,
        epochs=1500,
        batch_size=64,
        verbose=int(verbose),
    )

    e = time.time()
    model_training_time = e - s

    # Predict
    y_pred_is = model.predict(X_train_standardized)

    s = time.time()
    y_pred_oos = model.predict(X_test_standardized)
    e = time.time()
    model_oos_prediction_time = e - s

    return y_pred_is, y_pred_oos, model_training_time, model_oos_prediction_time


def baseline_full_run(
    project_data_pipeline,
    project_data_pipeline_kwargs,
    model_base_name,
    model_run_function,
    model_run_kwargs,
    evaluation_function,
    PROJECT_NAME,
):
    settings = {
        "project_data_pipeline_kwargs": project_data_pipeline_kwargs,
        "model_run_kwargs": model_run_kwargs,
        "evaluation_function": evaluation_function,
    }

    test_json_serializable(d=settings, name="settings")

    model_name = get_model_name(model_base_name=model_base_name)

    logger.info(f"Running baseline {model_base_name}.")
    logger.info(f"Settings are {settings}")
    logger.info(f"Full model name is {model_name}.")

    (
        X_train,
        X_test,
        Z,
        y_train,
        y_test,
        locs_poi,
        coords_poi,
        coords_train,
        coords_test,
        typeIndicator,
        M,
        typeMinDist,
        scale_x,
        scale_y,
        poi_types,
    ) = project_data_pipeline(**project_data_pipeline_kwargs)

    (
        y_pred_is,
        y_pred_oos,
        model_training_time,
        model_oos_prediction_time,
    ) = model_run_function(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        **model_run_kwargs,
    )

    model_evaluation = evaluation_function(
        y_train=y_train,
        y_test=y_test,
        y_pred_is=y_pred_is,
        y_pred_oos=y_pred_oos,
    )

    information = get_information_dict(
        settings=settings,
        model_evaluation=model_evaluation,
        model_base_name=model_base_name,
        model_name=model_name,
        model_pure_training_time=model_training_time,
        total_training_time=None,
        model_oos_prediction_time=model_oos_prediction_time,
    )

    logger.info(json.dumps(obj=information, cls=poi.helpers.CustomEncoder, indent=4))
    with open(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.information_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        json.dump(obj=information, fp=f, cls=poi.helpers.CustomEncoder, indent=4)


def run_all_baselines(
    project_data_pipeline,
    project_data_pipeline_kwargs_options,
    models,
    evaluation_function,
    PROJECT_NAME,
):
    for project_data_pipeline_kwargs in project_data_pipeline_kwargs_options:
        for model in models:
            model_base_name = model["model_base_name"]
            model_run_function = model["model_run_function"]
            model_run_kwargs = model["model_run_kwargs"]

            baseline_full_run(
                project_data_pipeline=project_data_pipeline,
                project_data_pipeline_kwargs=project_data_pipeline_kwargs,
                model_base_name=model_base_name,
                model_run_function=model_run_function,
                model_run_kwargs=model_run_kwargs,
                evaluation_function=evaluation_function,
                PROJECT_NAME=PROJECT_NAME,
            )
