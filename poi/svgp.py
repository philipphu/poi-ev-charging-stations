import datetime as dt
import json
from loguru import logger
import numpy as np
import pandas as pd
import pickle
import scipy.spatial.distance
import time
import warnings

with warnings.catch_warnings():
    # to hide the FutureWarning occuring during the import
    warnings.filterwarnings(action="ignore", category=FutureWarning)
    import tensorflow as tf

import gpflow
from gpflow.mean_functions import MeanFunction
from gpflow.kernels import Kernel
from gpflow.decors import params_as_tensors
from gpflow.params.parameter import Parameter
from gpflow import settings
from gpflow import transforms
from gpflow.training import NatGradOptimizer, AdamOptimizer

import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

import poi
from poi.helpers import (
    compute_rmse,
    gaussian,
    compute_var,
    get_information_dict,
    get_model_name,
    test_json_serializable,
)
import poi.outputs

# =============================================================================
# Helper function to calculate distance-weighted POI impact:
# =============================================================================
def square_dist(X, X2):
    """
    Returns ((X - X2ᵀ)/lengthscales)².
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    """
    Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.matrix_transpose(Xs)
        return dist

    X2s = tf.reduce_sum(tf.square(X2), axis=-1, keepdims=True)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += Xs + tf.matrix_transpose(X2s)
    return dist


class SlicedLinear(MeanFunction):
    """
    y_i = A x_i
    """

    def __init__(self, A=None, p=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = np.ones((1, 1)) if A is None else A
        MeanFunction.__init__(self)
        self.A = Parameter(np.atleast_2d(A), dtype=settings.float_type)
        self.p = p

    @params_as_tensors
    def __call__(self, X):
        return tf.matmul(X[:, 2 : self.p + 2], self.A)


class SlicedNN(MeanFunction):
    """
    y_i = A x_i
    """

    def __init__(self, p=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        self.p = p

        weights1 = np.random.rand(p, 4)
        weights2 = np.random.rand(4, 1)

        self.weights1 = Parameter(weights1, dtype=settings.float_type)
        self.weights2 = Parameter(weights2, dtype=settings.float_type)

    @params_as_tensors
    def __call__(self, X):
        layer1 = tf.nn.relu(tf.matmul(X[:, 2 : self.p + 2], self.weights1))
        output = tf.nn.relu(tf.matmul(layer1, self.weights2))
        return output


class SlicedNNwithbias(MeanFunction):
    """
    y_i = A x_i
    """

    def __init__(self, p=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        self.p = p

        hidden_units = 8

        bias1 = 1
        bias2 = 1

        weights1 = np.zeros((p, hidden_units))
        weights2 = np.zeros((hidden_units, 1))

        self.weights1 = Parameter(weights1, dtype=settings.float_type)
        self.weights2 = Parameter(weights2, dtype=settings.float_type)

        self.bias1 = Parameter(bias1, dtype=settings.float_type)
        self.bias2 = Parameter(bias2, dtype=settings.float_type)

    @params_as_tensors
    def __call__(self, X):
        layer1 = tf.nn.relu(tf.matmul(X[:, 2 : self.p + 2], self.weights1) + self.bias1)
        output = tf.matmul(layer1, self.weights2) + self.bias2
        return output


class POI(Kernel):
    def __init__(
        self,
        input_dim,
        locs_poi_j,
        active_dims=None,
        name=None,
        lengthscale=None,
        effects=None,
        locs=None,
        mindist=0.5,
        kernel_type="Linear",
    ):
        super().__init__(input_dim, active_dims, name=name)
        effects = np.random.uniform(low=0.5, high=1.0) if effects is None else effects
        MeanFunction.__init__(self)
        self.locs_poi_j = locs_poi_j.astype(np.float64)
        self.kernel_type = kernel_type

        self.lengthscale = 0  # Parameter(2, transform=transforms.Logistic(a=mindist, b=10), dtype=settings.float_type)
        self.effects = Parameter(
            effects,
            transform=transforms.Logistic(a=0.01, b=1),
            dtype=settings.float_type,
        )
        self.distmax = Parameter(
            0.5, transform=transforms.Logistic(a=0, b=1.5), dtype=settings.float_type
        )

    @params_as_tensors
    def transformed_sorted_distances(self, locs, locs2=None):
        """
        Returns h(-d(s_i, w_j)), where d is the squared distance between
        obs and POI. If d(s_i, w_j) > d(s_i, w_{N_cutoff}), then it returns 0.
        Currently, the decay fct is h = exp(-x^2/x)
        """
        if locs2 is None:
            d = tf.sqrt(square_dist(self.locs_poi_j, locs))
            if self.kernel_type == "Gaussian":
                d_new = self.effects * tf.exp(-0.5 * (d / self.distmax) ** 2)
            elif self.kernel_type == "Linear":
                d_new = self.effects * tf.nn.relu((-1 / self.distmax) * d + 1)
            else:
                raise Exception(f"Unsupported kernel type of {self.kernel_type}.")
            out = tf.matmul(d_new, d_new, transpose_a=True)
            return out + tf.eye(tf.shape(out)[0], dtype=tf.float64) * 1e-4
        else:
            d = tf.sqrt(square_dist(self.locs_poi_j, locs))
            d2 = tf.sqrt(square_dist(self.locs_poi_j, locs2))
            if self.kernel_type == "Gaussian":
                d_new = self.effects * tf.exp(-0.5 * (d / self.distmax) ** 2)
                d2_new = self.effects * tf.exp(-0.5 * (d2 / self.distmax) ** 2)
            elif self.kernel_type == "Linear":
                d_new = self.effects * tf.nn.relu((-1 / self.distmax) * d + 1)
                d2_new = self.effects * tf.nn.relu((-1 / self.distmax) * d2 + 1)
            else:
                raise Exception(f"Unsupported kernel type of {self.kernel_type}.")
            out = tf.matmul(d_new, d2_new, transpose_a=True)
            return out

    @params_as_tensors
    def K(self, X, X2=None):
        if X2 is None:
            out = self.transformed_sorted_distances(X[:, 0:2])
            return out
        else:
            out = self.transformed_sorted_distances(X[:, 0:2], X2[:, 0:2])
            return out

    @params_as_tensors
    def Kdiag(self, X):
        return tf.diag_part(self.K(X))


# =============================================================================
# Build GP model
# =============================================================================
def build_gp_model(
    X_train,
    Z,
    y_train,
    locs_poi,
    typeIndicator,
    typeMinDist,
    D,  # Input dimensions for kernel
    spatial_kernel_lengthscales,
    mean_fct,
    likelihood,
    poi_kernel_type,
    model_name,
):
    feature = mf.SharedIndependentMof(gpflow.features.InducingPoints(Z.copy()))

    # Define POI kernels
    n_poi_types = typeIndicator.shape[1]  # number of poi types
    kern_list = []
    for typIdx in range(n_poi_types):
        locs_poi_of_type = locs_poi[
            typeIndicator[:, typIdx] == 1, :
        ]  # select pois of type typIdx
        kern = POI(
            input_dim=D,
            locs_poi_j=locs_poi_of_type,
            active_dims=None,
            name=str(typIdx),
            lengthscale=5,
            effects=0.5,
            locs=Z[:, 0:2],
            mindist=typeMinDist[typIdx],
            kernel_type=poi_kernel_type,
        )
        kern_list.append(kern)

    # Add spatial kernel
    kern_spatial = gpflow.kernels.Matern32(
        input_dim=D,
        lengthscales=spatial_kernel_lengthscales,
        name="Matern32",
    )
    kern_list.append(kern_spatial)

    # Define kernel list
    L = len(kern_list)
    W = np.ones((L, 1))
    W_t = np.transpose(W)
    kernel = mk.SeparateMixedMok(kern_list, W=W_t)

    M = Z.shape[0]
    q_mu = np.random.normal(0.0, 1, (M, L))
    q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

    model = gpflow.models.SVGP(
        X=X_train,
        Y=y_train,
        kern=kernel,
        likelihood=likelihood,
        feat=feature,
        whiten=True,  # minibatch_size=len(X_train),
        mean_function=mean_fct,  # + mean_poi,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        name=model_name,
    )

    model.feature.trainable = False
    model.kern.W.trainable = False

    return model


# =============================================================================
# Restore GP model
# =============================================================================
def restore_gp_model(save_path, **build_gp_model_kwargs):
    with gpflow.defer_build():
        model = build_gp_model(
            **build_gp_model_kwargs
        )

    tf.local_variables_initializer()
    tf.global_variables_initializer()

    tf_graph = model.enquire_graph()
    tf_session = model.enquire_session()
    model.compile(tf_session)

    saver = tf.train.Saver()
    save_path = saver.restore(
        sess=tf_session,
        save_path=save_path,
    )
    logger.info(f"Model loaded from path: {save_path}")

    return model


def K_POI(
    X1,
    X2,
    locs_poi_j,
    # lengthscale,
    distmax,
    effect,
):
    d1 = scipy.spatial.distance.cdist(X1, locs_poi_j)
    d1_new = effect * np.maximum((-1 / distmax) * d1 + 1, 0)

    d2 = scipy.spatial.distance.cdist(X2, locs_poi_j)
    d2_new = effect * np.maximum((-1 / distmax) * d2 + 1, 0)

    out = np.matmul(d1_new, np.transpose(d2_new))

    return out


def predict_g(
    effect,
    # lengthscale,
    distmax,
    m,
    S,
    X,
    Z,
    pois,
):
    # Knm = K_POI(
    #     X1=X,
    #     X2=Z,
    #     locs_poi_j=pois,
    #     # lengthscale=lengthscale,
    #     distmax=distmax,
    #     effect=effect,
    # )
    Kmn = K_POI(
        X1=Z,
        X2=X,
        locs_poi_j=pois,
        # lengthscale=lengthscale,
        distmax=distmax,
        effect=effect,
    )
    Kmm = (
        K_POI(
            X1=Z,
            X2=Z,
            locs_poi_j=pois,
            # lengthscale=lengthscale,
            distmax=distmax,
            effect=effect,
        )
        + np.eye(Z.shape[0]) * 1e-8
    )
    Knn = K_POI(
        X1=X,
        X2=X,
        locs_poi_j=pois,
        # lengthscale=lengthscale,
        distmax=distmax,
        effect=effect,
    )

    Lm = np.linalg.cholesky(Kmm)
    A = np.matmul(np.linalg.inv(Lm), Kmn)
    m_out = np.matmul(np.transpose(A), m)

    # S_out = Knn + np.matmul(np.matmul(np.transpose(A), (S - Kmm)), A)

    return m_out  # , S_out


def individual_poi_posterior(
    locs_poi_j,
    distmax,
    effect,
    X,  # only n*2 shaped so only coordinates
    Z,  # only n*2 shaped so only coordinates
    m_g,
    S_g,
):
    g = predict_g(
        effect=effect,
        distmax=distmax,
        m=m_g,
        S=S_g,
        X=X,
        Z=Z,
        pois=locs_poi_j,
    )

    return g


def posterioreffect(
    locs_poi_j,
    X_test,
    Z,
    distmax,
    effect,
    q_mu_j,
    q_sqrt_j,
):
    X = X_test[:, 0:2]
    d = scipy.spatial.distance.cdist(locs_poi_j, X)
    k = np.maximum((-1 / distmax) * d + 1, 0)

    h = individual_poi_posterior(
        locs_poi_j=locs_poi_j,
        distmax=distmax,
        effect=effect,
        X=X,
        Z=Z[:, 0:2],
        m_g=q_mu_j,
        S_g=q_sqrt_j,
    )
    alpha = np.matmul(h, np.linalg.pinv(k))

    return alpha


def get_effects_df(
    kern_params,
    poi_types,  # name of poi types
    locs_poi,
    typeIndicator,
    X_test,
    Z,
    q_mu,
    q_sqrt,
    model_name,
):
    results = pd.DataFrame()

    for typIdx in range(len(poi_types)):
        locs_poi_of_type = locs_poi[
            typeIndicator[:, typIdx] == 1, :
        ]  # select pois of type typIdx
        distmax = kern_params[f"{model_name}/kern/kernels/{typIdx}/distmax"]
        effect_var = kern_params[f"{model_name}/kern/kernels/{typIdx}/effects"]
        alpha = posterioreffect(
            locs_poi_j=locs_poi_of_type,
            X_test=X_test,
            Z=Z,
            distmax=distmax,
            effect=effect_var,
            q_mu_j=q_mu[:, typIdx],
            q_sqrt_j=q_sqrt[typIdx, :, :],
        )
        # effect_size = np.mean(np.abs(alpha))
        effect_size = np.mean(alpha)
        effect_size_std = np.std(alpha)
        effect_size_abs = np.mean(np.abs(alpha))
        effect_size_abs_std = np.std(np.abs(alpha))
        row = pd.Series(
            {
                "Effect_var": effect_var,
                "Cut_off": distmax,
                "Effect_size": effect_size,
                "Effect_size_std": effect_size_std,
                "Effect_size_abs": effect_size_abs,
                "Effect_size_abs_std": effect_size_abs_std,
            },
            name=poi_types[typIdx],
        )
        results = results.append(row)

    results["Effect_size_percentage"] = (
        results["Effect_size"] / results["Effect_size"].max() * 100
    )

    results.index.name = "poi_type"
    results = results.reset_index()

    return results


def get_matern32_kernel_params(
    kern_params,
    model_name,
    matern32_kernel_index,
):
    lengthscales = kern_params[
        f"{model_name}/kern/kernels/{matern32_kernel_index}/lengthscales"
    ]
    variance = kern_params[
        f"{model_name}/kern/kernels/{matern32_kernel_index}/variance"
    ]

    return {"lengthscales": lengthscales, "variance": variance}


def K_spatial(
    X1,
    X2,
    lengthscale,
    variance,
):
    d = scipy.spatial.distance.cdist(X1, X2)
    sqrt3 = np.sqrt(3.0)
    return variance * (1.0 + sqrt3 * d / lengthscale) * np.exp(-sqrt3 * d / lengthscale)


def predict_spatial(
    X,  # only n*2 shaped so only coordinates
    Z,  # only n*2 shaped so only coordinates
    lengthscale,
    variance,
    m_g,
):
    Kmn = K_spatial(Z, X, lengthscale, variance)
    Kmm = K_spatial(Z, Z, lengthscale, variance) + np.eye(Z.shape[0]) * 1e-8

    Lm = np.linalg.cholesky(Kmm)
    A = np.matmul(np.linalg.inv(Lm), Kmn)
    m_out = np.matmul(np.transpose(A), m_g)

    # S_out = Knn+np.matmul(np.matmul(np.transpose(A), (S_g-Kmm)), A)

    return m_out  # , S_out


# =============================================================================
# Train GP model
# =============================================================================
def train_gp_model(
    model: gpflow.models.SVGP,
    iterations: int,
    PROJECT_NAME: str,
    model_name: str,
    logging_iteration: int,
    monitor_iteration: int,
    evaluation_iteration: int,
    poi_types,
    locs_poi,
    typeIndicator,
    X_train,
    X_test,
    y_train,
    y_test,
    Z,
    evaluation_function,
):
    tf.local_variables_initializer()
    tf.global_variables_initializer()

    tf_session = model.enquire_session()
    model.compile(tf_session)

    op_adam = AdamOptimizer(0.01).make_optimize_tensor(model)

    saver = tf.train.Saver()

    likelihood_monitor_df = pd.DataFrame()
    effects_monitor_df = pd.DataFrame()
    # Z_monitor = []
    matern32_monitor_df = pd.DataFrame()
    model_evaluation_monitor_df = pd.DataFrame()

    logger.info(f"Starting SVGP training of model {model_name}.")

    training_start_time = time.time()
    model_pure_training_time = 0

    for it in range(1, iterations + 1):
        s = time.time()
        tf_session.run(op_adam)
        e = time.time()
        model_pure_training_time += e - s
        if it % logging_iteration == 0 or it == 1 or it == iterations:
            likelihood = tf_session.run(model.likelihood_tensor)
            time_elapsed = time.time() - training_start_time
            time_remaining = time_elapsed * ((iterations / it) - 1)
            logger.info(
                f"{it}/{iterations}, ELBO = {likelihood:.4f}, "
                f"Time elapsed: {dt.timedelta(seconds=time_elapsed)}, "
                f"Time remaining: {dt.timedelta(seconds=time_remaining)}."
            )
        if it % monitor_iteration == 0 or it == 1 or it == iterations:
            # Monitor likelihood
            likelihood = tf_session.run(model.likelihood_tensor)
            likelihood_monitor_df = likelihood_monitor_df.append(
                pd.Series({"iteration": it, "likelihood": likelihood}),
                ignore_index=True,
            )
            # Read kern params with tf_session
            kern_params = model.kern.read_values(
                session=tf_session
            )  # using session to get actual current data
            # TODO: test without using session again but shouldn't work
            # Monitor effects
            effects_df = get_effects_df(
                kern_params=kern_params,
                poi_types=poi_types,  # name of poi types
                locs_poi=locs_poi,
                typeIndicator=typeIndicator,
                X_test=X_test,
                Z=Z,
                q_mu=model.q_mu.read_value(
                    session=tf_session
                ),
                q_sqrt=model.q_sqrt.read_value(session=tf_session),
                model_name=model.name,
            )
            effects_df["iteration"] = it
            effects_monitor_df = effects_monitor_df.append(effects_df)
            # Monitor Z
            # Z_monitor.append(
            #     {
            #         "iteration": it,
            #         "Z": model.feature.feat.read_values(session=tf_session)[
            #             f"{model.name}/feature/feat/Z"
            #         ],
            #     }
            # )
            # Monitor Matern32 kernel
            matern32_kernel_params = get_matern32_kernel_params(
                kern_params=kern_params,
                model_name=model.name,
                matern32_kernel_index=typeIndicator.shape[
                    1
                ],  # =number of poi types, poi kernels start with index 0
            )
            matern32_kernel_params["iteration"] = it
            matern32_monitor_df = matern32_monitor_df.append(
                matern32_kernel_params,
                ignore_index=True,
            )
        if it % evaluation_iteration == 0 or it == 1 or it == iterations:
            # Monitor evaluation
            model.anchor(tf_session)
            model_evaluation, model_oos_prediction_time = gp_model_evaluation(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                evaluation_function=evaluation_function,
            )
            model_evaluation["iteration"] = it
            model_evaluation_monitor_df = model_evaluation_monitor_df.append(
                model_evaluation,
                ignore_index=True,
            )

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    logger.info(
        f"Finished SVGP training. "
        f"Pure training time was {model_pure_training_time:.2f} seconds. "
        f"Total training time was {total_training_time:.2f} seconds."
    )

    # Save monitors
    likelihood_monitor_df.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.likelihood_monitor_filename,
            mkdir=True,
        ),
        index=False,
    )
    logger.info("Saved likelihood monitor.")

    effects_monitor_df.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.effects_monitor_filename,
            mkdir=True,
        ),
        index=False,
    )
    logger.info("Saved effects monitor.")

    # with open(
    #     poi.paths.project_output_model_file(
    #         project=PROJECT_NAME,
    #         model=model_name,
    #         filename=poi.Z_monitor_filename,
    #         mkdir=True,
    #     ),
    #     "wb",
    # ) as f:
    #     pickle.dump(Z_monitor, f)
    # logger.info("Saved Z monitor.")

    matern32_monitor_df.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.matern32_monitor_filename,
            mkdir=True,
        ),
        index=False,
    )
    logger.info("Saved Matern32 monitor.")

    model_evaluation_monitor_df.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.model_evaluation_monitor_filename,
            mkdir=True,
        ),
        index=False,
    )
    logger.info("Saved model evaluation monitor.")

    q_mu_final = model.q_mu.read_value(session=tf_session)
    q_sqrt_final = model.q_sqrt.read_value(session=tf_session)
    np.savez_compressed(
        file=poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.q_final_filename,
            mkdir=True,
        ),
        q_mu=q_mu_final,
        q_sqrt=q_sqrt_final,
    )
    logger.info("Saved q final.")

    # Save model
    # TODO: fix problems with restore and then save model again
    # save_path = saver.save(
    #     sess=tf_session,
    #     save_path=poi.paths.project_output_model_file(
    #         project=PROJECT_NAME,
    #         model=model_name,
    #         filename=poi.model_filename,
    #         mkdir=True,
    #     ),
    #     global_step=iterations,
    # )
    # logger.info(f"SVGP model saved in path: {save_path}")

    model.anchor(tf_session)

    out_kern_df = model.kern.as_pandas_table()
    out_kern_df.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.model_kern_filename,
            mkdir=True,
        ),
        index=True,
    )
    logger.info("Saved model kern.")

    out_likelihood_df = model.likelihood.as_pandas_table()
    if out_likelihood_df is not None:
        out_likelihood_df.to_csv(
            poi.paths.project_output_model_file(
                project=PROJECT_NAME,
                model=model_name,
                filename=poi.model_likelihood_filename,
                mkdir=True,
            ),
            index=True,
        )
        logger.info("Saved model likelihood.")
    else:
        logger.info("Model likelihood was None. Did not save therefore.")

    out_mean_function = model.mean_function.as_pandas_table()
    out_mean_function.to_csv(
        poi.paths.project_output_model_file(
            project=PROJECT_NAME,
            model=model_name,
            filename=poi.model_mean_function_filename,
            mkdir=True,
        ),
        index=True,
    )
    logger.info("Saved model mean function.")

    return model, model_pure_training_time


def gp_model_evaluation(
    model: gpflow.models.SVGP,  # already anchored
    X_train,
    X_test,
    y_train,
    y_test,
    evaluation_function,
):
    pred_mu_is, pred_var_is = model.predict_y(X_train)

    s = time.time()
    pred_mu_oos, pred_var_oos = model.predict_y(X_test)
    e = time.time()
    model_oos_prediction_time = e - s

    model_evaluation = evaluation_function(
        y_train=y_train,
        y_test=y_test,
        y_pred_is=pred_mu_is,
        y_pred_oos=pred_mu_oos,
    )

    ### Debug results
    try:
        var_model = model.likelihood.variance.value.item()
    except AttributeError:
        pass
    else:
        var_is = compute_var(y1=y_train, y2=pred_mu_is)
        var_oos = compute_var(y1=y_test, y2=pred_mu_oos)

        total_gaussian_lik_is_with_model_var = np.sum(
            gaussian(x=y_train, mu=pred_mu_is, var=var_model)
        )
        mean_gaussian_lik_is_with_model_var = np.mean(
            gaussian(x=y_train, mu=pred_mu_is, var=var_model)
        )
        total_gaussian_lik_oos_with_model_var = np.sum(
            gaussian(x=y_test, mu=pred_mu_oos, var=var_model)
        )
        mean_gaussian_lik_oos_with_model_var = np.mean(
            gaussian(x=y_test, mu=pred_mu_oos, var=var_model)
        )

        total_gaussian_lik_is_with_computed_var = np.sum(
            gaussian(x=y_train, mu=pred_mu_is, var=var_is)
        )
        mean_gaussian_lik_is_with_computed_var = np.mean(
            gaussian(x=y_train, mu=pred_mu_is, var=var_is)
        )
        total_gaussian_lik_oos_with_computed_var = np.sum(
            gaussian(x=y_test, mu=pred_mu_oos, var=var_oos)
        )
        mean_gaussian_lik_oos_with_computed_var = np.mean(
            gaussian(x=y_test, mu=pred_mu_oos, var=var_oos)
        )

        result_debug = {
            "var_model": var_model,
            "var_is": var_is,
            "var_oos": var_oos,
            "total_gaussian_lik_is_with_model_var": total_gaussian_lik_is_with_model_var,
            "mean_gaussian_lik_is_with_model_var": mean_gaussian_lik_is_with_model_var,
            "total_gaussian_lik_oos_with_model_var": total_gaussian_lik_oos_with_model_var,
            "mean_gaussian_lik_oos_with_model_var": mean_gaussian_lik_oos_with_model_var,
            "total_gaussian_lik_is_with_computed_var": total_gaussian_lik_is_with_computed_var,
            "mean_gaussian_lik_is_with_computed_var": mean_gaussian_lik_is_with_computed_var,
            "total_gaussian_lik_oos_with_computed_var": total_gaussian_lik_oos_with_computed_var,
            "mean_gaussian_lik_oos_with_computed_var": mean_gaussian_lik_oos_with_computed_var,
        }

        model_evaluation.update(result_debug)
    ###

    return model_evaluation, model_oos_prediction_time


def gp_model_full_run(
    project_data_pipeline,
    project_data_pipeline_kwargs,
    mean_fct_class,
    mean_fct_class_kwargs,
    likelihood_class,
    likelihood_class_kwargs,
    poi_kernel_type,
    evaluation_function,
    PROJECT_NAME,
    D,
    spatial_kernel_lengthscales,
    iterations,
    logging_iteration,
    monitor_iteration,
    evaluation_iteration,
):
    settings = {
        "project_data_pipeline_kwargs": project_data_pipeline_kwargs,
        "mean_fct_class": mean_fct_class,
        "mean_fct_class_kwargs": mean_fct_class_kwargs,
        "likelihood_class": likelihood_class,
        "likelihood_class_kwargs": likelihood_class_kwargs,
        "poi_kernel_type": poi_kernel_type,
        "evaluation_function": evaluation_function,
        "D": D,
        "spatial_kernel_lengthscales": spatial_kernel_lengthscales,
        "iterations": iterations,
        "logging_iteration": logging_iteration,
        "monitor_iteration": monitor_iteration,
        "evaluation_iteration": evaluation_iteration,
    }

    test_json_serializable(d=settings, name="settings")
    logger.info(json.dumps(obj=settings, cls=poi.helpers.CustomEncoder, indent=4))

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

    model_base_name = "svgp"
    model_name = get_model_name(model_base_name=model_base_name)

    s = time.time()

    with gpflow.defer_build():
        mean_fct = mean_fct_class(**mean_fct_class_kwargs)
        likelihood = likelihood_class(**likelihood_class_kwargs)

        model = build_gp_model(
            X_train=X_train,
            Z=Z,
            y_train=y_train,
            locs_poi=locs_poi,
            typeIndicator=typeIndicator,
            typeMinDist=typeMinDist,
            D=D,
            spatial_kernel_lengthscales=spatial_kernel_lengthscales,
            mean_fct=mean_fct,
            likelihood=likelihood,
            poi_kernel_type=poi_kernel_type,
            model_name=model_name,
        )

    model, model_pure_training_time = train_gp_model(
        model=model,
        iterations=iterations,
        PROJECT_NAME=PROJECT_NAME,
        model_name=model_name,
        logging_iteration=logging_iteration,
        monitor_iteration=monitor_iteration,
        evaluation_iteration=evaluation_iteration,
        poi_types=poi_types,
        locs_poi=locs_poi,
        typeIndicator=typeIndicator,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        Z=Z,
        evaluation_function=evaluation_function,
    )

    e = time.time()
    total_training_time = e - s

    model_evaluation, model_oos_prediction_time = gp_model_evaluation(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        evaluation_function=evaluation_function,
    )

    information = get_information_dict(
        settings=settings,
        model_evaluation=model_evaluation,
        model_base_name=model_base_name,
        model_name=model_name,
        model_pure_training_time=model_pure_training_time,
        total_training_time=total_training_time,
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
        json.dump(
            obj=information,
            fp=f,
            cls=poi.helpers.CustomEncoder,
            indent=4,
        )
    logger.info("Saved information json file.")

    poi.outputs.create_svgp_training_monitor_plot(
        project=PROJECT_NAME,
        model_name=model_name,
    )
    logger.info("Saved training monitor plot.")

    poi.outputs.create_svgp_decay_plot(
        project=PROJECT_NAME,
        model_name=model_name,
    )
    logger.info("Saved decay plot.")

    return model
