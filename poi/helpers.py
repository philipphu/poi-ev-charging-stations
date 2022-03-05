import cpuinfo
import datetime as dt
import json
from loguru import logger
import numpy as np
import os
import pandas as pd
import pathlib
from typing import List, Optional, Tuple, Union

from scipy.special import gammaln

import poi


def compute_var(y1, y2):
    return np.var(y1 - y2)


def compute_rmse(y1, y2):
    return np.sqrt(np.mean((y1 - y2) ** 2))


def gaussian(x, mu, var):
    return -0.5 * (np.log(2 * np.pi) + np.log(var) + np.square(mu - x) / var)


def gaussian_lik(
    y1, y2, var=None
):
    if var is None:
        var = compute_var(y1, y2)
    lik = gaussian(x=y1, mu=y2, var=var)
    total_lik = np.sum(lik)
    mean_lik = np.mean(lik)
    return total_lik, mean_lik


def rmse_gaussian_lik_evaluation(
    y_train,
    y_test,
    y_pred_is,
    y_pred_oos,
):
    rmse_is = compute_rmse(y1=y_train, y2=y_pred_is)
    total_lik_is, mean_lik_is = gaussian_lik(y1=y_train, y2=y_pred_is)

    rmse_oos = compute_rmse(y1=y_test, y2=y_pred_oos)
    total_lik_oos, mean_lik_oos = gaussian_lik(y1=y_test, y2=y_pred_oos)

    evaluation = get_evaluation_dict(
        rmse_is=rmse_is,
        total_lik_is=total_lik_is,
        mean_lik_is=mean_lik_is,
        rmse_oos=rmse_oos,
        total_lik_oos=total_lik_oos,
        mean_lik_oos=mean_lik_oos,
    )

    return evaluation


def poisson(x, mu):
    return -mu + x * np.log(mu) - gammaln(x + 1)


def poisson_lik(y1, y2):
    lik = poisson(x=y1, mu=np.clip(y2, a_min=0.0001, a_max=None))
    total_lik = np.sum(lik)
    mean_lik = np.mean(lik)
    return total_lik, mean_lik


def rmse_poisson_lik_evaluation(
    y_train,
    y_test,
    y_pred_is,
    y_pred_oos,
):
    rmse_is = compute_rmse(y1=y_train, y2=y_pred_is)
    total_lik_is, mean_lik_is = poisson_lik(y1=y_train, y2=y_pred_is)

    rmse_oos = compute_rmse(y1=y_test, y2=y_pred_oos)
    total_lik_oos, mean_lik_oos = poisson_lik(y1=y_test, y2=y_pred_oos)

    evaluation = get_evaluation_dict(
        rmse_is=rmse_is,
        total_lik_is=total_lik_is,
        mean_lik_is=mean_lik_is,
        rmse_oos=rmse_oos,
        total_lik_oos=total_lik_oos,
        mean_lik_oos=mean_lik_oos,
    )

    return evaluation


def get_system_resources():
    system_resources = {}

    try:
        os_sched_affinity = len(os.sched_getaffinity(0))
    except:
        os_sched_affinity = None
    system_resources["os_sched_affinity"] = os_sched_affinity

    try:
        os_cpu_count = os.cpu_count()
    except:
        os_cpu_count = None
    system_resources["os_cpu_count"] = os_cpu_count

    try:
        cpuinfo_cpu_info = cpuinfo.get_cpu_info()
    except:
        cpuinfo_cpu_info = None
    system_resources["cpuinfo_cpu_info"] = cpuinfo_cpu_info

    return system_resources


def get_information_dict(
    settings: dict,
    model_evaluation: dict,
    model_base_name: str,
    model_name: str,
    model_pure_training_time: float,
    total_training_time: Optional[float],
    model_oos_prediction_time: float,
):
    """Returns an information dictionary of the model training and testing process including environment information.

    Args:
        model_base_name:
        settings: Dictionary containing settings for the model run.
        model_evaluation: Dictionary containing the evaluation of the model performance.
        model_base_name: Base name of the model.
        model_name: Name of the model.
        model_pure_training_time: Duration of pure model training in seconds.
        total_training_time: Duration of the whole training in seconds.
        model_oos_prediction_time: Duration of model prediction on test set in seconds.

    Returns:
        Dictionary containing the information about the model run.
    """
    information = {
        "settings": settings,
        "model_evaluation": model_evaluation,
        "model_base_name": model_base_name,
        "model_name": model_name,
        "model_pure_training_time": model_pure_training_time,
        "total_training_time": total_training_time,
        "model_oos_prediction_time": model_oos_prediction_time,
        "finish_time_utc": str(dt.datetime.utcnow()),
        "system_resources": get_system_resources(),
    }

    return information


def get_evaluation_dict(
    rmse_is,
    total_lik_is,
    mean_lik_is,
    rmse_oos,
    total_lik_oos,
    mean_lik_oos,
    **kwargs,
):
    evaluation = {
        "RMSE_IS": rmse_is,
        "TOTAL_LIK_IS": total_lik_is,
        "MEAN_LIK_IS": mean_lik_is,
        "RMSE_OOS": rmse_oos,
        "TOTAL_LIK_OOS": total_lik_oos,
        "MEAN_LIK_OOS": mean_lik_oos,
    }

    evaluation.update(kwargs)

    return evaluation


def step_size_string(a):
    steps = a[1:] - a[:-1]
    average_step_size = np.mean(steps)
    all_steps_equal = np.all(steps == steps[0])

    if all_steps_equal:
        return f"Step size: {average_step_size:.0f}"
    else:
        return f"Average step size: {average_step_size:.0f}"


def get_saved_model_names(
    project: str,
    model_base_name_filter: Optional[Union[str, Tuple[str, ...]]] = None,
) -> List[str]:
    """Finds names of all saved models for the given project with the option to filter.

    Args:
        project: Name of the project.
        model_base_name_filter:
            List of model_base_names to filter for.
            If not None, will only return those model names that start
            with a string given in this tuple.

    Returns:
        List of names of all saved models - potentially filtered - for the given project.
    """
    p = poi.paths.project_output_models_dir(project=project)
    subdirectories = sorted(
        [f for f in p.iterdir() if f.is_dir()]
    )  # non-recursive as we just want model names from the names of the directories in the first level
    model_names = [f.stem for f in subdirectories]

    if model_base_name_filter is not None:
        x: str
        model_names = list(
            filter(lambda x: x.startswith(model_base_name_filter), model_names)
        )

    return model_names


def get_projects_information_dicts(
    projects: Union[str, List[str]],
    model_base_name_filter: Optional[Union[str, Tuple[str, ...]]] = None,
):
    if type(projects) == str:
        projects = [projects]

    projects_information_dicts = {}

    for project in projects:
        projects_information_dicts[project] = {}

        project_model_names = get_saved_model_names(
            project=project,
            model_base_name_filter=model_base_name_filter,
        )

        for model_name in project_model_names:
            with open(
                poi.paths.project_output_model_file(
                    project=project,
                    model=model_name,
                    filename=poi.information_filename,
                    mkdir=False,
                ),
                "r",
            ) as f:
                information = json.load(fp=f)
            model_base_name = information["model_base_name"]
            if not model_base_name in projects_information_dicts[project]:
                projects_information_dicts[project][model_base_name] = []
            projects_information_dicts[project][model_base_name].append(information)

    return projects_information_dicts


def save_df_as_latex(df: pd.DataFrame, path: pathlib.Path, index, **kwargs):
    with open(path, "w") as f:
        f.write(
            df.to_latex(
                index=index,
                float_format=lambda x: "%.3f" % x,
                **kwargs,
            )
        )


def compare_values(value1, value2, name, **parameters):
    comparison = value1 == value2
    if type(comparison) == bool:
        comparison_boolean = comparison
    else:
        # if comparison is not a boolean value yet, it is an array of boolean values
        # in this module so we call .all() to see if all entries are True
        comparison_boolean = comparison.all()

    if comparison_boolean:
        logger.info(f"{parameters}: {name}: {comparison_boolean}")
    else:
        logger.error(f"{parameters}: {name}: {comparison_boolean}")
        raise Exception(f"Comparison failed for {parameters} for {name}")


def get_model_name(model_base_name: str):
    return f"{model_base_name}_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"


def test_json_serializable(d: dict, name: str):
    try:
        json.dumps(obj=d, indent=4, cls=CustomEncoder)
    except TypeError:
        logger.error(f"Error testing that {name} dict is JSON serializable.")
        logger.error(f"Dict was: {d}.")
        for key, value in d.items():
            try:
                json.dumps(obj=key, cls=CustomEncoder)
            except TypeError:
                logger.error(f"Not JSON serializable was the key: {key}.")
            try:
                json.dumps(obj=value, cls=CustomEncoder)
            except TypeError:
                logger.error(
                    f"Not JSON serializable was the value of key: {key} with value: {value}."
                )
        raise


class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if callable(o):
            return o.__name__
        else:
            return str(o)
