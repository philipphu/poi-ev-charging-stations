import datetime as dt
import importlib
import json
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import shapefile
import shutil
from typing import List, Optional, Tuple, Union

import poi
import poi.helpers
from poi.helpers import (
    get_saved_model_names,
    get_projects_information_dicts,
    save_df_as_latex,
)
from poi.plots import (
    svgp_training_monitor_plot,
    cpu_threads_vs_training_time_plot,
    decay_plot,
    latent_poi_influence_plot,
    spatial_variation_plot,
)
import poi.svgp
from poi.project_config import SVGPConfig


def create_all_outputs(projects, project_groups):
    baseline_cpu_threads_selection = 2
    svgp_cpu_threads_selection = 24
    ### Model specific outputs

    # Recreate all svgp model plots
    create_all_svgp_model_plots(projects=projects)

    ### Project specific outputs

    # dataset information file
    for project in projects:
        create_dataset_information_file(project=project)

    # create the baselines table for each project
    for project in projects:
        create_project_baselines_table(
            project=project,
            cpu_threads_selection=baseline_cpu_threads_selection,
        )

    # create the svgp table for each project
    for project in projects:
        create_project_svgp_table(
            project=project,
            cpu_threads_selection=svgp_cpu_threads_selection,
        )

    # create the effects table
    for project in projects:
        create_project_effects_table(project=project)

    # create the performance overview table for each single project
    for project in projects:
        create_performance_overview_table(
            projects=[project],
            baselines_cpu_threads_selection=baseline_cpu_threads_selection,
            svgp_cpu_threads_selection=svgp_cpu_threads_selection,
        )
        create_performance_overview_table(
            projects=[project],
            baselines_cpu_threads_selection=baseline_cpu_threads_selection,
            svgp_cpu_threads_selection=svgp_cpu_threads_selection,
            baseline_feature_types=(None, "both"),
        )

    # create the latex codes for the latent poi influence and spatial variation figure for each single project
    for project in projects:
        create_latex_figure_latent_poi_influence_spatial_variation(project=project)

    # create the best svgp directory for each single project
    for project in projects:
        create_best_svgp_directory(project=project)

    # create the appendix tex file for each single project
    for project in projects:
        create_project_appendix(project=project)

    # create the knot points vs training time and rmse oos plot

    ### Project overlapping outputs

    # create the performance overview table for each project group
    for project_group in project_groups:
        create_performance_overview_table(
            projects=project_group,
            baselines_cpu_threads_selection=baseline_cpu_threads_selection,
            svgp_cpu_threads_selection=svgp_cpu_threads_selection,
        )

    # create the cpu threads vs training time plot
    # create_cpu_threads_vs_training_time_plot(
    #     projects=projects,
    #     model_base_name_filter=poi.baseline_names,
    #     filename=poi.baselines_cpu_threads_vs_training_time_plot_filename,
    # )
    # create_cpu_threads_vs_training_time_plot(
    #     projects=projects,
    #     model_base_name_filter="svgp",
    #     filename=poi.svgp_cpu_threads_vs_training_time_plot_filename,
    # )


def create_dataset_information_file(project):
    logger.info(f"{project} | Dataset information file.")

    project_config_module = importlib.import_module(f"poi.{project}.config")
    svgp_configs: List[SVGPConfig] = getattr(project_config_module, "svgp_configs")

    svgp_config = svgp_configs[0]
    project_data_pipeline = svgp_config.project_data_pipeline
    project_data_pipeline_kwargs = svgp_config.project_data_pipeline_kwargs

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

    n_data_points = len(X_train) + len(X_test)
    n_data_points_train = len(X_train)
    n_data_points_test = len(X_test)
    n_pois_total = len(locs_poi)
    n_pois_used = int(typeIndicator.sum())

    dataset_information = {
        "n_pois_used": n_pois_used,
        "n_data_points_train": n_data_points_train,
        "n_data_points_test": n_data_points_test,
        "n_data_points": n_data_points,
        "n_pois_total": n_pois_total,
    }

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.dataset_information_json_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        json.dump(
            obj=dataset_information,
            fp=f,
            indent=4,
        )

    latex_string = ""
    latex_string += random.choice(
        [
            f"A total of  $N =$ {n_data_points} data points were used. ",
            f"There were $N =$ {n_data_points} data points included in the analysis. ",
            f"The dataset contains $N =$ {n_data_points} data points. ",
        ]
    )
    if n_pois_total == n_pois_used:
        latex_string += random.choice(
            [
                f"All {n_pois_used} scraped POIs were included in the model.",
                f"Additionally, the {n_pois_used} scraped POIs were included.",
                f"The full set of {n_pois_used} scraped POIs were used.",
            ]
        )
    elif n_pois_used < n_pois_total:
        latex_string += random.choice(
            [
                f"Of the {n_pois_total} POIs scraped, {n_pois_used} POIs were included in the model.",
                f"After selecting which POI types to use, {n_pois_used} of {n_pois_total} scraped POIs were considered.",
                f"After scraping {n_pois_total} POIs, {n_pois_used} belonged to the selected types.",
            ]
        )
    else:
        raise Exception(f"More POIs used than scraped in project: {project}")

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.dataset_information_latex_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        f.write(latex_string)


def create_all_svgp_model_plots(projects):
    for project in projects:
        logger.info(f"{project} | SVGP plots.")
        project_svgp_model_names = get_saved_model_names(
            project=project,
            model_base_name_filter="svgp",
        )
        for model_name in project_svgp_model_names:
            create_project_svgp_model_plots(
                project=project,
                model_name=model_name,
            )


def create_project_svgp_model_plots(project: str, model_name: str):
    logger.info(f"{project} | {model_name} | SVGP plots.")

    create_svgp_training_monitor_plot(
        project=project,
        model_name=model_name,
    )

    create_svgp_decay_plot(
        project=project,
        model_name=model_name,
    )

    create_svgp_latent_poi_influence_plots(
        project=project,
        model_name=model_name,
    )

    create_svgp_spatial_variation_plot(
        project=project,
        model_name=model_name,
    )


def create_svgp_training_monitor_plot(project: str, model_name: str):
    logger.info(f"{project} | {model_name} | SVGP training monitor plot.")

    effects_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.effects_monitor_filename,
            mkdir=False,
        )
    )
    likelihood_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.likelihood_monitor_filename,
            mkdir=False,
        )
    )
    matern32_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.matern32_monitor_filename,
            mkdir=False,
        )
    )
    model_evaluation_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.model_evaluation_monitor_filename,
            mkdir=False,
        )
    )

    if poi.paths.project_input_file(
        project=project,
        filename="synthetic_dataset_information_dict.json",
        mkdir=False,
    ).exists():
        with open(
            poi.paths.project_input_file(
                project=project,
                filename="synthetic_dataset_information_dict.json",
                mkdir=False,
            ),
            "r",
        ) as f:
            synthetic_dataset_information_dict = json.load(
                fp=f,
            )
        synthetic_kernel_params = synthetic_dataset_information_dict["kernel_params"]
    else:
        synthetic_kernel_params = None

    fig = svgp_training_monitor_plot(
        effects_monitor_df=effects_monitor_df,
        likelihood_monitor_df=likelihood_monitor_df,
        matern32_monitor_df=matern32_monitor_df,
        model_evaluation_monitor_df=model_evaluation_monitor_df,
        synthetic_kernel_params=synthetic_kernel_params,
    )

    fig.savefig(
        fname=poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.training_monitor_plot_filename,
            mkdir=True,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_svgp_decay_plot(project: str, model_name: str):
    logger.info(f"{project} | {model_name} | SVGP decay plot.")

    with open(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.information_filename,
            mkdir=True,
        ),
        "r",
    ) as f:
        information_dict = json.load(fp=f)

    effects_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.effects_monitor_filename,
            mkdir=False,
        )
    )

    final_effects = effects_monitor_df[
        effects_monitor_df["iteration"] == effects_monitor_df["iteration"].max()
    ]

    poi_types = final_effects["poi_type"].values
    distmax_values = final_effects["Cut_off"].values

    fig = decay_plot(
        poi_types=poi_types,
        distmax_values=distmax_values,
        kernel_type=information_dict["settings"]["poi_kernel_type"],
    )

    fig.savefig(
        fname=poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.decay_plot_filename,
            mkdir=True,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_svgp_latent_poi_influence_plots(project: str, model_name: str):
    logger.info(f"{project} | {model_name} | SVGP latent poi influence plots.")

    effects_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.effects_monitor_filename,
            mkdir=False,
        )
    )

    final_effects = effects_monitor_df[
        effects_monitor_df["iteration"] == effects_monitor_df["iteration"].max()
    ]

    q_final_loaded = np.load(
        file=poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.q_final_filename,
            mkdir=True,
        )
    )
    q_mu = q_final_loaded["q_mu"]
    q_sqrt = q_final_loaded["q_sqrt"]

    project_data_pipeline_module = importlib.import_module(
        f"poi.{project}.data_pipeline"
    )
    project_data_pipeline = getattr(
        project_data_pipeline_module, "project_data_pipeline"
    )

    with open(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.information_filename,
            mkdir=True,
        ),
        "r",
    ) as f:
        information_dict = json.load(fp=f)

    project_data_pipeline_kwargs = information_dict["settings"][
        "project_data_pipeline_kwargs"
    ]

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

    if poi.paths.project_shape_file(
        project=project,
        mkdir=False,
    ).exists():
        shapes_reader = shapefile.Reader(
            str(
                poi.paths.project_shape_file(
                    project=project,
                    mkdir=False,
                )
            )
        )
    else:
        shapes_reader = None

    for typIdx, poi_type in enumerate(final_effects["poi_type"].values):
        final_effects_poi_type = final_effects[
            final_effects["poi_type"] == poi_type
        ].squeeze()

        distmax = final_effects_poi_type["Cut_off"]
        effect = final_effects_poi_type["Effect_var"]

        # typIdx = poi_types.index(poi_type)

        locs_poi_of_type = locs_poi[
            typeIndicator[:, typIdx] == 1, :
        ]  # select pois of type typIdx

        coords_poi_of_type = coords_poi[typeIndicator[:, typIdx] == 1, :]

        if len(X_test) <= 200:
            use_full_dataset = True
        else:
            use_full_dataset = False

        g = poi.svgp.individual_poi_posterior(
            locs_poi_j=locs_poi_of_type,
            distmax=distmax,
            effect=effect,
            # X=X_test[:, 0:2],  # only n*2 shaped so only coordinates
            X=np.concatenate((X_train, X_test))[:, 0:2] if use_full_dataset else X_test[:, 0:2],  # only n*2 shaped so only coordinates
            Z=Z[:, 0:2],  # only n*2 shaped so only coordinates
            m_g=q_mu[:, typIdx],
            S_g=q_sqrt[typIdx, :, :],
        )

        fig = latent_poi_influence_plot(
            g=g,
            # coords_test=coords_test,
            coords_test=np.concatenate((coords_train, coords_test)) if use_full_dataset else coords_test,
            coords_poi_j=coords_poi_of_type,
            poi_type=poi_type,
            shapes_reader=shapes_reader,
        )

        fig.savefig(
            fname=poi.paths.project_output_model_file(
                project=project,
                model=model_name,
                filename=poi.latent_poi_influence_plot_filename(
                    poi_type=poi_type.replace(" ", "_")
                ),
                mkdir=True,
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


def create_svgp_spatial_variation_plot(project: str, model_name: str):
    logger.info(f"{project} | {model_name} | SVGP spatial variation plot.")

    q_final_loaded = np.load(
        file=poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.q_final_filename,
            mkdir=True,
        )
    )
    q_mu = q_final_loaded["q_mu"]
    q_sqrt = q_final_loaded["q_sqrt"]

    project_data_pipeline_module = importlib.import_module(
        f"poi.{project}.data_pipeline"
    )
    project_data_pipeline = getattr(
        project_data_pipeline_module, "project_data_pipeline"
    )

    with open(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.information_filename,
            mkdir=True,
        ),
        "r",
    ) as f:
        information_dict = json.load(fp=f)

    project_data_pipeline_kwargs = information_dict["settings"][
        "project_data_pipeline_kwargs"
    ]

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

    out_kern_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.model_kern_filename,
            mkdir=False,
        ),
        index_col=0,
    )

    matern32_kernel_index = typeIndicator.shape[
        1
    ]  # =number of poi types, poi kernels start with index 0
    lengthscales = float(
        out_kern_df.loc[
            f"{model_name}/kern/kernels/{matern32_kernel_index}/lengthscales"
        ]["value"]
    )
    variance = float(
        out_kern_df.loc[f"{model_name}/kern/kernels/{matern32_kernel_index}/variance"][
            "value"
        ]
    )

    if len(X_test) <= 200:
        use_full_dataset = True
    else:
        use_full_dataset = False

    g = poi.svgp.predict_spatial(
        # X=X_test[:, 0:2],  # only n*2 shaped so only coordinates
        X=np.concatenate((X_train, X_test))[:, 0:2] if use_full_dataset else X_test[:, 0:2],
        # only n*2 shaped so only coordinates
        Z=Z[:, 0:2],  # only n*2 shaped so only coordinates
        lengthscale=lengthscales,
        variance=variance,
        m_g=q_mu[:, matern32_kernel_index],
    )

    if poi.paths.project_shape_file(
        project=project,
        mkdir=False,
    ).exists():
        shapes_reader = shapefile.Reader(
            str(
                poi.paths.project_shape_file(
                    project=project,
                    mkdir=False,
                )
            )
        )
    else:
        shapes_reader = None

    fig = spatial_variation_plot(
        g=g,
        # coords_test=coords_test,
        coords_test=np.concatenate((coords_train, coords_test)) if use_full_dataset else coords_test,
        shapes_reader=shapes_reader,
    )

    fig.savefig(
        fname=poi.paths.project_output_model_file(
            project=project,
            model=model_name,
            filename=poi.spatial_variation_plot_filename,
            mkdir=True,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_best_svgp_directory(project: str):
    logger.info(f"{project} | Best SVGP directory creation.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter="svgp",
    )

    project_information_dict = projects_information_dicts[project]

    min_rmse = 1000
    min_rmse_model_name = None
    min_rmse_model_information_dict = None

    for model_base_name, model_information_dicts in project_information_dict.items():
        for information_dict in model_information_dicts:
            if information_dict["model_evaluation"]["RMSE_OOS"] < min_rmse:
                min_rmse = information_dict["model_evaluation"]["RMSE_OOS"]
                min_rmse_model_name = information_dict["model_name"]
                min_rmse_model_information_dict = information_dict

    logger.info(
        f"{project} | Using model {min_rmse_model_name} for best svgp directory."
    )

    if poi.paths.project_output_model_dir(
        project=project,
        model=poi.best_svgp_name,
    ).exists():
        shutil.rmtree(
            path=poi.paths.project_output_model_dir(
                project=project,
                model=poi.best_svgp_name,
            )
        )

    shutil.copytree(
        src=poi.paths.project_output_model_dir(
            project=project,
            model=min_rmse_model_name,
        ),
        dst=poi.paths.project_output_model_dir(
            project=project,
            model=poi.best_svgp_name,
        ),
    )


def create_project_baselines_table(project: str, cpu_threads_selection: int):
    logger.info(f"{project} | Baselines table.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter=poi.baseline_names,
    )

    project_information_dict = projects_information_dicts[project]

    rows = []

    for model_base_name, model_information_dicts in project_information_dict.items():
        for information_dict in model_information_dicts:
            if (
                information_dict["system_resources"]["os_sched_affinity"]
                != cpu_threads_selection
            ):
                continue
            if information_dict["settings"]["project_data_pipeline_kwargs"][
                "feature_type"
            ] not in [None, "min_dist", "count", "both"]:
                continue
            d = {}
            d["Model"] = information_dict["model_base_name"]
            feature_type = information_dict["settings"]["project_data_pipeline_kwargs"][
                "feature_type"
            ]
            if feature_type is None:
                d["Feature engineering"] = "None"
            else:
                d["Feature engineering"] = feature_type.replace("_", "\_") + " " + str(information_dict["settings"]["project_data_pipeline_kwargs"].get("count_feature_d_max"))
            # d["Feature engineering"] = information_dict["settings"][
            #     "project_data_pipeline_kwargs"
            # ]["feature_type"].replace("_", "\_")
            # model = information_dict["model_base_name"]
            # feature_engineering = information_dict["settings"][
            #     "project_data_pipeline_kwargs"
            # ]["feature_type"]
            # d["Model [feature engineering]"] = f"{model} [{feature_engineering}]"
            d["RMSE OOS"] = information_dict["model_evaluation"]["RMSE_OOS"]
            d["Likelihood OOS"] = information_dict["model_evaluation"]["TOTAL_LIK_OOS"]
            d["Training time"] = str(
                dt.timedelta(
                    seconds=round(information_dict["model_pure_training_time"], 0)
                )
            )
            d["Prediction time"] = information_dict["model_oos_prediction_time"]
            rows.append(d)

    table = ""
    table += r"\begin{table}[htb]" + "\n"
    table += r"\centering" + "\n"
    table += r"\begin{tabular}" + "{llSScS}" + "\n"
    table += r"\toprule" + "\n"
    table += (
        r"{Model} & {Feature engineering} & {RMSE OOS} & {Likelihood OOS} & {Training time} & {Prediction time} \\"
        + "\n"
    )
    table += r"\midrule" + "\n"

    for d in rows:
        table += d["Model"]
        table += r" & "
        table += d["Feature engineering"]
        table += r" & "
        table += f"{d['RMSE OOS']:.3f}"
        table += r" & "
        table += f"{d['Likelihood OOS']:.3f}"
        table += r" & "
        table += f"{d['Training time']}"
        table += r" & "
        table += f"{d['Prediction time']:.3f}"
        table += r" \\" + "\n"

    table += r"\bottomrule" + "\n"
    table += r"\end{tabular}" + "\n"
    table += (
        r"\caption{Results of multiple baseline models. All of them were run with 1 CPU core.}"
        + "\n"
    )
    table += (
        r"\label{tbl:results_baseline_" + project.replace("_", " ").title() + "}" + "\n"
    )
    table += r"\end{table}" + "\n"

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.baselines_table_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        f.write(table)

    # df = pd.DataFrame(rows)
    #
    # save_df_as_latex(
    #     df=df,
    #     path=poi.paths.project_output_file(
    #         project=project,
    #         filename=poi.baselines_table_filename,
    #         mkdir=True,
    #     ),
    #     index=False,
    #     column_format="l" * len(df.columns),
    #     caption=f"Results of multiple baseline models. All of them were run with {cpu_threads_selection // 2} CPU core{'s' if (cpu_threads_selection // 2) > 1 else ''}.",
    #     label=f"baseline_results_{project}",
    # )


def create_project_svgp_table(project: str, cpu_threads_selection: int):
    logger.info(f"{project} | SVGP table.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter="svgp",
    )

    project_information_dict = projects_information_dicts[project]

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.dataset_information_json_filename,
            mkdir=False,
        ),
        "r",
    ) as f:
        dataset_information = json.load(fp=f)

    rows = []

    for model_base_name, model_information_dicts in project_information_dict.items():
        for information_dict in model_information_dicts:
            if (
                information_dict["system_resources"]["os_sched_affinity"]
                != cpu_threads_selection
            ):
                continue
            d = {}
            d["Mean Function"] = (
                information_dict["settings"]["mean_fct_class"]
                .replace("SlicedNNwithbias", "Neural network")
                .replace("SlicedNN", "Neural network")
                .replace("SlicedLinear", "Linear")
            )
            d["Kernel"] = information_dict["settings"]["poi_kernel_type"].replace("Linear", "ReLU")
            # d["Likelihood"] = information_dict["settings"]["likelihood_class"]
            knot_points = information_dict["settings"]["project_data_pipeline_kwargs"][
                "knot_points"
            ]
            # d["Knot Points"] = knot_points
            if knot_points > 1:
                n_knot_points = knot_points
                fraction_knot_points = (
                    n_knot_points / dataset_information["n_data_points_train"]
                )
            else:
                fraction_knot_points = knot_points
                n_knot_points = int(
                    fraction_knot_points * dataset_information["n_data_points_train"]
                )
            d["Knot Points"] = f"{fraction_knot_points:.2f} ({n_knot_points})"
            d["RMSE OOS"] = information_dict["model_evaluation"]["RMSE_OOS"]
            d["Likelihood OOS"] = information_dict["model_evaluation"]["TOTAL_LIK_OOS"]
            d["Training time"] = str(
                dt.timedelta(
                    seconds=round(information_dict["model_pure_training_time"], 0)
                )
            )
            d["Prediction time"] = information_dict["model_oos_prediction_time"]
            # print(json.dumps(obj=information_dict, cls=poi.helpers.CustomEncoder, indent=4))

            rows.append(d)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["RMSE OOS", "Likelihood OOS"])

    table = ""
    table += r"\begin{table}[htbp]" + "\n"
    table += r"\centering" + "\n"
    table += r"\scriptsize" + "\n"
    table += r"\setlength\tabcolsep{2pt}" + "\n"
    table += r"\begin{tabular}" + "{lllSScS}" + "\n"
    table += r"\toprule" + "\n"

    if project == "chargingpoints":
        table += (
            r"{Charger influence $\bm{g}$} & {Kernel $k_\gamma$} & {RMSE} & {Log-lik.}\\"
            + "\n"
        )
    else:
        table += (
            r"{Mean Function} & {Kernel} & {Knot Points} & {RMSE} & {Log-lik.} & {Training time} & {Prediction time}\\"
            + "\n"
        )
    table += r"\midrule" + "\n"

    for _, row in df.iterrows():
        table += row["Mean Function"]
        table += r" & "
        table += row["Kernel"]
        if project != "chargingpoints":
            table += r" & "
            table += f"{row['Knot Points']}"
        table += r" & "
        table += f"{row['RMSE OOS']:.3f}"
        table += r" & "
        table += f"{row['Likelihood OOS']:.3f}"
        if project != "chargingpoints":
            table += r" & "
            table += f"{row['Training time']}"
            table += r" & "
            table += f"{row['Prediction time']:.3f}"
        table += r" \\" + "\n"

    table += r"\bottomrule" + "\n"
    table += r"\end{tabular}" + "\n"
    # table += (
    #     r"\caption{Ouf of sample performance of different svgp models for project "
    #     + project.replace("_", " ").title()
    #     + ".}"
    #     + "\n"
    # )
    table += (
        r"\caption{Sensitivity analysis comparing the ouf-of-sample performance across different model specifications.}"
        + "\n"
    )
    table += r"\label{tbl:sensitivity_analysis_" + project + "}" + "\n"
    if project == "chargingpoints":
        table += r"\vspace{-0.5cm}" + "\n"
    table += r"\end{table}" + "\n"

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.svgp_table_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        f.write(table)

    # save_df_as_latex(
    #     df=df,
    #     path=poi.paths.project_output_file(
    #         project=project,
    #         filename=poi.svgp_table_filename,
    #         mkdir=True,
    #     ),
    #     index=False,
    #     column_format="l" * len(df.columns),
    #     caption=f"Results of svgp models.",
    #     label=f"svgp_results_{project}",
    # )


def create_cpu_threads_vs_training_time_plot(
    projects: List[str],
    model_base_name_filter: Optional[Union[str, Tuple[str, ...]]],
    filename: str,
):
    projects_information_dicts = get_projects_information_dicts(
        projects=projects,
        model_base_name_filter=model_base_name_filter,
    )
    # TODO(critical): take care of possible different amount of iterations and knot points in svgp models

    fig = cpu_threads_vs_training_time_plot(
        projects_information_dicts=projects_information_dicts
    )

    fig.savefig(
        fname=poi.paths.general_output_file(
            filename=filename,
            mkdir=True,
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_project_effects_table(project: str):
    logger.info(f"{project} | SVGP effects table.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter="svgp",
    )

    project_information_dict = projects_information_dicts[project]

    min_rmse = 1000
    min_rmse_model_name = None
    min_rmse_model_information_dict = None

    for model_base_name, model_information_dicts in project_information_dict.items():
        for information_dict in model_information_dicts:
            if information_dict["model_evaluation"]["RMSE_OOS"] < min_rmse:
                min_rmse = information_dict["model_evaluation"]["RMSE_OOS"]
                min_rmse_model_name = information_dict["model_name"]
                min_rmse_model_information_dict = information_dict

    logger.info(f"{project} | Using model {min_rmse_model_name} for effects table.")
    effects_monitor_df = pd.read_csv(
        poi.paths.project_output_model_file(
            project=project,
            model=min_rmse_model_name,
            filename=poi.effects_monitor_filename,
            mkdir=False,
        )
    )

    final_effects = effects_monitor_df[
        effects_monitor_df["iteration"] == effects_monitor_df["iteration"].max()
    ]

    table = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength\tabcolsep{2pt}
\begin{tabular}{l SSS}
\toprule
&\multicolumn{1}{c}{Distance}&\multicolumn{2}{c}{Magnitude}\\
\cmidrule(lr){2-2}
\cmidrule(lr){3-4}
POI type $(\gamma)$ & {Cut-off $\theta_{\gamma}$ [in km]} & {Average effect $\bar{\alpha}_\gamma$} & {SD}\\		
\midrule
"""

    for _, row in final_effects.iterrows():
        table += f"""{row["poi_type"].replace("_", " ").capitalize()} & {row["Cut_off"]:.3f} & {row["Effect_size_abs"]:.3f} & {row["Effect_size_abs_std"]:.3f}"""
        table += r"\\"
        table += "\n"

    table += r"""\bottomrule
\multicolumn{1}{l}{SD: standard deviation}
\end{tabular}
"""
    # caption = (
    #     r"{Estimated parameters of the POI influence for the project "
    #     + project.replace("_", " ").title()
    #     + r".}"
    # )
    caption = r"{Estimated parameters of the POI influence.}"
    table += f"""\caption{caption}""" + "\n"

    label = "{" + f"tbl:effects_{project}" + "}"
    table += f"""\label{label}""" + "\n"

    if project == "chargingpoints":
        table += r"\vspace{-0.5cm}" + "\n"

    table += r"""\end{table}""" + "\n"

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.effects_table_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        f.write(table)


def create_performance_overview_table(
    projects: List[str],
    baselines_cpu_threads_selection: int,
    svgp_cpu_threads_selection: int,
    baseline_feature_types = (None, "min_dist", "count", "both"),
):
    if len(projects) == 1:
        logger.info(f"{projects[0]} | Single project performance overview table.")
    else:
        logger.info(
            f"{'_'.join(projects)} | Multiple projects performance overview table."
        )
    baselines_data = {}
    svgp_data = {}

    projects_information_dicts = get_projects_information_dicts(
        projects=projects,
    )

    for project, project_information_dict in projects_information_dicts.items():
        for (
            model_base_name,
            model_information_dicts,
        ) in project_information_dict.items():
            if model_base_name in poi.baseline_names:
                for information_dict in model_information_dicts:
                    if (
                        information_dict["system_resources"]["os_sched_affinity"]
                        != baselines_cpu_threads_selection
                    ):
                        continue
                    if information_dict["settings"]["project_data_pipeline_kwargs"][
                        "feature_type"
                    ] not in baseline_feature_types:
                        continue
                    model = information_dict["model_base_name"]
                    feature_engineering = str(
                        information_dict["settings"]["project_data_pipeline_kwargs"][
                            "feature_type"
                        ]
                    ).replace("_", " ").replace("None", "none")
                    model_display_name = f"{model} [{feature_engineering}]"
                    data_to_add = {
                        "RMSE_OOS": information_dict["model_evaluation"]["RMSE_OOS"],
                        "TOTAL_LIK_OOS": information_dict["model_evaluation"][
                            "TOTAL_LIK_OOS"
                        ],
                    }
                    if model_display_name not in baselines_data:
                        baselines_data[model_display_name] = {}
                    if project not in baselines_data[model_display_name]:
                        baselines_data[model_display_name][project] = data_to_add
                    else:
                        if data_to_add["RMSE_OOS"] < baselines_data[model_display_name][project]["RMSE_OOS"]:
                            baselines_data[model_display_name][project] = data_to_add

            elif model_base_name == "svgp":
                min_rmse = 1000
                min_rmse_model_information_dict = None
                for information_dict in model_information_dicts:
                    if (
                        information_dict["system_resources"]["os_sched_affinity"]
                        != svgp_cpu_threads_selection
                    ):
                        continue
                    if information_dict["model_evaluation"]["RMSE_OOS"] < min_rmse:
                        min_rmse = information_dict["model_evaluation"]["RMSE_OOS"]
                        min_rmse_model_information_dict = information_dict
                # model_display_name = "SVGP [n/a]"
                model_display_name = "POI model (ours)"
                data_to_add = {
                    "RMSE_OOS": min_rmse_model_information_dict["model_evaluation"][
                        "RMSE_OOS"
                    ],
                    "TOTAL_LIK_OOS": min_rmse_model_information_dict[
                        "model_evaluation"
                    ]["TOTAL_LIK_OOS"],
                }
                if model_display_name not in svgp_data:
                    svgp_data[model_display_name] = {}
                svgp_data[model_display_name][project] = data_to_add
            else:
                raise Exception(f"Found unexpected model base name: {model_base_name}.")

    best_models = {}
    for project in projects:
        best_models[project] = {}
        best_models[project]["RMSE_OOS"] = {
            "model_display_name": None,
            "best_value": 1000,
        }
        best_models[project]["TOTAL_LIK_OOS"] = {
            "model_display_name": None,
            "best_value": -1000000,
        }
    for model_display_name, model_project_results in {
        **baselines_data,
        **svgp_data,
    }.items():
        for project, results in model_project_results.items():
            if results["RMSE_OOS"] < best_models[project]["RMSE_OOS"]["best_value"]:
                best_models[project]["RMSE_OOS"][
                    "model_display_name"
                ] = model_display_name
                best_models[project]["RMSE_OOS"]["best_value"] = results["RMSE_OOS"]
            if (
                results["TOTAL_LIK_OOS"]
                > best_models[project]["TOTAL_LIK_OOS"]["best_value"]
            ):
                best_models[project]["TOTAL_LIK_OOS"][
                    "model_display_name"
                ] = model_display_name
                best_models[project]["TOTAL_LIK_OOS"]["best_value"] = results[
                    "TOTAL_LIK_OOS"
                ]

    def baselines_sorter(x):
        model_display_name = x[0]
        model_name, feature_name = model_display_name.split(" [")
        feature_name = feature_name.replace("]", "")

        assert model_name in poi.baseline_names, f"Unexpected model_name {model_name}"

        sort_key = poi.baseline_names.index(model_name) * len(baseline_feature_types)
        sort_key += tuple((str(i).replace("_", " ").replace("None", "none") for i in baseline_feature_types)).index(feature_name)

        return sort_key

    table = ""
    table += r"\begin{table}[htb]" + "\n"
    table += r"\centering" + "\n"
    table += r"\scriptsize" + "\n"
    table += r"\setlength\tabcolsep{2pt}" + "\n"
    table += r"\begin{tabular}" + "{l" + "SS" * len(projects) + "}" + "\n"
    table += r"\toprule" + "\n"
    if len(projects) > 1:
        for project in projects:
            table += (
                r"&\multicolumn{2}{c}{"
                + project.replace("houseprices_", "").replace("_", " ").title()
                + "}"
            )
        table += r"\\" + "\n"
        i = 2
        for _ in range(len(projects)):
            table += r"\cmidrule(lr){" + f"{i}-{i+1}" + "}"
            i += 2
        table += "\n"
    table += "Model [feature engineering]"
    for _ in projects:
        table += r"& \multicolumn{1}{c}{RMSE} & \multicolumn{1}{c}{Log-lik.}"
    table += r"\\" + "\n"
    table += r"\midrule" + "\n"
    # baseline models
    for model_display_name, model_project_results in dict(sorted(baselines_data.items(), key=baselines_sorter)).items():
        table += model_display_name
        for project, results in model_project_results.items():
            if (
                best_models[project]["RMSE_OOS"]["model_display_name"]
                == model_display_name
            ):
                table += " &" + r"\bfseries" + f" {results['RMSE_OOS']:.3f} "
            else:
                table += f" &{results['RMSE_OOS']:.3f} "
            if (
                best_models[project]["TOTAL_LIK_OOS"]["model_display_name"]
                == model_display_name
            ):
                table += "& " + r"\bfseries" + f" {results['TOTAL_LIK_OOS']:.3f} "
            else:
                table += f"& {results['TOTAL_LIK_OOS']:.3f} "
            # table += f" &{results['RMSE_OOS']:.3f} & {results['TOTAL_LIK_OOS']:.3f} "
        table += r"\\" + "\n"
    table += r"\midrule" + "\n"
    # svgp model
    for model_display_name, model_project_results in svgp_data.items():
        table += model_display_name
        for project, results in model_project_results.items():
            if (
                best_models[project]["RMSE_OOS"]["model_display_name"]
                == model_display_name
            ):
                table += " &" + r"\bfseries" + f" {results['RMSE_OOS']:.3f} "
            else:
                table += f" &{results['RMSE_OOS']:.3f} "
            if (
                best_models[project]["TOTAL_LIK_OOS"]["model_display_name"]
                == model_display_name
            ):
                table += "& " + r"\bfseries" + f" {results['TOTAL_LIK_OOS']:.3f} "
            else:
                table += f"& {results['TOTAL_LIK_OOS']:.3f} "
            # table += f" &{results['RMSE_OOS']:.3f} & {results['TOTAL_LIK_OOS']:.3f} "
        table += r"\\" + "\n"
    table += r"\bottomrule" + "\n"
    table += r"\multicolumn{3}{l}{Note: Best value per column is in bold.}" + "\n"
    table += r"\end{tabular}" + "\n"
    # if len(projects) == 1:
    #     table += (
    #         r"\caption{Out-of-sample performance of baselines and POI model for project "
    #         + project.replace("_", " ").title()
    #         + r".}"
    #         + "\n"
    #     )
    # else:
    #     table += (
    #         r"\caption{Out-of-sample performance of baselines and POI model for projects "
    #         + ", ".join(
    #             [project.replace("_", " ").title() for project in projects[:-1]]
    #         )
    #         + " and "
    #         + projects[-1].replace("_", " ").title()
    #         + r".}"
    #         + "\n"
    #     )
    table += (
        r"\caption{Out-of-sample performance of the different models.}"
        + "\n"
    )
    table += r"\label{tbl:results_" + f"_".join(projects) + "}" + "\n"

    if len(projects) == 1 and projects[0] == "chargingpoints":
        table += r"\vspace{-0.5cm}" + "\n"

    table += r"\end{table}" + "\n"

    if baseline_feature_types == (None, "both"):
        filename_appendix = "_www"
    else:
        filename_appendix = ""


    if len(projects) == 1:  # performance table for a single project
        with open(
            poi.paths.project_output_file(
                project=projects[0],
                filename=poi.performance_overview_table_filename(
                    project=projects[0],
                    appendix=filename_appendix,
                ),
                mkdir=True,
            ),
            "w",
        ) as f:
            f.write(table)
    else:  # performance table for multiple projects
        with open(
            poi.paths.general_output_file(
                filename=poi.performance_overview_table_filename(
                    project=f"_".join(projects),
                    appendix=filename_appendix,
                ),
                mkdir=True,
            ),
            "w",
        ) as f:
            f.write(table)


def create_latex_figure_latent_poi_influence_spatial_variation(project: str):
    logger.info(f"{project} | SVGP latent poi influence and spatial variation figure.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter="svgp",
    )

    project_information_dict = projects_information_dicts[project]

    for model_base_name, model_information_dicts in project_information_dict.items():
        for model_information_dict in model_information_dicts:
            model_name = model_information_dict["model_name"]

            effects_monitor_df = pd.read_csv(
                poi.paths.project_output_model_file(
                    project=project,
                    model=model_name,
                    filename=poi.effects_monitor_filename,
                    mkdir=False,
                )
            )

            for poi_type in effects_monitor_df["poi_type"].unique():
                figure = ""
                figure += r"\begin{figure}" + "\n"
                figure += r"\begin{subfigure}{.5\textwidth}" + "\n"
                figure += r"\centering" + "\n"
                figure += (
                    r"\includegraphics[width=\linewidth]{../data/"
                    + project
                    + r"/output/models/"
                    + model_name
                    + "/latent_poi_influence_"
                    + poi_type.replace(" ", "_")
                    + ".pdf}"
                    + "\n"
                )
                figure += r"\end{subfigure}" + "\n"
                figure += r"\begin{subfigure}{.5\textwidth}" + "\n"
                figure += r"\centering" + "\n"
                figure += (
                    r"\includegraphics[width=\linewidth]{../data/"
                    + project
                    + r"/output/models/"
                    + model_name
                    + "/spatial_variation.pdf}"
                    + "\n"
                )
                figure += r"\end{subfigure}" + "\n"
                figure += (
                    r"\caption{" + project.replace("_", " ").title() + r": \emph{Left:} Recovered latent POI influence $\bm{h}_\gamma$ for POI type \textquote{"
                    + poi_type.lower()
                    + "}."
                    + r" \emph{Right:} Remaining spatial variation $\bm{h}_0$. "
                    + r"Note the different scales of the colorbar.}"
                    + "\n"
                )
                figure += (
                    r"\label{fig:"
                    + project
                    + "_"
                    + poi_type.replace(" ", "_").lower()
                    + "_latent_poi_influence_spatial_variation}"
                    + "\n"
                )
                figure += "\end{figure}"

                with open(
                    poi.paths.project_output_model_file(
                        project=project,
                        model=model_name,
                        filename=poi.latent_poi_influence_spatial_variation_figure(
                            poi_type=poi_type.replace(" ", "_").lower()
                        ),
                        mkdir=True,
                    ),
                    "w",
                ) as f:
                    f.write(figure)


def create_project_appendix(project: str):
    logger.info(f"{project} | Appendix.")

    projects_information_dicts = get_projects_information_dicts(
        projects=project,
        model_base_name_filter="svgp",
    )

    project_information_dict = projects_information_dicts[project]

    poi_types_from_latent_influence_plots = []
    for path in sorted(
        poi.paths.project_output_model_dir(
            project=project,
            model=poi.best_svgp_name,
        ).glob(
            "*".join(
                poi.latent_poi_influence_plot_filename("NOTHINGTOSEEHERE").split(
                    "NOTHINGTOSEEHERE"
                )
            )
        )
    ):
        poi_type = path.stem.replace(
            poi.latent_poi_influence_plot_filename("NOTHINGTOSEEHERE").split(
                "NOTHINGTOSEEHERE"
            )[0],
            "",
        )
        poi_types_from_latent_influence_plots.append(poi_type)

    latex = ""

    latex += r"\section{" + project.replace("_", " ").title() + r"}" + "\n"

    latex += r"\input{../data/" + project + r"/output/svgp_table.tex}" + "\n"

    # latex += r"\FloatBarrier" + "\n"

    latex += (
        r"\input{../data/"
        + project
        + r"/output/performance_overview_table_"
        + project
        + r".tex}"
        + "\n"
    )

    latex += r"\input{../data/" + project + r"/output/effects_table.tex}" + "\n"

    latex += r"\begin{figure}[htb]" + "\n"
    latex += r"\begin{subfigure}{\textwidth}" + "\n"
    latex += r"\centering" + "\n"
    latex += (
        r"\includegraphics[width=0.7\linewidth]{../data/"
        + project
        + r"/output/models/"
        + poi.best_svgp_name
        + r"/"
        + poi.decay_plot_filename
        + r"}"
        + "\n"
    )
    latex += r"\end{subfigure}" + "\n"
    latex += (
        r"\caption{Kernel decay plot for the best performing POI model of the dataset "
        + project.replace("_", " ").title()
        + r".}"
        + "\n"
    )
    latex += r"\label{fig:appendix_" + project + r"_kernel_decay_plot}" + "\n"
    latex += r"\end{figure}" + "\n"

    latex += r"\FloatBarrier" + "\n"

    if len(poi_types_from_latent_influence_plots) + 1 <= 6:
        textwidth_fraction = "0.5"
    elif len(poi_types_from_latent_influence_plots) + 1 <= 9:
        textwidth_fraction = "0.3"
    else:
        raise Exception("More than 9 poi types. Not tested.")

    if len(poi_types_from_latent_influence_plots) > 0:
        latex += r"\begin{figure}[htb]" + "\n"

        for poi_type in poi_types_from_latent_influence_plots:
            latex += (
                r"\begin{subfigure}[t]{" + textwidth_fraction + r"\textwidth}" + "\n"
            )
            latex += (
                r"\includegraphics[width=\linewidth]{../data/"
                + project
                + r"/output/models/"
                + poi.best_svgp_name
                + r"/latent_poi_influence_"
                + poi_type
                + r".pdf}"
                + "\n"
            )
            latex += r"\caption{" + poi_type.replace("_", " ") + r"}" + "\n"
            latex += (
                r"\label{fig:appendix_"
                + project
                + r"_latent_poi_influence_plots_"
                + poi_type
                + r"}"
                + "\n"
            )
            latex += r"\end{subfigure}" + "\n"

        latex += r"\begin{subfigure}[t]{" + textwidth_fraction + r"\textwidth}" + "\n"
        latex += (
            r"\includegraphics[width=\linewidth]{../data/"
            + project
            + r"/output/models/"
            + poi.best_svgp_name
            + r"/spatial_variation.pdf}"
            + "\n"
        )
        latex += r"\caption{Spatial variation $\bm{h}_0$}" + "\n"
        latex += r"\label{fig:appendix_" + project + r"_spatial_variation_plot}" + "\n"
        latex += r"\end{subfigure}" + "\n"

        latex += (
            r"\caption{Recovered latent POI influences $\bm{h}_\gamma$ for each POI type and remaining spatial variation $\bm{h}_0$ for the dataset "
            + project.replace("_", " ").title()
            + r". "
            + r"Note the different scales of the colorbar.}"
            + "\n"
        )
        latex += (
            r"\label{fig:appendix_"
            + project
            + r"_latent_poi_influence_spatial_variation_plots}"
            + "\n"
        )
        latex += r"\end{figure}" + "\n"

    latex += r"\begin{figure}[htb]" + "\n"
    latex += r"\begin{subfigure}{\textwidth}" + "\n"
    latex += r"\centering" + "\n"
    latex += (
        r"\includegraphics[width=\linewidth]{../data/"
        + project
        + r"/output/models/"
        + poi.best_svgp_name
        + r"/"
        + poi.training_monitor_plot_filename
        + r"}"
        + "\n"
    )
    latex += r"\end{subfigure}" + "\n"
    latex += (
        r"\caption{Training monitor for the best performing POI model of the dataset "
        + project.replace("_", " ").title()
        + r".}"
        + "\n"
    )
    latex += r"\label{fig:appendix_" + project + r"_training_monitor_plot}" + "\n"
    latex += r"\end{figure}" + "\n"

    latex += r"\FloatBarrier" + "\n"
    latex += r"\newpage" + "\n"

    with open(
        poi.paths.project_output_file(
            project=project,
            filename=poi.appendix_latex_filename,
            mkdir=True,
        ),
        "w",
    ) as f:
        f.write(latex)
