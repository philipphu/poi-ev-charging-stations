import argparse
import importlib
from typing import List

from poi.project_config import SVGPConfig
from poi.svgp import gp_model_full_run


def run_project_svgp(svgp_config: SVGPConfig):
    model = gp_model_full_run(
        project_data_pipeline=svgp_config.project_data_pipeline,
        project_data_pipeline_kwargs=svgp_config.project_data_pipeline_kwargs,
        mean_fct_class=svgp_config.mean_fct_class,
        mean_fct_class_kwargs=svgp_config.mean_fct_class_kwargs,
        likelihood_class=svgp_config.likelihood_class,
        likelihood_class_kwargs=svgp_config.likelihood_class_kwargs,
        poi_kernel_type=svgp_config.poi_kernel_type,
        evaluation_function=svgp_config.evaluation_function,
        PROJECT_NAME=svgp_config.PROJECT_NAME,
        D=svgp_config.D,
        spatial_kernel_lengthscales=svgp_config.spatial_kernel_lengthscales,
        iterations=svgp_config.iterations,
        logging_iteration=svgp_config.logging_iteration,
        monitor_iteration=svgp_config.monitor_iteration,
        evaluation_iteration=svgp_config.evaluation_iteration,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="name of the project to run svgp for")
    parser.add_argument(
        "config", type=int, help="index of the config in the svgp configs list to run"
    )
    args = parser.parse_args()
    project_name = args.project
    config = args.config
    project_config_module = importlib.import_module(f"poi.{project_name}.config")
    svgp_configs: List[SVGPConfig] = getattr(project_config_module, "svgp_configs")
    svgp_config = svgp_configs[config]

    run_project_svgp(svgp_config=svgp_config)


if __name__ == "__main__":
    main()
