import argparse
import importlib

from poi.project_config import BaselinesConfig
from poi.baselines import run_all_baselines


def run_project_baselines(baselines_config: BaselinesConfig):
    run_all_baselines(
        project_data_pipeline=baselines_config.project_data_pipeline,
        project_data_pipeline_kwargs_options=baselines_config.project_data_pipeline_kwargs_options,
        models=baselines_config.models,
        evaluation_function=baselines_config.evaluation_function,
        PROJECT_NAME=baselines_config.PROJECT_NAME,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="name of the project to run baselines for")
    args = parser.parse_args()
    project_name = args.project
    project_config_module = importlib.import_module(f"poi.{project_name}.config")
    baselines_config: BaselinesConfig = getattr(
        project_config_module, "baselines_config"
    )
    run_project_baselines(baselines_config=baselines_config)


if __name__ == "__main__":
    main()
