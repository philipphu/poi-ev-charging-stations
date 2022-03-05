import json
import poi.helpers


class ProjectConfig:
    def __init__(
        self,
        PROJECT_NAME,
        project_data_pipeline,
        evaluation_function,
    ):
        # General
        self.PROJECT_NAME = PROJECT_NAME
        self.project_data_pipeline = project_data_pipeline
        self.evaluation_function = evaluation_function

    def __str__(self):
        return json.dumps(
            obj=self.__dict__,
            cls=poi.helpers.CustomEncoder,
            indent=4,
        )


class BaselinesConfig(ProjectConfig):
    def __init__(
        self,
        PROJECT_NAME,
        project_data_pipeline,
        evaluation_function,
        project_data_pipeline_kwargs_options,
        models,
    ):
        super().__init__(
            PROJECT_NAME=PROJECT_NAME,
            project_data_pipeline=project_data_pipeline,
            evaluation_function=evaluation_function,
        )
        self.project_data_pipeline_kwargs_options = project_data_pipeline_kwargs_options
        self.models = models


class SVGPConfig(ProjectConfig):
    def __init__(
        self,
        PROJECT_NAME,
        project_data_pipeline,
        evaluation_function,
        project_data_pipeline_kwargs,
        mean_fct_class,
        mean_fct_kwargs,
        likelihood_class,
        likelihood_class_kwargs,
        poi_kernel_type,
        D,
        spatial_kernel_lengthscales,
        iterations,
        logging_iteration,
        monitor_iteration,
        evaluation_iteration,
    ):
        super().__init__(
            PROJECT_NAME=PROJECT_NAME,
            project_data_pipeline=project_data_pipeline,
            evaluation_function=evaluation_function,
        )
        self.project_data_pipeline_kwargs = project_data_pipeline_kwargs
        self.mean_fct_class = mean_fct_class
        self.mean_fct_class_kwargs = mean_fct_kwargs
        self.likelihood_class = likelihood_class
        self.likelihood_class_kwargs = likelihood_class_kwargs
        self.poi_kernel_type = poi_kernel_type
        self.D = D
        self.spatial_kernel_lengthscales = spatial_kernel_lengthscales
        self.iterations = iterations
        self.logging_iteration = logging_iteration
        self.monitor_iteration = monitor_iteration
        self.evaluation_iteration = evaluation_iteration
