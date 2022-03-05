"""
poi package containing subpackages for each project and general modules to be used in the subpackages.
"""
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 15

from poi.paths_handler import Paths

paths = Paths()

# baseline names
baseline_names = ("GWR", "Linear kriging", "RF kriging", "Neural network")

# filenames

## general training files
information_filename = "information.json"

## svgp training files
likelihood_monitor_filename = "likelihood_monitor.csv"
effects_monitor_filename = "effects_monitor.csv"
Z_monitor_filename = "Z_monitor.pkl"
matern32_monitor_filename = "matern32_monitor.csv"
model_evaluation_monitor_filename = "model_evaluation_monitor.csv"
q_final_filename = "q_final.npz"
model_filename = "model"
model_kern_filename = "kern.csv"
model_likelihood_filename = "likelihood.csv"
model_mean_function_filename = "mean_function.csv"

## general output files
# baselines_cpu_threads_vs_training_time_plot_filename = (
#     "baselines_cpu_threads_vs_training_time.png"
# )
# svgp_cpu_threads_vs_training_time_plot_filename = (
#     "svgp_cpu_threads_vs_training_time.png"
# )
baselines_cpu_threads_vs_training_time_plot_filename = (
    "baselines_cpu_threads_vs_training_time.pdf"
)
svgp_cpu_threads_vs_training_time_plot_filename = (
    "svgp_cpu_threads_vs_training_time.pdf"
)

## project output files
baselines_table_filename = "baselines_table.tex"
svgp_table_filename = "sensitivity_analysis.tex"
effects_table_filename = "effects_table.tex"
dataset_information_json_filename = "dataset_information.json"
dataset_information_latex_filename = "dataset_information.tex"
appendix_latex_filename = "appendix.tex"

## svgp output files
# training_monitor_plot_filename = "training_monitor.png"
# decay_plot_filename = "decay_plot.png"
# spatial_variation_plot_filename = "spatial_variation.png"
# latent_poi_influence_plot_filename = lambda poi_type: f"latent_poi_influence_{poi_type}.png"
training_monitor_plot_filename = "training_monitor.pdf"
decay_plot_filename = "decay_plot.pdf"
spatial_variation_plot_filename = "spatial_variation.pdf"
latent_poi_influence_plot_filename = (
    lambda poi_type: f"latent_poi_influence_{poi_type}.pdf"
)

## latex output files
performance_overview_table_filename = (
    lambda project, appendix: f"performance_overview_table_{project}{appendix}.tex"
)
latent_poi_influence_spatial_variation_figure = (
    lambda poi_type: f"latent_poi_influence_spatial_variation_figure_{poi_type}.tex"
)

## other stuff
best_svgp_name = "best_svgp"
