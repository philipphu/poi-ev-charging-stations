import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from poi.helpers import step_size_string


def svgp_training_monitor_plot(
    effects_monitor_df,
    likelihood_monitor_df,
    matern32_monitor_df,
    model_evaluation_monitor_df,
    synthetic_kernel_params=None,
) -> plt.Figure:
    widths = [1]
    heights = [2, 2, 2, 1, 1, 1, 1, 1]
    fig, axes = plt.subplots(
        ncols=len(widths),
        nrows=len(heights),
        sharex="all",
        constrained_layout=True,
        gridspec_kw={
            "width_ratios": widths,
            "height_ratios": heights,
        },
        figsize=(15, len(heights) * 3),
    )
    fig.suptitle("Training monitor")
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = axes

    for poi_type, poi_effects_monitor in effects_monitor_df.groupby("poi_type"):
        ax1.plot(
            poi_effects_monitor["iteration"],
            poi_effects_monitor["Effect_size_abs"],
            label=poi_type,
        )
        effect_var_plot = ax2.plot(
            poi_effects_monitor["iteration"],
            poi_effects_monitor["Effect_var"],
            label=poi_type if synthetic_kernel_params is None else None,
        )
        if synthetic_kernel_params is not None:
            ax2.axhline(
                y=synthetic_kernel_params["poi_kernels"][
                    poi_type.replace("POI_Type_", "")
                ]["effects"],
                color=effect_var_plot[0].get_color(),
                linestyle="--",
                label=poi_type
                + f""" ({synthetic_kernel_params["poi_kernels"][
                    poi_type.replace("POI_Type_", "")
                ]["effects"]})""",
            )
        cut_off_plot = ax3.plot(
            poi_effects_monitor["iteration"],
            poi_effects_monitor["Cut_off"],
            label=poi_type if synthetic_kernel_params is None else None,
        )
        if synthetic_kernel_params is not None:
            ax3.axhline(
                y=synthetic_kernel_params["poi_kernels"][
                    poi_type.replace("POI_Type_", "")
                ]["distmax"],
                color=cut_off_plot[0].get_color(),
                linestyle="--",
                label=poi_type
                + f""" ({synthetic_kernel_params["poi_kernels"][
                    poi_type.replace("POI_Type_", "")
                ]["distmax"]})""",
            )

    ax1.title.set_text(
        f"Effect_size_abs ({step_size_string(effects_monitor_df['iteration'].unique())})"
    )
    ax1.set_ylim(ymin=0)
    ax1.legend()
    ax1.grid()

    ax2.title.set_text(
        f"Effect_var ({step_size_string(effects_monitor_df['iteration'].unique())})"
    )
    ax2.set_ylim(ymin=0)
    ax2.legend()
    ax2.grid()

    ax3.title.set_text(
        f"Cut_off ({step_size_string(effects_monitor_df['iteration'].unique())})"
    )
    ax3.set_ylim(ymin=0)
    ax3.legend()
    ax3.grid()

    ax4.title.set_text(
        f"ELBO ({step_size_string(likelihood_monitor_df['iteration'].values)})"
    )
    ax4.plot(
        likelihood_monitor_df["iteration"],
        likelihood_monitor_df["likelihood"],
        label="ELBO",
    )
    ax4.set_ylim(ymax=0)
    ax4.set_yscale("symlog")
    ax4.legend()
    ax4.grid()

    ax5.title.set_text(
        f"Matern32 - Lengthscales ({step_size_string(matern32_monitor_df['iteration'].values)})"
    )
    matern32_lengthscales_plot = ax5.plot(
        matern32_monitor_df["iteration"],
        matern32_monitor_df["lengthscales"],
        label="Lengthscales" if synthetic_kernel_params is None else None,
    )
    ax5.grid()
    if synthetic_kernel_params is not None:
        ax5.axhline(
            y=synthetic_kernel_params["matern32_kernel"]["lengthscales"],
            color=matern32_lengthscales_plot[0].get_color(),
            linestyle="--",
            label=f"""Lengthscales ({synthetic_kernel_params["matern32_kernel"]["lengthscales"]})""",
        )
    ax5.legend()

    ax6.title.set_text(
        f"Matern32 - Variance ({step_size_string(matern32_monitor_df['iteration'].values)})"
    )
    matern32_variance_plot = ax6.plot(
        matern32_monitor_df["iteration"],
        matern32_monitor_df["variance"],
        label="Variance" if synthetic_kernel_params is None else None,
    )
    ax6.grid()
    if synthetic_kernel_params is not None:
        ax6.axhline(
            y=synthetic_kernel_params["matern32_kernel"]["variance"],
            color=matern32_variance_plot[0].get_color(),
            linestyle="--",
            label=f"""Variance ({synthetic_kernel_params["matern32_kernel"]["variance"]})""",
        )
    ax6.legend()

    ax7.title.set_text(
        f"RMSE ({step_size_string(model_evaluation_monitor_df['iteration'].values)})"
    )
    ax7.plot(
        model_evaluation_monitor_df["iteration"],
        model_evaluation_monitor_df["RMSE_IS"],
        label="RMSE_IS",
        marker="+",
        linestyle="dashed",
    )
    ax7.plot(
        model_evaluation_monitor_df["iteration"],
        model_evaluation_monitor_df["RMSE_OOS"],
        label="RMSE_OOS",
        marker="+",
        linestyle="dashed",
    )
    ax7.set_yscale("log")
    ax7.legend()
    ax7.grid()

    ax8.title.set_text(
        f"Total Likelihood ({step_size_string(model_evaluation_monitor_df['iteration'].values)})"
    )
    ax8.plot(
        model_evaluation_monitor_df["iteration"],
        model_evaluation_monitor_df["TOTAL_LIK_IS"],
        label="TOTAL_LIK_IS",
        marker="+",
        linestyle="dashed",
    )
    ax8.plot(
        model_evaluation_monitor_df["iteration"],
        model_evaluation_monitor_df["TOTAL_LIK_OOS"],
        label="TOTAL_LIK_OOS",
        marker="+",
        linestyle="dashed",
    )
    ax8.set_yscale("symlog")
    ax8.legend()
    ax8.grid()

    return fig


def cpu_threads_vs_training_time_plot(projects_information_dicts) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.suptitle("Training time vs. available CPU threads.")

    for project, project_information_dicts in projects_information_dicts.items():
        for (
            model_base_name,
            project_model_information_dicts,
        ) in project_information_dicts.items():
            ax.scatter(
                [
                    information_dict["system_resources"]["os_sched_affinity"]
                    for information_dict in project_model_information_dicts
                ],
                [
                    information_dict["model_pure_training_time"]
                    for information_dict in project_model_information_dicts
                ],
                label=f"{project} ({model_base_name})",
                marker="+",
            )

    ax.set_xlabel("Number of available CPU threads")
    ax.set_ylabel("Pure model training time in seconds")
    ax.legend()
    ax.grid()

    return fig


def decay_plot(poi_types, distmax_values, kernel_type):
    max_distance = max(distmax_values) * 1000
    max_distance_to_plot = ((max_distance // 250) + 1) * 250
    if kernel_type == "Gaussian":
        max_distance_to_plot *= 3
    d = np.linspace(0, max_distance_to_plot, 100)

    fig, ax = plt.subplots(figsize=(10, 10))

    for poi_type, distmax in zip(poi_types, distmax_values):
        distmax *= 1000
        d_out = None
        if kernel_type == "Linear":
            d_out = np.maximum((-1 / distmax) * d + 1, 0) * 100
        elif kernel_type == "Gaussian":
            d_out = np.exp(-0.5 * (d / distmax) ** 2)
        else:
            raise Exception(f"Unknown kernel type {kernel_type}")
        ax.plot(d, d_out, label=poi_type)
    ax.set_xlabel("Distance [in meter]")
    ax.set_ylabel("Relative effect size [in percent]")
    ax.legend(
        # loc=1,
        framealpha=1,
        fancybox=True,
    )

    return fig


def latent_poi_influence_plot(
    g,
    coords_test,
    coords_poi_j,
    poi_type,
    shapes_reader,
):
    idx = np.argsort(g)

    fig, ax = plt.subplots(figsize=(10, 10))

    sc = ax.scatter(
        coords_test[idx, 0],
        coords_test[idx, 1],
        s=80,
        c=g[idx],
        vmin=-max(abs(g)),
        # vmin=-0.1,
        vmax=max(abs(g)),
        # vmax=0.1,
        alpha=0.75,
        rasterized=True,
    )

    ax.scatter(
        x=coords_poi_j[:, 0],
        y=coords_poi_j[:, 1],
        s=30,
        label=poi_type,
        c="red",
        alpha=0.75,
    )

    ax.legend(
        loc=(0.01, 0.93),
        framealpha=1,
        fancybox=True,
        prop={"size": 15},
    )

    cb = colorbar(sc)
    cb.set_label(
        label=r"$h_\gamma\rm\ for\ " + poi_type + r"$",
        fontsize=15,
    )

    if shapes_reader is not None:
        shapes = []
        for s in shapes_reader.shapeRecords():
            shapes.append(s.shape)

        for s in shapes:
            x_coords = [i[0] for i in s.points[:]]
            y_coords = [i[1] for i in s.points[:]]
            ax.plot(x_coords, y_coords, color="grey")

    bottom_lim, top_lim = np.quantile(coords_test[:, 1], q=[0.01, 0.99])
    left_lim, right_lim = np.quantile(coords_test[:, 0], q=[0.01, 0.99])

    length = max(top_lim - bottom_lim, right_lim - left_lim) * 1.1
    vertical_middle = top_lim - (top_lim - bottom_lim) / 2
    horizontal_middle = right_lim - (right_lim - left_lim) / 2

    ax.set_ylim(
        top=vertical_middle + length / 2,
        bottom=vertical_middle - length / 2,
    )
    ax.set_xlim(
        left=horizontal_middle - length / 2,
        right=horizontal_middle + length / 2,
    )

    return fig


def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def spatial_variation_plot(
    g,
    coords_test,
    shapes_reader,
):
    idx = np.argsort(g)

    fig, ax = plt.subplots(figsize=(10, 10))

    sc = ax.scatter(
        x=coords_test[idx, 0],
        y=coords_test[idx, 1],
        s=80,
        c=g[idx],
        # vmin=-max(abs(g)),
        vmin=min(g),
        # vmin=-2.5,
        # vmax=max(abs(g)),
        # vmax=0,
        vmax=max(g),
        alpha=0.75,
        rasterized=True,
    )

    cb = colorbar(sc)
    cb.set_label(
        label=r"Spatial variation $h_0$",
        fontsize=15,
    )

    if shapes_reader is not None:
        shapes = []
        for s in shapes_reader.shapeRecords():
            shapes.append(s.shape)

        for s in shapes:
            x_coords = [i[0] for i in s.points[:]]
            y_coords = [i[1] for i in s.points[:]]
            ax.plot(x_coords, y_coords, color="grey")

    bottom_lim, top_lim = np.quantile(coords_test[:, 1], q=[0.01, 0.99])
    left_lim, right_lim = np.quantile(coords_test[:, 0], q=[0.01, 0.99])

    length = max(top_lim - bottom_lim, right_lim - left_lim) * 1.1
    vertical_middle = top_lim - (top_lim - bottom_lim) / 2
    horizontal_middle = right_lim - (right_lim - left_lim) / 2

    ax.set_ylim(
        top=vertical_middle + length / 2,
        bottom=vertical_middle - length / 2,
    )
    ax.set_xlim(
        left=horizontal_middle - length / 2,
        right=horizontal_middle + length / 2,
    )

    return fig
