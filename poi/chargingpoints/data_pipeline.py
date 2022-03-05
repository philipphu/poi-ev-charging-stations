from loguru import logger
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import poi
from poi.chargingpoints import PROJECT_NAME
from poi.data_pipeline import data_pipeline


def project_data_pipeline(
    feature_engineering=False,
    feature_type=None,
    knot_points=80,
    count_feature_d_max=2,
):
    poi_df = pd.read_csv(
        poi.paths.project_input_file(
            project=PROJECT_NAME,
            filename="amsterdam_pois_osm.csv",
            mkdir=False,
        ),
        index_col=0,
    )

    if not poi.paths.project_input_file(
        project=PROJECT_NAME,
        filename="cp_densities_filtered_converted.csv",
        mkdir=False,
    ).exists():
        cps_excel = pd.read_excel(
            poi.paths.project_input_file(
                project=PROJECT_NAME,
                filename="cp_densities_filtered.xlsx",
                mkdir=False,
            ),
            sheet_name=0,
            engine="openpyxl",
        )
        cps_excel.to_csv(
            poi.paths.project_input_file(
                project=PROJECT_NAME,
                filename="cp_densities_filtered_converted.csv",
                mkdir=True,
            ),
            index=False,
        )
        logger.info("Created cps csv file from excel file.")
        del cps_excel
    cps = pd.read_csv(
        poi.paths.project_input_file(
            project=PROJECT_NAME,
            filename="cp_densities_filtered_converted.csv",
            mkdir=False,
        )
    )

    cps = cps.drop_duplicates(
        subset=["stationID", "lat", "lon", "avg_utilization"],
        keep="first",
    )

    y = cps[["avg_utilization"]].values
    y = (y - np.mean(y)) / np.std(y)

    data_longitude = cps[["lon"]].values
    data_latitude = cps[["lat"]].values
    data_response_variable = y

    poi_longitude = poi_df["lon"].values
    poi_latitude = poi_df["lat"].values

    poi_broader_categories = {
        "restaurant": [
            "restaurant",
        ],
        "store": [
            "clothing_store",
            "department_store",
            "grocery_or_supermarket",
            "shopping_mall",
            "supermarket",
        ],
        "education": [
            "school",
            "university",
        ],
        "public_transportation": [
            "bus_station",
            "light_rail_station",
            "subway_station",
            "train_station",
        ],
    }

    poi_df_new = poi_df[["lat", "lon"]].copy(deep=True)
    poi_types = []

    for category, category_types in poi_broader_categories.items():
        poi_df_new[category] = poi_df[category_types].any(axis=1).astype(int)
        poi_types.append(category)

    poi_type_indicator = poi_df_new[poi_types].values
    poi_types = [
        a.title().replace("_", " ") for a in poi_types
    ]  # to have correct spelling everywhere else

    # data_covariates = cps[
    #     [
    #         "traffic250",
    #         # "traffic500",
    #         # "traffic750",
    #         # "traffic1000",
    #         "PopDens",
    #         "LogIncPP",
    #         "CarDens",
    #     ]
    # ].values

    data_covariates = cps[["traffic250"]]  # "traffic500", # "traffic750", # "traffic1000",

    standard_scaler = StandardScaler()

    data_covariates = np.append(
        data_covariates,
        standard_scaler.fit_transform(
            cps[
                [
                    "PopDens",
                    "LogIncPP",
                    "CarDens",
                ]
            ]
        ),
        axis=1,
    )

    intercept = np.ones((data_covariates.shape[0], 1))
    data_covariates = np.append(
        intercept,
        data_covariates,
        axis=1,
    )

    latitude_center = 52.361327
    poi_spatial_outlier = {
        "method": "distance_from_data",
        "parameter": 3,
        "type": "remove",
    }

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
    ) = data_pipeline(
        data_longitude=data_longitude,
        data_latitude=data_latitude,
        data_response_variable=data_response_variable,
        poi_longitude=poi_longitude,
        poi_latitude=poi_latitude,
        poi_type_indicator=poi_type_indicator,
        data_covariates=data_covariates,
        latitude_center=latitude_center,
        train_split_size=0.8,
        knot_points=knot_points,
        feature_engineering=feature_engineering,
        feature_type=feature_type,
        poi_spatial_outlier=poi_spatial_outlier,
        count_feature_d_max=count_feature_d_max,
    )

    return (
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
    )
