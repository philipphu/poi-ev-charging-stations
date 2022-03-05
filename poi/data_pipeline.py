"""
Defines a general data_pipeline function and some helper functions needed by the data_pipeline.
"""
import numpy as np
import scipy.spatial.distance

from loguru import logger


def convert_coordinates(
    longitude,
    latitude,
    center: float,
):
    """Converts pair(s) of longitude and latitude to x, y coordinates on a plane in km using equirectangular projection.

    Args:
        longitude:
            Single value or sequence of values for longitude in degrees in the range -180 to 180.
        latitude:
            Single value or sequence of values for latitude in degrees in the range -90 to 90.
        center:
            Latitude value for the center of the map in degrees.

    Returns:
    """
    r = 6371  # Earth Radius in km

    x = r * np.deg2rad(longitude) * np.cos(np.deg2rad(center))
    y = r * np.deg2rad(latitude)

    return x, y


def min_dist(locs, locs_poi):
    """Calculates for each location in ``locs`` the minimum distance to a poi in ``locs_poi``.

    Args:
        locs:
            2D array of (x,y) coordinates of the locations for which to calculate the minimum distances
        locs_poi:
            2D array of (x,y) coordinates of the locations of the poi's to which to calculate the minimum distance.

    Returns:
        2D array containing the minimum distance values.
    """
    d = scipy.spatial.distance.cdist(locs_poi, locs)
    d = np.sort(d, axis=0)
    return (d[0, :]).reshape(-1, 1)


def poi_count(locs, locs_poi, radius):
    """Calculates for each location in ``locs`` the number of poi's within the given radius.

    Args:
        locs:
            2D array of (x,y) coordinates of the locations for which to calculate the poi counts.
        locs_poi:
            2D array of (x,y) coordinates of the locations of the poi's to which to calculate the poi counts.
        radius:
            Radius in km (if (x,y) coordinates are generated with the earth radius in km as well).

    Returns:
        2D array containing the poi count values.
    """
    d = scipy.spatial.distance.cdist(locs_poi, locs)
    poi_in_radius = d < radius
    return sum(poi_in_radius, 0).reshape(-1, 1)


def data_pipeline(
    data_longitude,
    data_latitude,
    data_response_variable,
    poi_longitude,
    poi_latitude,
    poi_type_indicator,
    data_covariates=None,
    latitude_center=None,
    train_split_size=0.8,
    knot_points=0.3,
    feature_engineering=False,
    feature_type=None,
    poi_spatial_outlier=None,
    count_feature_d_max=2,
):
    coords = np.c_[data_longitude, data_latitude]
    # Map coordinates on plane
    if latitude_center is None:
        # if no center of latitude for the map was specified, just use the median of the latitude values of the data
        latitude_center = np.median(data_latitude)
        logger.info(
            f"Using computed latitude_center for coordinate conversion: {latitude_center}"
        )
    data_x, data_y = convert_coordinates(
        longitude=data_longitude,
        latitude=data_latitude,
        center=latitude_center,
    )

    poi_x, poi_y = convert_coordinates(
        longitude=poi_longitude,
        latitude=poi_latitude,
        center=latitude_center,
    )

    # Scaling for numerical stability of covariance matrix
    scale_x, scale_y = np.mean(data_x), np.mean(data_y)

    data_x -= scale_x
    data_y -= scale_y

    poi_x -= scale_x
    poi_y -= scale_y

    # Build design matrix
    X = np.c_[data_x, data_y]
    if data_covariates is not None:
        X = np.c_[X, data_covariates]

    # Response variable
    data_response_variable = data_response_variable.astype(np.float64)

    # POI data
    typeIndicator = poi_type_indicator

    locs_poi = np.c_[poi_x, poi_y]
    coords_poi = np.c_[poi_longitude, poi_latitude]

    # Handle spatial outliers
    if poi_spatial_outlier is not None:
        if poi_spatial_outlier["method"] == "distance_from_center":
            selector = (
                np.sqrt(locs_poi[:, 0] ** 2 + locs_poi[:, 1] ** 2)
                <= poi_spatial_outlier["parameter"]
            )
        elif poi_spatial_outlier["method"] == "distance_from_data":
            selector = np.any(
                scipy.spatial.distance.cdist(locs_poi, X[:, 0:2])
                <= poi_spatial_outlier["parameter"],
                axis=1,
            )
        else:
            raise Exception(
                f"Selected poi_spatial_outlier method of {poi_spatial_outlier['method']} not supported."
            )
        if poi_spatial_outlier["type"] == "remove":  # remove the spatial outliers
            pass  # no need to change the selector
        elif (
            poi_spatial_outlier["type"] == "only"
        ):  # option to only keep the spatial outliers
            selector = ~selector  # inverse the selector to get the outliers
        else:
            raise Exception(
                f"Selected poi_spatial_outlier type of {poi_spatial_outlier['type']} not supported."
            )
        typeIndicator = typeIndicator[selector, :]
        coords_poi = coords_poi[selector, :]
        locs_poi = locs_poi[selector, :]

    typeMinDist = np.array(
        [
            np.percentile(
                scipy.spatial.distance.cdist(
                    XA=X[:, 0:2],  # x and y coordinates of the data
                    XB=locs_poi[
                        typeIndicator[:, typIdx] == 1, :
                    ],  # x and y coordinates of the type typIdx
                ),
                q=0.05,
            )
            for typIdx in range(typeIndicator.shape[1])
        ]
    )

    if feature_engineering == True:
        if feature_type not in ("min_dist", "count", "both"):
            raise Exception(f"Selected feature_type of {feature_type} not supported.")
        for typIdx in range(typeIndicator.shape[1]):
            locs_poi_of_type = locs_poi[
                typeIndicator[:, typIdx] == 1, :
            ]  # select pois of type typIdx
            if feature_type in ("min_dist", "both"):
                min_d = min_dist(
                    locs=X[:, 0:2],
                    locs_poi=locs_poi_of_type,
                )
                X = np.append(X, min_d, axis=1)
            if feature_type in ("count", "both"):
                count = poi_count(
                    locs=X[:, 0:2],
                    locs_poi=locs_poi_of_type,
                    radius=count_feature_d_max,
                )
                X = np.append(X, count, axis=1)

    # split training and test
    np.random.seed(0)
    idx = np.random.rand(X.shape[0]) < train_split_size
    X_train, X_test = X[idx, :], X[~idx, :]
    y_train, y_test = data_response_variable[idx, :], data_response_variable[~idx, :]
    coords_train, coords_test = coords[idx], coords[~idx]

    # Knot points for FITC
    if knot_points > 1:
        M = knot_points
    else:
        M = int(knot_points * X_train.shape[0])
    Z = X_train[np.random.choice(X_train.shape[0], M, replace=False)]

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
    )
