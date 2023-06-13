import argparse
import logging
import uuid
import warnings
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rasterio
from rasterio import features
from scipy.ndimage import binary_dilation
from scipy.spatial import distance_matrix
from tqdm import tqdm

logging.basicConfig(filename='./log_skred_prop.log', filemode='w', level=logging.INFO)


def get_indices(arr):
    """Returns the indices of the True values in a boolean array"""
    return [i for i in range(len(arr)) if arr[i]]


def is_discontinuous(arr):
    """Returns True if the array is not continuous, i.e. if there is a gap between two consecutive values"""
    return not all(arr[i] + 1 == arr[i + 1] for i in range(len(arr) - 1))


def landslide_retrogression_2d(prof: np.ndarray, index_init: int, res: int = 1, min_slope=1 / 15, min_length=15,
                               max_length=2000, tol=5):
    """Returns the indices of the landslide propagation in a 2D profile.

    Args:
        prof (np.ndarray): 2D array with the profile to be analyzed. The first column must be the distance,
                           the second column the elevation.
        index_init (int): index of the landslide head in the prof array.
        res (int, optional): resolution of the profile in meters. Defaults to 1.
        min_slope (float, optional): minimum slope allowed for the landslide propagation. Defaults to 1/15.
        min_length (int, optional): minimum length of the landslide propagation in meters. Defaults to 15.
        max_length (int, optional): maximum length of the landslide propagation in meters. Defaults to 2000.
        tol (int, optional): tolerance in meters for the maximum difference in elevation between two consecutive points
                                in the landslide propagation. Defaults to 5.

    Returns:
        (np.ndarray): 1D array with the indices of the landslide propagation in the prof array.
    """

    # shut up RuntimeWarning
    np.seterr(divide='ignore', invalid='ignore')
    logging.info(f"New 2D Analysis = {datetime.now()}")

    max_iter = int(max_length // res)
    min_iter = int(min_length // res)

    logging.info(f"max_iter = {max_iter}")
    logging.info(f"min_iter = {min_iter}")

    n_iter = 1
    mask = np.zeros_like(prof[:, 1]).astype(bool)

    while n_iter < max_iter:
        n_min = index_init - n_iter
        n_min = n_min if n_min >= 0 else 0

        n_max = index_init + n_iter
        n_max = n_max if n_max <= len(prof) else len(prof)

        ii = [i for i in range(n_min, n_max)]

        rel_dist = distance_matrix(np.c_[prof, np.zeros((len(prof), 1))],
                                   np.c_[prof[index_init, :].reshape(1, 2), np.zeros(1, )]).flatten()

        rel_h = np.zeros_like(prof[:, 1])
        rel_h[ii] = prof[ii, 1] - prof[index_init, 1]
        slope = rel_h / rel_dist

        mask_new = slope > min_slope

        if ((np.all(mask_new == mask) and is_discontinuous(get_indices(mask_new))) or
            np.all(mask_new == np.zeros_like(mask).astype(bool))) and n_iter > min_iter:
            logging.info(f"in tolerance loop at {n_iter}")
            mask = np.zeros_like(mask_new).astype(bool)
            if len(get_indices(mask_new)) > 0:
                mask[np.arange(get_indices(mask_new)[0], get_indices(mask_new)[-1] + 1)] = True

            tol -= 1
            if tol <= 0:
                logging.info(f"tolerance loop break at {n_iter}")
                break
        else:
            mask = mask_new

        n_iter += 1

    logging.info(f"Done at {n_iter} iterations")

    plt.plot(prof[:, 0], prof[:, 1])
    plt.scatter(prof[mask, 0], prof[mask, 1], c="r")
    plt.scatter(prof[index_init, 0], prof[index_init, 1])
    plt.plot(prof[:, 0], rel_dist / 15 + prof[index_init, 1], "--", c="gray", linewidth=0.5)
    plt.show()


def landslide_retrogression_3d(dem: np.ndarray, initial_release: np.ndarray, dem_transform: rasterio.transform.Affine,
                               min_slope: float = 1 / 15, min_length: float = 200, max_length: float = 2000,
                               min_height: float = 5, initial_release_depth: float = 0, mask: np.ndarray = None, ):
    """
    Propagates a landslide from a release area in a DEM. Stop criteria is defined by the maximum slope, minimum and
    maximum length of the landslide. The propagation is done iteratively, starting from the release area and moving
    outwards. The propagation is done in 3D, i.e. the landslide can propagate in any direction.

    Parameters:
        dem (np.ndarray): DEM as a numpy array
        initial_release (np.ndarray): initial release area as a boolean numpy array.
                                      Must have the same shape and same transform as the DEM.
        dem_transform (Affine): affine transformation of the DEM/release.
        min_slope (float): minimum slope of the landslide. Default is 1/15 as in NVE's guidelines
        min_length (float): minimum length of the landslide. Default is 200 m.
        max_length (float): maximum length of the landslide. Default is 2000 m.
        min_height (float): minimum height of the landslide (checked after min_iter). Default is 5 m.
        initial_release_depth (float): depth of the initial release area. Default is 0.
        #todo: change to depth in the raster (as pixel value) instead.
        mask (np.ndarray): mask of the area outside analysis. Must have the same shape and same transform as the DEM.

    Returns:
        release (np.ndarray): propagated release area of the landslide as a boolean numpy array


    """
    print("runing landslide propagation...")
    if abs(round(dem_transform[0], 2)) != abs(round(dem_transform[4], 2)):
        print("Warning: DEM is not square")

    res = abs(dem_transform[0])

    min_iter = int(min_length // res)
    max_iter = int(max_length // res)

    # shut up RuntimeWarning
    np.seterr(divide='ignore', invalid='ignore')
    logging.info(f"New 3D Analysis = {datetime.now()}")

    logging.info(f"max_iter = {max_iter}")
    logging.info(f"min_iter = {min_iter}")

    n_iter = 1

    release = initial_release.copy()

    mask = np.zeros_like(dem).astype(bool) if mask is None else mask

    release_coords, _, _ = get_coordinates(initial_release, dem, dem_transform)
    release_coords[:, 2] = release_coords[:, 2] - initial_release_depth

    animation = []
    animation.append(initial_release)

    with tqdm(total=max_iter, desc="iterations") as pbar:
        while n_iter < max_iter:

            buffered = create_buffer(release, 1)
            buffered_coords, i_buffered, j_buffered = get_coordinates(buffered, dem, dem_transform)

            h_min = 0 if n_iter <= min_iter else min_height
            slopes, _, _ = compute_slope(buffered_coords, release_coords, h_min=h_min)

            neighbours_filtered = [(i_buffered[ii], j_buffered[ii]) for ii in
                                   list(np.where(np.array(slopes) > min_slope)[0])]

            release_after = release.copy()

            for ii in neighbours_filtered:
                release_after[ii] = 1

            release_after[mask == 1] = 0

            if np.all(release.astype(bool) == release_after.astype(bool)) and n_iter > min_iter:
                # todo: check height as well

                print(f"Calculation done in {n_iter} iterations")
                break

            release = release_after.copy()
            animation.append(release_after)
            n_iter += 1
            pbar.update(1)

    return release, animation


def get_coordinates(arr, dem, transform):
    """
    Returns the coordinates of the points in the array arr in the same order as in the array.
    Parameters:
        arr (np.ndarray): array of points
        dem (np.ndarray): DEM as a numpy array
        transform (Affine): affine transformation of the DEM/release.
    Returns:
        arr_coords (np.ndarray): array of coordinates of the points in the same order as in arr
    """
    i_arr, j_arr = np.where(arr == 1)
    x_arr, y_arr = rasterio.transform.xy(transform, i_arr, j_arr)
    z_arr = np.array([dem[ii, jj] for ii, jj in zip(i_arr, j_arr)])
    arr_coords = np.c_[x_arr, y_arr, z_arr]
    return arr_coords, i_arr, j_arr


def create_buffer(image, buffer_size):
    # Perform binary dilation on the image
    dilated_image = binary_dilation(image, iterations=buffer_size)

    # Calculate the buffer by subtracting the original image
    # from the dilated image
    buffer = ((dilated_image - image) > 0).astype(bool)

    return buffer


def compute_slope(coords_1: np.ndarray, coords_2: np.ndarray, h_min: float = 0, nodata: int = -9999) -> tuple:
    """
    Compute the slopes of the given dem with respect to the (source) points
    Args:
        coords_1: raster 1 coordinates
        coords_2: raster 2 coordinates
        h_min: minimum height difference where slopes are calculated
        nodata: value given to pixels with no data

    Returns:
        max_slope: array with slopes (same shape as input dem)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xy_1 = coords_1[:, :2]
        xy_2 = coords_2[:, :2]
        z1 = coords_1[:, -1]
        z2 = coords_2[:, -1]
        distance_mtx = distance_matrix(xy_1, xy_2)
        height_mtx = z1[:, np.newaxis] - z2
        hl_ratio = height_mtx / distance_mtx
        hl_ratio[height_mtx < h_min] = nodata
        max_slope = np.max(hl_ratio, axis=1)
        ii_max = np.argmax(hl_ratio, axis=1)
        dist_of_max = np.array([dd[ii_max[ii]] for ii, dd in enumerate(distance_mtx)])
        height_of_max = np.array([dd[ii_max[ii]] for ii, dd in enumerate(height_mtx)])

        return max_slope, dist_of_max, height_of_max


def rasterize_release(file_path: str, dem_profile: rasterio.profiles.Profile, out_path: str = None) -> np.ndarray:
    """
    Rasterize the release area
    Args:
        file_path: path to the shapefile
        dem_profile: profile of the dem
        out_path: path to the output raster

    Returns:
        rasterized: rasterized release area as a numpy array
    """
    release_shp = gpd.read_file(file_path)
    dem_height = dem_profile['height']
    dem_width = dem_profile['width']
    dem_transform = dem_profile['transform']

    geom = [shapes for shapes in release_shp.geometry]

    rasterized = features.rasterize(geom,
                                    out_shape=(dem_height, dem_width),
                                    fill=0,
                                    out=None,
                                    transform=dem_transform,
                                    all_touched=True,
                                    default_value=1,
                                    dtype=None)
    if out_path is not None:
        if not out_path.endswith(".tif"):
            out_path += ".tif"
        with rasterio.open(out_path, 'w', **dem_profile) as dst:
            dst.write(rasterized, indexes=1)

    return rasterized


def polygonize_results(result_array: np.ndarray, dem_profile: rasterio.profiles.Profile, out_path: str):
    """
    Polygonize the results and save them as a shapefile
    Args:
        result_array: array with the results
        dem_profile: profile of the dem
        out_path: path to the output shapefile

    Returns:
        None
    """
    if not out_path.endswith(".shp"):
        out_path += ".shp"

    raster_transform = dem_profile['transform']
    raster_crs = dem_profile['crs']

    results = ({"properties": {"value": int(v)}, "geometry": s}
               for i, (s, v) in enumerate(features.shapes(result_array, mask=None, transform=raster_transform)))
    geoms = list(results)
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(geoms)
    gpd_polygonized_raster = gpd_polygonized_raster[gpd_polygonized_raster.value == 1]
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs(raster_crs)
    gpd_polygonized_raster.to_file(out_path)


def save_results(results, raster_profile, filename):
    """
    Save the results to a raster file
    Args:
        results: results to save
        raster_profile: profile of the raster to save
        filename: filename of the raster to save

    Returns:
        None
    """
    try:
        with rasterio.open(filename, "w+", **raster_profile) as out:
            out.write(results, indexes=1)
    except rasterio._err.CPLE_AppDefinedError:
        print("File already exists")
        new_filename = filename.split(".tif")[0] + "_" + str(uuid.uuid1())[:8] + ".tif"
        print(f"Saving as {new_filename}")
        save_results(results, raster_profile, new_filename)


def get_msml_mask(bounds: tuple, profile):
    """
    Get the MSML mask as an array for the given bounds
    """
    from os import path
    from urllib.request import urlopen
    import tempfile
    import pandas as pd
    with tempfile.TemporaryDirectory() as tempdir:
        xmin, ymin, xmax, ymax = bounds

        # Get MSML mask
        url_nve_msml = "https://gis3.nve.no/map/rest/services/Mapservices/MarinGrense/MapServer/7/query?" \
                       "geometry=xmin%3A+{}%2C+ymin%3A+{}%2C+xmax%3A+{}%2C+ymax%3A+{}&" \
                       "geometryType=esriGeometryEnvelope&f=geojson"
        query_msml_mask = url_nve_msml.format(xmin, ymin, xmax, ymax)
        response = urlopen(query_msml_mask)
        json_mask = response.read()

        with open(path.join(tempdir, "msml_mask.json"), "wb") as file_json_mask:
            file_json_mask.write(json_mask)
        mask_msml = gpd.read_file(path.join("test", "msml_mask.json"), driver='GeoJSON').to_crs(epsg=25833)

        # Get Areal under MG mask
        url_nve_aumg = "https://gis3.nve.no/map/rest/services/Mapservices/MarinGrense/MapServer/8/query?" \
                       "geometry=xmin%3A+{}%2C+ymin%3A+{}%2C+xmax%3A+{}%2C+ymax%3A+{}&" \
                       "geometryType=esriGeometryEnvelope&f=geojson"
        query_aumg_mask = url_nve_aumg.format(xmin, ymin, xmax, ymax)
        response_2 = urlopen(query_aumg_mask)
        json_mask_2 = response_2.read()
        with open(path.join(tempdir, "marin_mask.json"), "wb") as file_json_mask:
            file_json_mask.write(json_mask_2)
        mask_aumg = gpd.read_file(path.join("test", "marin_mask.json"), driver='GeoJSON').to_crs(epsg=25833)

        # concatenate masks
        mask = gpd.GeoDataFrame(pd.concat([mask_msml, mask_aumg], ignore_index=True)).set_crs(epsg=25833)

        mask.to_file(f"{tempdir}/msml_mask.shp")
        mask_array = rasterize_release(f"{tempdir}/msml_mask.shp", profile)

    return mask_array


def animate_landslide_retrogresion(animation, dem, n_frames=10):
    """
    Animate the landslide retrogression
    Args:
        animation: list of arrays with the landslide retrogression
        dem: array with the dem
        n_frames: number of frames to use

    Returns:
        fig: plotly figure with the animation
    """
    print("Animating landslide retrogression")
    if n_frames > len(animation):
        n_frames = len(animation)
    frame_step = len(animation) // n_frames if len(animation) // n_frames > 1 else 2

    color_red = 'rgba(255, 0, 0, 0.5)'
    color_white = 'rgba(255, 255, 255, 0.0)'
    color_black = 'rgba(0, 0, 0, 0.5)'

    fig_data = [go.Heatmap(z=dem, colorscale="earth", showscale=False),
                go.Heatmap(z=animation[0], colorscale=[[0, color_white], [1, color_black]], showscale=False)]
    fig = go.Figure(
        data=fig_data,
        layout=go.Layout(
            title="Step 0",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None])])]
        ),
    )

    frames = [go.Frame(data=[go.Heatmap(z=dem, colorscale="earth", showscale=False),
                             go.Heatmap(z=animation[i], colorscale=[[0, color_white], [1, color_red]],
                                        showscale=False)],
                       layout=go.Layout(title_text=f"Step {i}"))
              for i in range(1, len(animation), frame_step)]
    frames.append(go.Frame(data=[go.Heatmap(z=dem, colorscale="earth", showscale=False),
                                 go.Heatmap(z=animation[-1], colorscale=[[0, color_white], [1, color_red]],
                                            showscale=False)],
                           layout=go.Layout(title_text=f"Step {len(animation)}")))
    fig.frames = frames

    height, width = dem.shape

    fig.update_xaxes(scaleanchor="y")
    fig.update_yaxes(scaleratio=1, autorange="reversed")
    fig.update_layout(xaxis_range=[0, width], yaxis_range=[0, height])
    fig.update_layout(width=1000, height=1000, coloraxis_showscale=False, plot_bgcolor=color_white,
                      )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="Path to the source area shapefile", required=True)
    parser.add_argument("--dem_path", help="Path to the dem", required=True)
    parser.add_argument("--out_path", help="Path to the output shapefile", default=None, required=False)
    parser.add_argument("--verbose", help="Show plots", type=bool, default=False, required=False)
    parser.add_argument("--min_slope", help="Minimum slope", type=float, default=1 / 15, required=False)
    parser.add_argument("--initial_release_depth", help="Initial release depth", type=float, default=0, required=False)

    args = parser.parse_args()

    with rasterio.open(args.dem_path) as src:
        dem_array = src.read(1)
        transform = src.transform
        profile = src.profile
    rel = rasterize_release(args.source_path, profile)

    release_result, animation = landslide_retrogression_3d(dem_array, rel, transform,
                                                           min_slope=args.min_slope,
                                                           initial_release_depth=args.initial_release_depth)

    if args.verbose:
        fig = animate_landslide_retrogresion(animation, dem_array)
        fig.show()

    if args.out_path is not None:
        polygonize_results(release_result, profile, args.out_path)


if __name__ == '__main__':
    main()
