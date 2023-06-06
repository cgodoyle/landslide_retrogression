import logging
import timeit
import uuid
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
from matplotlib.colors import ListedColormap
from rasterio import features
from scipy.ndimage import binary_dilation
from scipy.spatial import distance_matrix

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
                               time_limit: tuple = (60, 1800), verbose: bool = False):
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
        time_limit (tuple): time limit for the propagation. Default is 60 seconds for the first iteration
                            and 1800 for total run time.
        verbose (bool): if True, prints the number of iterations and plots propagation. Default is False.

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
    start_time = timeit.default_timer()

    if verbose:
        cmap = ListedColormap(['white', 'red'])
        fig, ax = plt.subplots()
        im = ax.imshow(release, cmap=cmap)
        print("iteration: ")

    while n_iter < max_iter:
        if n_iter % 2 == 0 and verbose:
            print(n_iter, end=",")
            im.set_data(release)
            plt.pause(0.1)
            fig.canvas.draw()

        i_rel, j_rel = np.where(release == 1)
        x_rel, y_rel = rasterio.transform.xy(dem_transform, i_rel, j_rel)
        z_rel = np.array([dem[ii, jj] for ii, jj in zip(i_rel, j_rel)])
        release_coords = np.c_[x_rel, y_rel, z_rel]

        buffered = create_buffer(release, 1)
        i_buffered, j_buffered = np.where(buffered == 1)
        x_buffered, y_buffered = rasterio.transform.xy(dem_transform, i_buffered, j_buffered)
        z_buffered = np.array([dem[ii, jj] for ii, jj in zip(i_buffered, j_buffered)])
        buffered_coords = np.c_[x_buffered, y_buffered, z_buffered]

        slopes = compute_slope(buffered_coords, release_coords, h_min=0)

        neighbours_filtered = [(i_buffered[ii], j_buffered[ii]) for ii in
                               list(np.where(np.array(slopes) > min_slope)[0])]

        release_after = release.copy()

        for ii in neighbours_filtered:
            release_after[ii] = 1

        if np.all(release.astype(bool) == release_after.astype(bool)) and n_iter > min_iter:
            # todo: check height as well

            end_time = timeit.default_timer()
            print(f"break at {n_iter}")
            print(f"total time = {end_time - start_time}")
            break

        release = release_after.copy()
        n_iter += 1

    if verbose:
        plt.show()

    return release


def create_buffer(image, buffer_size):
    # Perform binary dilation on the image
    dilated_image = binary_dilation(image, iterations=buffer_size)

    # Calculate the buffer by subtracting the original image
    # from the dilated image
    buffer = ((dilated_image - image) > 0).astype(bool)

    return buffer


def compute_slope(coords_1: np.ndarray, coords_2: np.ndarray, h_min: float = 0, nodata: int = -9999) -> np.ndarray:
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

        #todo: return slope / distance / height
        return max_slope


def rasterize_release(file_path, dem_profile, out_path):
    """
    Rasterize the release area
    Args:
        file_path: path to the shapefile
        dem_profile: profile of the dem
        out_path: path to the output raster

    Returns:
        None
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

    with rasterio.open(out_path, 'w', **dem_profile) as dst:
        dst.write(rasterized, indexes=1)


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
        new_filename = filename.split(".tif")[0]+"_"+str(uuid.uuid1())[:8]+".tif"
        print(f"Saving as {new_filename}")
        save_results(results, raster_profile, new_filename)


if __name__ == '__main__':
    with rasterio.open("gjerdrum_dem.tif") as src:
        dem_test = src.read(1)
        transform = src.transform
        profile = src.profile
    with rasterio.open("release.tif") as src_rel:
        rel = src_rel.read(1)

    release_result = landslide_retrogression_3d(dem_test, rel, transform, verbose=False, min_slope=1 / 15)

    save_results(release_result, profile, "./test_release_result.tif")
