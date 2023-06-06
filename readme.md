# Quick-Clay Retrogression Module

This module provides two functions to iteratively calculate the propagation of the failure surface in 2D and 3D starting
from an initial failure zone (initial release/source area). 
The module is designed to calculate the retrogression of landslides in quick-clay defined by geometric conditions 
(generally slope greater than 1:15).

The retrogression computation is based on the terrain criteria for landslide retrogression showed in [NVE Kvikkleireveileder 1/2019](https://publikasjoner.nve.no/veileder/2019/veileder2019_01.pdf).

![terrain criteria for landslide retrogression](landslide_retrogression.png)

## Usage

```python
import rasterio
from landslide_retrogression import landslide_retrogression_3d, save_results

# import dem and release raster
with rasterio.open("dem.tif") as src:
    dem_test = src.read(1)
    transform = src.transform
    profile = src.profile
with rasterio.open("release.tif") as src_rel:
    rel = src_rel.read(1)

# calculate retrogression
release_result = landslide_retrogression_3d(dem_test, rel, transform, verbose=False, min_slope=1 / 15)

# save results
save_results(release_result, profile, "./release_result.tif")
```