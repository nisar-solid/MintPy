#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Sara Mirzaee, Jul 2023                           #
#         Emre Havazli, Feb 2026                           #
############################################################

import datetime
import glob
import os
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from mintpy.constants import EARTH_RADIUS, SPEED_OF_LIGHT
from mintpy.utils import ptime, writefile


# ---------------------------------------------------------------------
# Constants / HDF5 paths (GUNW frequencyA, unwrappedInterferogram)
# ---------------------------------------------------------------------
DATASET_ROOT_UNW = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
PARAMETERS = (
    "/science/LSAR/GUNW/metadata/processingInformation/parameters/"
    "unwrappedInterferogram/frequencyA"
)
IDENTIFICATION = "/science/LSAR/identification"
RADARGRID_ROOT = "science/LSAR/GUNW/metadata/radarGrid"

DATASETS = {
    "xcoord": f"{DATASET_ROOT_UNW}/POL/xCoordinates",
    "ycoord": f"{DATASET_ROOT_UNW}/POL/yCoordinates",
    "unw": f"{DATASET_ROOT_UNW}/POL/unwrappedPhase",
    "cor": f"{DATASET_ROOT_UNW}/POL/coherenceMagnitude",
    "connComp": f"{DATASET_ROOT_UNW}/POL/connectedComponents",
    "ion": f"{DATASET_ROOT_UNW}/POL/ionospherePhaseScreen",
    "epsg": f"{DATASET_ROOT_UNW}/POL/projection",
    "xSpacing": f"{DATASET_ROOT_UNW}/POL/xCoordinateSpacing",
    "ySpacing": f"{DATASET_ROOT_UNW}/POL/yCoordinateSpacing",
    "polarization": "/science/LSAR/GUNW/grids/frequencyA/listOfPolarizations",
    "range_look": f"{PARAMETERS}/numberOfRangeLooks",
    "azimuth_look": f"{PARAMETERS}/numberOfAzimuthLooks",
}

PROCESSINFO = {
    "centerFrequency": "/science/LSAR/GUNW/grids/frequencyA/centerFrequency",
    "orbit_direction": f"{IDENTIFICATION}/orbitPassDirection",
    "platform": f"{IDENTIFICATION}/missionId",
    "start_time": f"{IDENTIFICATION}/referenceZeroDopplerStartTime",
    "end_time": f"{IDENTIFICATION}/referenceZeroDopplerEndTime",
    "rdr_xcoord": f"{RADARGRID_ROOT}/xCoordinates",
    "rdr_ycoord": f"{RADARGRID_ROOT}/yCoordinates",
    "rdr_slant_range": f"{RADARGRID_ROOT}/referenceSlantRange",
    "rdr_height": f"{RADARGRID_ROOT}/heightAboveEllipsoid",
    "rdr_incidence": f"{RADARGRID_ROOT}/incidenceAngle",
    "rdr_wet_tropo": f"{RADARGRID_ROOT}/wetTroposphericPhaseScreen",
    "rdr_hs_tropo": f"{RADARGRID_ROOT}/hydrostaticTroposphericPhaseScreen",
    "rdr_SET": f"{RADARGRID_ROOT}/slantRangeSolidEarthTidesPhase",
    "bperp": f"{RADARGRID_ROOT}/perpendicularBaseline",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _datasets_for_pol(polarization: str) -> dict:
    """Return a per-call datasets dict without mutating the global DATASETS."""
    out = {}
    for k, v in DATASETS.items():
        out[k] = v.replace("POL", polarization) if isinstance(v, str) and "POL" in v else v
    return out


def _grid_bounds_from_xy(xcoord: np.ndarray, ycoord: np.ndarray):
    """
    Compute pixel-edge bounds aligned to the xcoord/ycoord grid (pixel centers).
    Returns bounds in (minx, miny, maxx, maxy) and dx, dy (signed spacings).
    """
    if xcoord.size < 2 or ycoord.size < 2:
        raise ValueError("xcoord/ycoord must have at least 2 elements to infer spacing.")

    dx = float(xcoord[1] - xcoord[0])
    dy = float(ycoord[1] - ycoord[0])

    left = float(xcoord[0] - dx / 2.0)
    right = float(xcoord[-1] + dx / 2.0)

    top = float(ycoord[0] - dy / 2.0)
    bottom = float(ycoord[-1] + dy / 2.0)

    miny, maxy = (bottom, top) if bottom < top else (top, bottom)
    minx, maxx = (left, right) if left < right else (right, left)
    return (minx, miny, maxx, maxy), dx, dy


def _warp_to_grid_mem(
    *,
    src_path: str,
    src_epsg: int,
    dst_epsg: int,
    xcoord: np.ndarray,
    ycoord: np.ndarray,
    resample_alg: str,
):
    """
    Warp a raster to the exact xcoord/ycoord grid using MEM output.
    Uses bounds derived from pixel-edge and xRes/yRes with targetAlignedPixels.
    """
    bounds, dx, dy = _grid_bounds_from_xy(xcoord, ycoord)

    warp_opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=bounds,
        srcSRS=f"EPSG:{src_epsg}",
        dstSRS=f"EPSG:{dst_epsg}",
        xRes=abs(dx),
        yRes=abs(dy),
        targetAlignedPixels=True,
        resampleAlg=resample_alg,
    )
    dst = gdal.Warp("", src_path, options=warp_opts)
    if dst is None:
        raise RuntimeError(f"GDAL Warp failed for {src_path}")
    arr = dst.ReadAsArray()
    if arr is None:
        raise RuntimeError(f"Failed reading warped array for {src_path}")
    return arr


def _read_raster_epsg(path: str) -> int:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise OSError(f"Cannot open raster: {path}")
    srs = gdal.osr.SpatialReference(wkt=ds.GetProjection())
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    if epsg is None:
        raise ValueError(f"Could not determine EPSG from raster projection: {path}")
    return int(epsg)


def _make_rgi(grid_axes, values, method="linear"):
    """
    RegularGridInterpolator wrapper:
      - flips decreasing axes (SciPy requires increasing)
      - bounds_error=False + fill_value=np.nan to avoid crashing
    """
    axes = [np.asarray(a) for a in grid_axes]
    vals = values
    for dim, ax in enumerate(axes):
        if ax[0] > ax[-1]:
            axes[dim] = ax[::-1]
            vals = np.flip(vals, axis=dim)

    return RegularGridInterpolator(
        tuple(axes),
        vals,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )


def _read_valid_unw_mask(gunw_file: str, xybbox, pol: str):
    """
    Validity mask is ALWAYS based on finite unwrappedPhase (+ _FillValue check),
    using:
      /science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram/{pol}/unwrappedPhase
    """
    path = f"{DATASET_ROOT_UNW}/{pol}/unwrappedPhase"
    with h5py.File(gunw_file, "r") as ds:
        dset = ds[path]
        unw = dset[xybbox[1] : xybbox[3], xybbox[0] : xybbox[2]]
        fill = dset.attrs.get("_FillValue", None)

    valid = np.isfinite(unw)
    if fill is not None:
        valid &= (unw != fill)
    return valid


# ---------------------------------------------------------------------
# Primary workflow
# ---------------------------------------------------------------------
def load_nisar(inps):
    """Prepare and load NISAR data and metadata into HDF5/MintPy format."""
    print(f"update mode: {inps.update_mode}")

    input_files = sorted(glob.glob(inps.input_glob))
    print(f"Found {len(input_files)} unwrapped files")

    if inps.subset_lat:
        bbox = (inps.subset_lon[0], inps.subset_lat[0], inps.subset_lon[1], inps.subset_lat[1])
    else:
        bbox = None

    # extract metadata
    pol = getattr(inps, "polarization", "HH")
    metadata, bounds = extract_metadata(input_files, bbox=bbox, polarization=pol)

    # output filename
    stack_file = os.path.join(inps.out_dir, "inputs/ifgramStack.h5")
    geometry_file = os.path.join(inps.out_dir, "inputs/geometryGeo.h5")
    ion_stack_file = os.path.join(inps.out_dir, "inputs/ionStack.h5")
    tropo_stack_file = os.path.join(inps.out_dir, "inputs/tropoStack.h5")
    set_stack_file = os.path.join(inps.out_dir, "inputs/setStack.h5")

    # date pairs
    date12_list = _get_date_pairs(input_files)

    # geometry
    metadata = prepare_geometry(
        outfile=geometry_file,
        metaFile=input_files[0],
        bbox=bounds,
        metadata=metadata,
        demFile=inps.dem_file,
        maskFile=inps.mask_file,
        polarization=pol,
    )

    # ifgram stack
    prepare_stack(
        outfile=stack_file,
        inp_files=input_files,
        metadata=metadata,
        demFile=inps.dem_file,
        bbox=bounds,
        date12_list=date12_list,
        polarization=pol,
    )

    # ionosphere stack
    prepare_stack(
        outfile=ion_stack_file,
        inp_files=input_files,
        metadata=metadata,
        demFile=inps.dem_file,
        bbox=bounds,
        date12_list=date12_list,
        polarization=pol,
    )

    # troposphere stack
    prepare_stack(
        outfile=tropo_stack_file,
        inp_files=input_files,
        metadata=metadata,
        demFile=inps.dem_file,
        bbox=bounds,
        date12_list=date12_list,
        polarization=pol,
    )
    print("Done.")

    # SET stack
    prepare_stack(
        outfile=set_stack_file,
        inp_files=input_files,
        metadata=metadata,
        demFile=inps.dem_file,
        bbox=bounds,
        date12_list=date12_list,
        polarization=pol,
    )
    print("Done.")
    return


# ---------------------------------------------------------------------
# Metadata / subset utilities
# ---------------------------------------------------------------------
def extract_metadata(input_files, bbox=None, polarization="HH"):
    """Extract NISAR metadata for MintPy."""
    meta_file = input_files[0]
    meta = {}

    datasets = _datasets_for_pol(polarization)

    with h5py.File(meta_file, "r") as ds:
        pixel_height = ds[datasets["ySpacing"]][()]
        pixel_width = ds[datasets["xSpacing"]][()]
        x_origin = float(np.min(ds[datasets["xcoord"]][()]))
        y_origin = float(np.max(ds[datasets["ycoord"]][()]))
        xcoord = ds[datasets["xcoord"]][()]
        ycoord = ds[datasets["ycoord"]][()]
        meta["EPSG"] = int(ds[datasets["epsg"]][()])
        meta["WAVELENGTH"] = SPEED_OF_LIGHT / ds[PROCESSINFO["centerFrequency"]][()]
        meta["ORBIT_DIRECTION"] = ds[PROCESSINFO["orbit_direction"]][()].decode("utf-8")
        meta["POLARIZATION"] = polarization
        meta["ALOOKS"] = ds[datasets["azimuth_look"]][()]
        meta["RLOOKS"] = ds[datasets["range_look"]][()]
        meta["PLATFORM"] = ds[PROCESSINFO["platform"]][()].decode("utf-8")
        meta["STARTING_RANGE"] = float(np.min(ds[PROCESSINFO["rdr_slant_range"]][()].flatten()))

        start_time = datetime.datetime.strptime(
            ds[PROCESSINFO["start_time"]][()].decode("utf-8")[0:26], "%Y-%m-%dT%H:%M:%S.%f"
        )
        end_time = datetime.datetime.strptime(
            ds[PROCESSINFO["end_time"]][()].decode("utf-8")[0:26], "%Y-%m-%dT%H:%M:%S.%f"
        )

    t_mid = start_time + (end_time - start_time) / 2.0
    meta["CENTER_LINE_UTC"] = (t_mid - datetime.datetime(t_mid.year, t_mid.month, t_mid.day)).total_seconds()

    # These were previously using //2 (integer) which is wrong for float spacing.
    meta["X_FIRST"] = x_origin - float(pixel_width) / 2.0
    meta["Y_FIRST"] = y_origin - float(pixel_height) / 2.0
    meta["X_STEP"] = float(pixel_width)
    meta["Y_STEP"] = float(pixel_height)

    if meta["EPSG"] == 4326:
        meta["X_UNIT"] = meta["Y_UNIT"] = "degree"
    else:
        meta["X_UNIT"] = meta["Y_UNIT"] = "meters"
        if str(meta["EPSG"]).startswith("326"):
            meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + "N"
        else:
            meta["UTM_ZONE"] = str(meta["EPSG"])[3:] + "S"
    meta["EARTH_RADIUS"] = EARTH_RADIUS

    # NISAR altitude (kept as original)
    meta["HEIGHT"] = 747000

    # placeholder pixel sizes (kept as original)
    meta["RANGE_PIXEL_SIZE"] = abs(float(pixel_width))
    meta["AZIMUTH_PIXEL_SIZE"] = abs(float(pixel_height))

    # bbox handling
    if bbox:
        epsg_src = 4326
        utm_bbox = bbox_to_utm(bbox, meta["EPSG"], epsg_src)
    else:
        utm_bbox = None

    bounds = common_raster_bound(input_files, utm_bbox, polarization=polarization)
    meta["bbox"] = ",".join([str(b) for b in bounds])

    col1, row1, col2, row2 = get_rows_cols(xcoord, ycoord, bounds)
    meta["LENGTH"] = int(row2 - row1)
    meta["WIDTH"] = int(col2 - col1)

    return meta, bounds


def get_rows_cols(xcoord, ycoord, bounds):
    """Get row and cols of the bounding box to subset"""
    xindex = np.where(np.logical_and(xcoord >= bounds[0], xcoord <= bounds[2]))[0]
    yindex = np.where(np.logical_and(ycoord >= bounds[1], ycoord <= bounds[3]))[0]
    row1, row2 = int(np.min(yindex)), int(np.max(yindex))
    col1, col2 = int(np.min(xindex)), int(np.max(xindex))
    return col1, row1, col2, row2


def get_raster_corners(input_file, polarization="HH"):
    """Get the (west, south, east, north) bounds of the image."""
    datasets = _datasets_for_pol(polarization)
    with h5py.File(input_file, "r") as ds:
        xcoord = ds[datasets["xcoord"]][:]
        ycoord = ds[datasets["ycoord"]][:]
        west = max(np.min(ds[PROCESSINFO["rdr_xcoord"]][:]), np.min(xcoord))
        east = min(np.max(ds[PROCESSINFO["rdr_xcoord"]][:]), np.max(xcoord))
        north = min(np.max(ds[PROCESSINFO["rdr_ycoord"]][:]), np.max(ycoord))
        south = max(np.min(ds[PROCESSINFO["rdr_ycoord"]][:]), np.min(ycoord))
    return float(west), float(south), float(east), float(north)


def common_raster_bound(input_files, utm_bbox=None, polarization="HH"):
    """Get common bounds among all data"""
    x_bounds = []
    y_bounds = []
    for file in input_files:
        west, south, east, north = get_raster_corners(file, polarization=polarization)
        x_bounds.append([west, east])
        y_bounds.append([south, north])

    if utm_bbox is not None:
        x_bounds.append([utm_bbox[0], utm_bbox[2]])
        y_bounds.append([utm_bbox[1], utm_bbox[3]])

    bounds = max(x_bounds)[0], max(y_bounds)[0], min(x_bounds)[1], min(y_bounds)[1]
    return bounds


def bbox_to_utm(bbox, epsg_dst, epsg_src=4326):
    """Convert bbox to epsg_dst."""
    xmin, ymin, xmax, ymax = bbox
    t = Transformer.from_crs(epsg_src, epsg_dst, always_xy=True)
    xs = [xmin, xmax]
    ys = [ymin, ymax]
    xt, yt = t.transform(xs, ys)
    xys = list(zip(xt, yt))
    return (*xys[0], *xys[1])


def read_subset(inp_file, bbox, polarization="HH", geometry=False):
    """Read a subset of data using bounding box in rows and cols"""
    dataset = {}
    datasets = _datasets_for_pol(polarization)
    with h5py.File(inp_file, "r") as ds:
        xcoord = ds[datasets["xcoord"]][:]
        ycoord = ds[datasets["ycoord"]][:]
        col1, row1, col2, row2 = get_rows_cols(xcoord, ycoord, bbox)

        if geometry:
            dataset["xybbox"] = (col1, row1, col2, row2)
        else:
            dataset["unw_data"] = ds[datasets["unw"]][row1:row2, col1:col2]
            dataset["cor_data"] = ds[datasets["cor"]][row1:row2, col1:col2]
            dataset["conn_comp"] = ds[datasets["connComp"]][row1:row2, col1:col2].astype(np.float32)
            dataset["conn_comp"][dataset["conn_comp"] > 254] = np.nan
            dataset["ion_data"] = ds[datasets["ion"]][row1:row2, col1:col2]
            dataset["pbase"] = np.nanmean(ds[PROCESSINFO["bperp"]][()])
    return dataset


# ---------------------------------------------------------------------
# Resample + interpolate (DEM + geometry + tropo + SET)
# ---------------------------------------------------------------------
def read_and_interpolate_geometry(gunw_file, dem_file, xybbox, polarization="HH", mask_file=None):
    """
    Warp DEM to the interferogram grid (aligned), then interpolate slant range & incidence.
    Interpolation is evaluated at valid pixels only (validity from unwrappedPhase finite + _FillValue).
    """
    dem_src_epsg = _read_raster_epsg(dem_file)

    datasets = _datasets_for_pol(polarization)
    rdr_coords = {}

    with h5py.File(gunw_file, "r") as ds:
        dst_epsg = int(ds[datasets["epsg"]][()])
        xcoord = ds[datasets["xcoord"]][xybbox[0] : xybbox[2]]
        ycoord = ds[datasets["ycoord"]][xybbox[1] : xybbox[3]]

        rdr_coords["xcoord_radar_grid"] = ds[PROCESSINFO["rdr_xcoord"]][()]
        rdr_coords["ycoord_radar_grid"] = ds[PROCESSINFO["rdr_ycoord"]][()]
        rdr_coords["height_radar_grid"] = ds[PROCESSINFO["rdr_height"]][()]
        rdr_coords["slant_range"] = ds[PROCESSINFO["rdr_slant_range"]][()]
        rdr_coords["incidence_angle"] = ds[PROCESSINFO["rdr_incidence"]][()]

    # Warp DEM to exact grid
    dem_subset_array = _warp_to_grid_mem(
        src_path=dem_file,
        src_epsg=dem_src_epsg,
        dst_epsg=dst_epsg,
        xcoord=xcoord,
        ycoord=ycoord,
        resample_alg="bilinear",
    )

    # Build meshgrid in output CRS
    Y_2d, X_2d = np.meshgrid(ycoord, xcoord, indexing="ij")

    # Valid pixels from unwrappedPhase
    valid = _read_valid_unw_mask(gunw_file, xybbox, polarization)

    # Interpolate geometry at valid pixels only
    slant_range, incidence_angle = interpolate_geometry(X_2d, Y_2d, dem_subset_array, rdr_coords, valid)

    # Mask handling (optional external mask warped to grid; otherwise ones)
    if mask_file in ["auto", "None", None]:
        mask_subset_array = np.ones(dem_subset_array.shape, dtype="byte")
    else:
        mask_src_epsg = _read_raster_epsg(mask_file)
        mask_subset_array = _warp_to_grid_mem(
            src_path=mask_file,
            src_epsg=mask_src_epsg,
            dst_epsg=dst_epsg,
            xcoord=xcoord,
            ycoord=ycoord,
            resample_alg="near",
        ).astype("byte")

    return dem_subset_array, slant_range, incidence_angle, mask_subset_array


def interpolate_geometry(X_2d, Y_2d, dem, rdr_coords, valid_mask):
    """Interpolate slant range and incidence angle at valid pixels only."""
    length, width = Y_2d.shape
    out_slant = np.full((length, width), np.nan, dtype=np.float32)
    out_incid = np.full((length, width), np.nan, dtype=np.float32)

    ii, jj = np.where(valid_mask)
    if ii.size == 0:
        return out_slant, out_incid

    pts = np.column_stack(
        [
            dem[ii, jj].astype(np.float64),
            Y_2d[ii, jj].astype(np.float64),
            X_2d[ii, jj].astype(np.float64),
        ]
    )

    grid = (
        rdr_coords["height_radar_grid"],
        rdr_coords["ycoord_radar_grid"],
        rdr_coords["xcoord_radar_grid"],
    )

    slant_itp = _make_rgi(grid, rdr_coords["slant_range"], method="linear")
    inc_itp = _make_rgi(grid, rdr_coords["incidence_angle"], method="linear")

    sl = slant_itp(pts)
    inc = inc_itp(pts)

    out_slant[ii, jj] = sl.astype(np.float32)
    out_incid[ii, jj] = inc.astype(np.float32)
    return out_slant, out_incid


def read_and_interpolate_troposphere(gunw_file, dem_file, xybbox, polarization="HH", mask_file=None):
    """Warp DEM to aligned grid and interpolate combined tropo at valid pixels only."""
    dem_src_epsg = _read_raster_epsg(dem_file)
    datasets = _datasets_for_pol(polarization)
    rdr_coords = {}

    with h5py.File(gunw_file, "r") as ds:
        dst_epsg = int(ds[datasets["epsg"]][()])
        xcoord = ds[datasets["xcoord"]][xybbox[0] : xybbox[2]]
        ycoord = ds[datasets["ycoord"]][xybbox[1] : xybbox[3]]

        rdr_coords["xcoord_radar_grid"] = ds[PROCESSINFO["rdr_xcoord"]][()]
        rdr_coords["ycoord_radar_grid"] = ds[PROCESSINFO["rdr_ycoord"]][()]
        rdr_coords["height_radar_grid"] = ds[PROCESSINFO["rdr_height"]][()]
        rdr_coords["wet_tropo"] = ds[PROCESSINFO["rdr_wet_tropo"]][()]
        rdr_coords["hydrostatic_tropo"] = ds[PROCESSINFO["rdr_hs_tropo"]][()]

    dem_subset_array = _warp_to_grid_mem(
        src_path=dem_file,
        src_epsg=dem_src_epsg,
        dst_epsg=dst_epsg,
        xcoord=xcoord,
        ycoord=ycoord,
        resample_alg="bilinear",
    )

    Y_2d, X_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    valid = _read_valid_unw_mask(gunw_file, xybbox, polarization)

    total_tropo = interpolate_troposphere(X_2d, Y_2d, dem_subset_array, rdr_coords, valid)
    return total_tropo


def interpolate_troposphere(X_2d, Y_2d, dem, rdr_coords, valid_mask):
    """Interpolate total tropo (hydrostatic + wet) at valid pixels only."""
    length, width = Y_2d.shape
    out = np.full((length, width), np.nan, dtype=np.float32)

    ii, jj = np.where(valid_mask)
    if ii.size == 0:
        return out

    pts = np.column_stack(
        [
            dem[ii, jj].astype(np.float64),
            Y_2d[ii, jj].astype(np.float64),
            X_2d[ii, jj].astype(np.float64),
        ]
    )

    total = rdr_coords["hydrostatic_tropo"] + rdr_coords["wet_tropo"]
    grid = (
        rdr_coords["height_radar_grid"],
        rdr_coords["ycoord_radar_grid"],
        rdr_coords["xcoord_radar_grid"],
    )
    itp = _make_rgi(grid, total, method="linear")
    val = itp(pts)
    out[ii, jj] = val.astype(np.float32)
    return out


def read_and_interpolate_SET(gunw_file, dem_file, xybbox, polarization="HH", mask_file=None):
    """Warp DEM to aligned grid and interpolate SET phase at valid pixels only."""
    dem_src_epsg = _read_raster_epsg(dem_file)
    datasets = _datasets_for_pol(polarization)
    rdr_coords = {}

    with h5py.File(gunw_file, "r") as ds:
        dst_epsg = int(ds[datasets["epsg"]][()])
        xcoord = ds[datasets["xcoord"]][xybbox[0] : xybbox[2]]
        ycoord = ds[datasets["ycoord"]][xybbox[1] : xybbox[3]]

        rdr_coords["xcoord_radar_grid"] = ds[PROCESSINFO["rdr_xcoord"]][()]
        rdr_coords["ycoord_radar_grid"] = ds[PROCESSINFO["rdr_ycoord"]][()]
        rdr_coords["height_radar_grid"] = ds[PROCESSINFO["rdr_height"]][()]
        rdr_coords["rdr_SET"] = ds[PROCESSINFO["rdr_SET"]][()]

    dem_subset_array = _warp_to_grid_mem(
        src_path=dem_file,
        src_epsg=dem_src_epsg,
        dst_epsg=dst_epsg,
        xcoord=xcoord,
        ycoord=ycoord,
        resample_alg="bilinear",
    )

    Y_2d, X_2d = np.meshgrid(ycoord, xcoord, indexing="ij")
    valid = _read_valid_unw_mask(gunw_file, xybbox, polarization)

    set_phase = interpolate_set(X_2d, Y_2d, dem_subset_array, rdr_coords, valid)
    return set_phase


def interpolate_set(X_2d, Y_2d, dem, rdr_coords, valid_mask):
    """Interpolate SET phase at valid pixels only."""
    length, width = Y_2d.shape
    out = np.full((length, width), np.nan, dtype=np.float32)

    ii, jj = np.where(valid_mask)
    if ii.size == 0:
        return out

    pts = np.column_stack(
        [
            dem[ii, jj].astype(np.float64),
            Y_2d[ii, jj].astype(np.float64),
            X_2d[ii, jj].astype(np.float64),
        ]
    )

    grid = (
        rdr_coords["height_radar_grid"],
        rdr_coords["ycoord_radar_grid"],
        rdr_coords["xcoord_radar_grid"],
    )
    itp = _make_rgi(grid, rdr_coords["rdr_SET"], method="linear")
    val = itp(pts)
    out[ii, jj] = val.astype(np.float32)
    return out


# ---------------------------------------------------------------------
# MintPy file builders
# ---------------------------------------------------------------------
def _get_date_pairs(filenames):
    str_list = [Path(f).stem for f in filenames]
    return [
        str(f.split("_")[11].split("T")[0]) + "_" + str(f.split("_")[13].split("T")[0]) for f in str_list
    ]


def prepare_geometry(outfile, metaFile, metadata, bbox, demFile, maskFile, polarization="HH"):
    """Prepare the geometry file."""
    print("-" * 50)
    print(f"preparing geometry file: {outfile}")

    meta = {key: value for key, value in metadata.items()}

    geo_ds = read_subset(metaFile, bbox, polarization=polarization, geometry=True)
    dem_subset_array, slant_range, incidence_angle, mask = read_and_interpolate_geometry(
        metaFile,
        demFile,
        geo_ds["xybbox"],
        polarization=polarization,
        mask_file=maskFile,
    )

    length, width = dem_subset_array.shape
    ds_name_dict = {
        "height": [np.float32, (length, width), dem_subset_array],
        "incidenceAngle": [np.float32, (length, width), incidence_angle],
        "slantRangeDistance": [np.float32, (length, width), slant_range],
    }
    if maskFile:
        ds_name_dict["waterMask"] = [np.bool_, (length, width), mask.astype(bool)]

    meta["FILE_TYPE"] = "geometry"
    meta["STARTING_RANGE"] = float(np.nanmin(slant_range))
    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)
    return meta


def prepare_stack(outfile, inp_files, metadata, demFile, bbox, date12_list, polarization="HH"):
    """Prepare the input stacks."""
    print("-" * 50)
    print(f"preparing ifgramStack file: {outfile}")

    meta = {key: value for key, value in metadata.items()}
    num_pair = len(inp_files)
    print(f"number of inputs/unwrapped interferograms: {num_pair}")

    pbase = np.zeros(num_pair, dtype=np.float32)
    cols, rows = meta["WIDTH"], meta["LENGTH"]

    date12_arr = np.array([x.split("_") for x in date12_list], dtype=np.bytes_)
    drop_ifgram = np.ones(num_pair, dtype=np.bool_)

    ds_name_dict = {
        "date": [date12_arr.dtype, (num_pair, 2), date12_arr],
        "bperp": [np.float32, (num_pair,), pbase],
        "dropIfgram": [np.bool_, (num_pair,), drop_ifgram],
        "unwrapPhase": [np.float32, (num_pair, rows, cols), None],
        "coherence": [np.float32, (num_pair, rows, cols), None],
        "connectComponent": [np.float32, (num_pair, rows, cols), None],
    }

    if "inputs/geometryGeo.h5" in outfile:
        meta["FILE_TYPE"] = "geometry"
    else:
        meta["FILE_TYPE"] = "ifgramStack"

    writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)

    print(f"writing data to HDF5 file {outfile} with a mode ...")

    if "inputs/ifgramStack.h5" in outfile:
        with h5py.File(outfile, "a") as f:
            prog_bar = ptime.progressBar(maxValue=num_pair)
            for i, file in enumerate(inp_files):
                dataset = read_subset(file, bbox, polarization=polarization)
                f["unwrapPhase"][i] = dataset["unw_data"]
                f["coherence"][i] = dataset["cor_data"]
                f["connectComponent"][i] = dataset["conn_comp"]
                f["bperp"][i] = dataset["pbase"]
                prog_bar.update(i + 1, suffix=date12_list[i])
            prog_bar.close()

    elif "inputs/ionStack.h5" in outfile:
        with h5py.File(outfile, "a") as f:
            prog_bar = ptime.progressBar(maxValue=num_pair)
            for i, file in enumerate(inp_files):
                dataset = read_subset(file, bbox, polarization=polarization)
                f["unwrapPhase"][i] = dataset["ion_data"]
                f["coherence"][i] = dataset["cor_data"]
                f["connectComponent"][i] = dataset["conn_comp"]
                f["bperp"][i] = dataset["pbase"]
                prog_bar.update(i + 1, suffix=date12_list[i])
            prog_bar.close()

    elif "inputs/tropoStack.h5" in outfile:
        geo_ds = read_subset(inp_files[0], bbox, polarization=polarization, geometry=True)
        total_tropo = read_and_interpolate_troposphere(
            inp_files[0], demFile, geo_ds["xybbox"], polarization=polarization
        )
        with h5py.File(outfile, "a") as f:
            prog_bar = ptime.progressBar(maxValue=num_pair)
            for i, _file in enumerate(inp_files):
                f["unwrapPhase"][i] = total_tropo
                prog_bar.update(i + 1, suffix=date12_list[i])
            prog_bar.close()

    elif "inputs/setStack.h5" in outfile:
        geo_ds = read_subset(inp_files[0], bbox, polarization=polarization, geometry=True)
        set_phase = read_and_interpolate_SET(
            inp_files[0], demFile, geo_ds["xybbox"], polarization=polarization
        )
        with h5py.File(outfile, "a") as f:
            prog_bar = ptime.progressBar(maxValue=num_pair)
            for i, _file in enumerate(inp_files):
                f["unwrapPhase"][i] = set_phase
                prog_bar.update(i + 1, suffix=date12_list[i])
            prog_bar.close()

    print(f"finished writing to HDF5 file: {outfile}")
    return outfile
