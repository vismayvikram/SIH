"""
Tiled raster service for serving GeoTIFF as XYZ/TMS map tiles using Rasterio and PIL.
Optimized for local dynamic rendering without copying huge raster files.
"""
import io
import math
import os
import warnings
from typing import Dict, Any, List, Optional, Tuple
import rasterio
from rasterio.windows import from_bounds
from PIL import Image
import numpy as np

# Suppress upstream rasterio 1.5 PendingDeprecationWarning on Python 3.14 (transform * window)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="rasterio")

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_GEOTIFF_PATH = os.path.join(
    WORKSPACE_ROOT, "data", "acquisition", "SIH26012_INDIA_CANDIDATE_01", "working", "lalpur_orthomosaic.tif"
)

ORIGIN_SHIFT = 20037508.342789244  # Earth circumference / 2 in EPSG:3857
TRANSPARENT_TILE_PNG: Optional[bytes] = None

def get_transparent_png() -> bytes:
    global TRANSPARENT_TILE_PNG
    if TRANSPARENT_TILE_PNG is None:
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        TRANSPARENT_TILE_PNG = buf.getvalue()
    return TRANSPARENT_TILE_PNG

def tile_bounds_3857(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Computes EPSG:3857 (minx, miny, maxx, maxy) bounds for Google/OSM XYZ tile coords.
    """
    initial_res = 2 * ORIGIN_SHIFT / 256.0
    res = initial_res / (2.0 ** z)
    minx = x * 256.0 * res - ORIGIN_SHIFT
    maxx = (x + 1) * 256.0 * res - ORIGIN_SHIFT
    maxy = ORIGIN_SHIFT - y * 256.0 * res
    miny = ORIGIN_SHIFT - (y + 1) * 256.0 * res
    return minx, miny, maxx, maxy

class RasterTileService:
    def __init__(
        self,
        tif_path: str = DEFAULT_GEOTIFF_PATH,
        rgb_band_mapping: Optional[List[int]] = None,
        alpha_band: Optional[int] = None,
    ):
        self.tif_path = tif_path
        self.rgb_band_mapping = rgb_band_mapping
        self.alpha_band = alpha_band
        self._src: Optional[rasterio.DatasetReader] = None
        self.is_available = False
        self.metadata: Dict[str, Any] = {}
        self.init_raster()

    def init_raster(self):
        if not os.path.exists(self.tif_path):
            self.is_available = False
            self.metadata = {
                "available": False,
                "error": f"GeoTIFF file not found at {self.tif_path}",
                "path": self.tif_path
            }
            return

        try:
            with rasterio.open(self.tif_path) as src:
                self.metadata = {
                    "available": True,
                    "filename": os.path.basename(self.tif_path),
                    "path": self.tif_path,
                    "crs": str(src.crs),
                    "bounds_3857": {
                        "left": src.bounds.left,
                        "bottom": src.bounds.bottom,
                        "right": src.bounds.right,
                        "top": src.bounds.top
                    },
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "dtypes": [str(d) for d in src.dtypes],
                    "file_size_bytes": os.path.getsize(self.tif_path),
                    "min_zoom": 14,
                    "max_zoom": 21
                }
                self.is_available = True
        except Exception as e:
            self.is_available = False
            self.metadata = {
                "available": False,
                "error": str(e),
                "path": self.tif_path
            }

    def _get_dataset(self) -> Optional[rasterio.DatasetReader]:
        if not self.is_available:
            return None
        if self._src is None or self._src.closed:
            self._src = rasterio.open(self.tif_path)
        return self._src

    def get_tile_png(self, z: int, x: int, y: int, *, strict: bool = False) -> bytes:
        """
        Renders a 256x256 PNG tile for the given XYZ coordinates.
        Returns a transparent PNG if tile does not intersect raster bounds or if raster is unavailable.
        """
        if not self.is_available:
            return get_transparent_png()

        src = self._get_dataset()
        if src is None:
            return get_transparent_png()

        tb = src.bounds
        minx, miny, maxx, maxy = tile_bounds_3857(z, x, y)

        # Check bounding box intersection
        if maxx < tb.left or minx > tb.right or maxy < tb.bottom or miny > tb.top:
            return get_transparent_png()

        try:
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            if self.rgb_band_mapping:
                indexes = list(self.rgb_band_mapping)
                if self.alpha_band:
                    indexes.append(self.alpha_band)
                data = src.read(
                    indexes,
                    window=window,
                    out_shape=(len(indexes), 256, 256),
                    boundless=True,
                    fill_value=0,
                    masked=True,
                )
                rgb = np.transpose(np.ma.filled(data[:3], 0), (1, 2, 0)).astype(np.uint8)
                invalid = np.ma.getmaskarray(data[:3]).any(axis=0)
                if self.alpha_band:
                    alpha = np.ma.filled(data[3], 0).astype(np.uint8)
                    alpha[invalid] = 0
                else:
                    alpha = np.where(~invalid & np.any(rgb > 0, axis=-1), 255, 0).astype(np.uint8)
                img = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                return buf.getvalue()

            # Read 256x256 image window
            data = src.read(window=window, out_shape=(src.count, 256, 256), boundless=True, fill_value=0)

            # If all zeros (empty padding outside raster border), return transparent
            if not np.any(data):
                return get_transparent_png()

            if src.count >= 4:
                # RGBA
                rgba = np.transpose(data[:4], (1, 2, 0)).astype(np.uint8)
                img = Image.fromarray(rgba, mode='RGBA')
            elif src.count == 3:
                # RGB - add alpha where all channels are 0
                rgb = np.transpose(data[:3], (1, 2, 0)).astype(np.uint8)
                mask = np.any(rgb > 0, axis=-1).astype(np.uint8) * 255
                rgba = np.dstack((rgb, mask))
                img = Image.fromarray(rgba, mode='RGBA')
            else:
                # Single band grayscale
                gray = data[0].astype(np.uint8)
                mask = (gray > 0).astype(np.uint8) * 255
                rgba = np.dstack((gray, gray, gray, mask))
                img = Image.fromarray(rgba, mode='RGBA')

            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            return buf.getvalue()
        except Exception:
            if strict:
                raise
            return get_transparent_png()

# Singleton instance
raster_tile_service = RasterTileService()
