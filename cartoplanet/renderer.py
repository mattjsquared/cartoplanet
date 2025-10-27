import matplotlib as mpl
import geopandas as gpd
import cartopy
import inspect, pickle, warnings

from shapely.geometry import Polygon
from typing import Union
from cartoplanet import config
from cartoplanet.layer import GridLayer, PointLayer, GeometryLayer, SHLayer
from cartoplanet.projections import PLATE_CARREE

with open(config["PATH"]["hillshade"], 'rb') as f:
  topo_layer = pickle.load(f)


class Renderer:
  def __init__(self):
    pass

  def draw(self, layer, ax, **style):
    raise AttributeError(f"`draw()` is not implemented for objects of type {type(layer)}.")
  
  def get_style_kws(self, func):
    return inspect.signature(func).parameters

class GridRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, ax, shaded=False, rasterized=True, **style):
    xgrid, ygrid = layer.get_coord_grids()
    if not shaded:
      return ax.pcolormesh(xgrid, ygrid, layer.data.values, transform=layer.crs, rasterized=rasterized, **style)
    else:
      # Regrid data and topo to common (lower) resolution
      from cartoplanet.utils import regrid_to_lowest_resolution
      topo_regrid, data_regrid = regrid_to_lowest_resolution(topo_layer.data, layer.data)
      # Compute unshaded data RGBA
      norm = style.get('norm', mpl.colors.Normalize(vmin=style.get('vmin', layer.values.min()), vmax=style.get('vmax', layer.values.max())))
      colorizer = style.get('colorizer', mpl.colorizer.Colorizer(style.get('cmap'), norm))
      colorizer.autoscale_None(layer.values)
      data_rgba = colorizer.to_rgba(data_regrid.values)
      # Apply hillshading
      ls = mpl.colors.LightSource(azdeg=315, altdeg=45)
      data_rgba = ls.shade_rgb(
        data_rgba, 
        topo_regrid.values, 
        blend_mode=style.get('blend_mode', 'overlay'), 
        vert_exag=style.get('vert_exag', 1), 
        fraction=style.get('fraction', 1)
      )
      return ax.imshow(data_rgba, transform=layer.crs, rasterized=rasterized, **style)


class PointRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, ax, **style):
    return ax.scatter(layer.x, layer.y, transform=layer.crs, **style)


class GeometryRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(
      self, 
      layer: GeometryLayer, 
      ax: cartopy.mpl.geoaxes.GeoAxes, 
      dissolved: bool = False,
      simplified: Union[bool, float] = 1,
      **style
  ):
    gdf = layer.data
    if dissolved:
      gdf = self.dissolve_geometry(gdf)
    if simplified:
      if isinstance(simplified, bool):
        tolerance = 1
      else:
        tolerance = simplified
      gdf = gdf.simplify(tolerance=tolerance, preserve_topology=False)
    return ax.add_geometries(gdf.to_crs(PLATE_CARREE).geometry, crs=PLATE_CARREE, **style)
  
  def dissolve_geometry(
      self,
      gdf: gpd.GeoDataFrame,
  ) -> gpd.GeoDataFrame:
    dissolved = gpd.GeoDataFrame(
      geometry = [gdf.unary_union],
      crs = gdf.crs
    )
    return dissolved


class SHRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, ax, **style):
    raise NotImplementedError("Drawing directly from `SHLayer` is not implemented yet. Convert to another type, such as `GridLayer`, first. (E.g., `layer.to_grid()`)")


RENDERERS = {
    "grid": GridRenderer(),
    "point": PointRenderer(),
    "geometry": GeometryRenderer(),
    "sh": SHRenderer(),
}
