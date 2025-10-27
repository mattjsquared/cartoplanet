# Canvas and Pane classes for figure layout

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import warnings

from typing import Optional, Union, List, Tuple, Dict, Any
from cartoplanet.projections import GLOBE, PLATE_CARREE
from cartoplanet.types import ProjectionType, BoundaryType, KWType
from cartoplanet.layer import Layer
from cartoplanet.renderer import RENDERERS


class Canvas:
  def __init__(
      self,
      nrows: int = 1,
      ncols: int = 1,
      ax_height: float = 3,
      with_cax: bool = True,
      cax_orientation: str = 'horizontal',
      cax_ratio: float = 0.08,
      projections: ProjectionType = None,
      boundaries: BoundaryType = None,
      kw_fig: KWType = None,
      kw_gspec: KWType = None,
      kw_ax: KWType = None
  ) -> None:
    # Save metadata
    self.nrows = nrows
    self.ncols = ncols  
    self.ax_height = ax_height
    # Set default canvas parameters
    kw_fig = kw_fig or {}
    kw_gspec = kw_gspec or {}
    kw_ax = kw_ax or {}
    kw_fig.setdefault('facecolor', (1, 1, 1, 0))
    kw_gspec.setdefault('hspace', 0.05)
    kw_gspec.setdefault('wspace', -0.03)
    kw_ax.setdefault('facecolor', (1, 1, 1, 0))
    # Check inputs
    nax = nrows * ncols
    projections, boundaries = self._validate_inputs(nax, projections, boundaries, cax_orientation)
    # Define the canvas
    fig_h, fig_w, hratios, wratios, nrow_gspec, ncol_gspec = self._calculate_layout(nrows, ncols, ax_height, with_cax, cax_orientation, cax_ratio)
    self.fig, gspec = self._initialize_figure(fig_h, fig_w, nrow_gspec, ncol_gspec, kw_fig, kw_gspec, hratios, wratios)
    # Define the panes and optional colorbar axis
    projections = np.array(projections).reshape(nrows, ncols)
    boundaries = np.array(boundaries).reshape(nrows, ncols)
    self.panes = self._initialize_panes(nrows, ncols, projections, boundaries, gspec, kw_ax)
    self.cax = self._initialize_colorbar_axis(with_cax, cax_orientation, self.fig, gspec)

  def _validate_inputs(
      self,
      nax: int,
      projections: ProjectionType,
      boundaries: BoundaryType,
      cax_orientation: str
  ) -> Tuple[np.ndarray, np.ndarray]:
    # projections
    if projections is None:
      projections = PLATE_CARREE
    if isinstance(projections, ccrs.Projection):
      projections = [projections] * nax
    if not all(isinstance(p, ccrs.Projection) for p in projections):
      raise ValueError(f"Invalid projection(s) found: {projections}. All projections must be instances of cartopy.crs.Projection.")
    # boundaries
    if boundaries is None:
      boundaries = [None] * nax
    elif (len(boundaries) == nax) and all(b is None for b in boundaries):
      pass
    else:
      if isinstance(boundaries, mpl.path.Path):
        boundaries = [boundaries] * nax
      if not all(isinstance(b, mpl.path.Path) for b in boundaries):
        raise ValueError(f"Invalid boundary(s) found: {boundaries}. All boundaries must be instances of matplotlib.path.Path.")
    # colorbar axis
    if cax_orientation not in ['horizontal', 'vertical']:
      raise ValueError(f"Invalid colorbar orientation: {cax_orientation}. Must be either 'horizontal' or 'vertical'.")
    return projections, boundaries

  def _calculate_layout(
      self,
      nrows: int,
      ncols: int,
      ax_height: float,
      with_cax: bool,
      cax_orientation: str,
      cax_ratio: float
  ) -> Tuple[float, float, List[float], List[float], int, int]:
    fig_h = ax_height * nrows
    fig_w = ax_height * ncols
    hratios = [1] * nrows
    wratios = [1] * ncols
    nrow_gspec, ncol_gspec = nrows, ncols
    if with_cax:
      adj = cax_ratio * ax_height
      if cax_orientation == 'horizontal':
        fig_h += adj
        nrow_gspec += 1
        hratios.append(cax_ratio)
      else:
        fig_w += adj
        ncol_gspec += 1
        wratios.append(cax_ratio)
    return fig_h, fig_w, hratios, wratios, nrow_gspec, ncol_gspec

  def _initialize_figure(
      self,
      fig_h: float,
      fig_w: float,
      nrow_gspec: int,
      ncol_gspec: int,
      kw_fig: KWType,
      kw_gspec: KWType,
      hratios: List[float],
      wratios: List[float]
  ) -> Tuple[plt.Figure, mpl.gridspec.GridSpec]:
    # Warn if user-specified settings are ignored
    if kw_fig.get('figsize'):
      warnings.warn("User-specified 'figsize' is ignored; it is set automatically based on ax_height and layout.")
    if any(kw_gspec.get(k) is not None for k in ['height_ratios', 'width_ratios']):
      warnings.warn("User-specified 'height_ratios' and 'width_ratios' are ignored; they are set automatically based on nrows and colorbar.")
    # Define figure and gridspec layout
    kw_fig.update({'figsize': (fig_w, fig_h)})
    fig = plt.figure(**kw_fig)
    kw_gspec.update({'height_ratios': hratios, 'width_ratios': wratios})
    gspec = fig.add_gridspec(nrow_gspec, ncol_gspec, **kw_gspec)
    return fig, gspec

  def _initialize_panes(
      self,
      nrows: int,
      ncols: int,
      projections: np.ndarray,
      boundaries: np.ndarray,
      gspec: mpl.gridspec.GridSpec,
      kw_ax: KWType
  ) -> List[Pane]:
    panes = []
    for i in range(nrows):
      for j in range(ncols):
        proj = projections[i, j]
        ax = self.fig.add_subplot(gspec[i, j], projection=proj, **kw_ax)
        pane = Pane(
          row = i,
          col = j,
          ax = ax,
          gridspec = gspec,
          kw_ax = kw_ax,
          projection = proj,
          boundary = boundaries[i, j]
        )
        panes.append(pane)
    return panes

  def _initialize_colorbar_axis(
      self,
      with_cax: bool,
      cax_orientation: str,
      fig: plt.Figure,
      gspec: mpl.gridspec.GridSpec
  ) -> Optional[mpl.axes.Axes]:
    if not with_cax:
      return None
    if cax_orientation == 'horizontal':
      return fig.add_subplot(gspec[-1, :])
    else:
      return fig.add_subplot(gspec[:, -1])
  
  def get_axes(self) -> List[mpl.axes.Axes]:
    """
    Return a flat list of matplotlib Axes from all Panes.
    """
    return [pane.ax for pane in self.panes]

  def save(self, filename, kw_save=None):
    kw_save = kw_save or {}
    self.fig.savefig(filename, **kw_save)

  def show(self):
    plt.show()

  def __getitem__(self, idx):
    return self.panes[idx]

  def __len__(self):
    return len(self.panes)

  def __iter__(self):
    return iter(self.panes)


class Pane:
  def __init__(
      self,
      row: int,
      col: int,
      ax: mpl.axes.Axes,
      gridspec: mpl.gridspec.GridSpec,
      kw_ax: KWType,
      projection: ProjectionType,
      boundary: BoundaryType
  ) -> None:
    self._row = row
    self._col = col
    self._ax = ax
    self.__gridspec = gridspec
    self.kw_ax = kw_ax
    self._projection = projection
    self._boundary = boundary
    # Set boundary if provided
    if boundary is not None and projection is not None:
      self._set_boundary(boundary, projection)

  @property
  def row(self):
    return self._row

  @property
  def col(self):
    return self._col

  @property
  def ax(self):
    return self._ax

  @property
  def projection(self):
    return self._projection

  @projection.setter
  def projection(self, value):
    self._projection = value
    self._redraw_ax()

  @property
  def boundary(self):
    return self._boundary

  @boundary.setter
  def boundary(self, value):
    self._boundary = value
    self._redraw_ax()
  
  def update(
      self,
      projection: ProjectionType = None,
      boundary: BoundaryType = None
  ):
    if projection is not None:
      self._projection = projection
    if boundary is not None:
      self._boundary = boundary
    self._redraw_ax()

  def _redraw_ax(self):
    # Remove the old axes from the figure
    fig = self._ax.figure
    try:
      fig.delaxes(self._ax)
    except Exception:
      pass
    # Robustly recreate axes using stored gridspec, row, and col
    new_ax = fig.add_subplot(self.__gridspec[self._row, self._col], projection=self._projection, **self.kw_ax)
    self._ax = new_ax
    # Set boundary if available
    if self._boundary is not None and self._projection is not None:
      self._set_boundary(self._boundary, self._projection)

  def _set_boundary(
      self,
      boundary: BoundaryType,
      projection: ProjectionType
  ):
    clon = getattr(projection, 'central_longitude', None)
    if clon is None and hasattr(projection, 'proj4_params'):
      clon = projection.proj4_params.get('lon_0', 0)
    tf = ccrs.PlateCarree(central_longitude=clon, globe=GLOBE) if clon is not None else ccrs.PlateCarree(globe=GLOBE)
    self.ax.set_boundary(boundary, transform=tf)
    verts = boundary.vertices
    self.ax.set_extent([verts[:,0].min(), verts[:,0].max(), verts[:,1].min(), verts[:,1].max()], crs=tf)

  def draw(
      self,
      layer: Layer,
      **style
  ):
    renderer = RENDERERS.get(layer.kind)
    if renderer:
      return renderer.draw(layer=layer, ax=self.ax, **style)
    else:
      raise ValueError(f"Unknown layer kind: {layer.kind}")

  def plot_demo(self):
    # Generate a low-resolution grid
    lon = np.linspace(-180, 180, 30)
    lat = np.linspace(-90, 90, 15)
    Lon, Lat = np.meshgrid(lon, lat)
    Z = np.sin(np.radians(Lat)) * np.cos(np.radians(Lon))
    # Create GridLayer
    from cartoplanet.layer import GridLayer
    grid_layer = GridLayer(
      name="demo",
      data=Z,
      x=lat,
      y=lon,
      crs=PLATE_CARREE
    )
    # Plot using Pane.draw
    self.draw(grid_layer, cmap="viridis", shading='nearest')
    self.ax.set_title('Low-resolution grid demo')
