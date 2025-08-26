# Renderer base class and all renderer skeletons

class Renderer:
  def __init__(self):
    pass

  def draw(self, layer, pane, **style):
    raise NotImplementedError("Subclasses must implement draw().")


class GridRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, ax, **style):
    lon, lat = layer.get_coord_grids()
    return ax.pcolormesh(lon, lat, layer.data.values, transform=layer.crs, **style)


class PointRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, pane, **style):
    # Will implement point plotting logic here
    pass


class GeometryRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, pane, **style):
    # Will implement geometry plotting logic here
    pass


class SHRenderer(Renderer):
  def __init__(self):
    super().__init__()

  def draw(self, layer, pane, **style):
    # Will implement spherical harmonics plotting logic here
    pass


RENDERERS = {
    "grid": GridRenderer(),
    "point": PointRenderer(),
    "geometry": GeometryRenderer(),
    "sh": SHRenderer(),
}
