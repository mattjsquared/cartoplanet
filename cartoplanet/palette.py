import matplotlib.pyplot as plt
import matplotlib as mpl

from typing import Optional, Union, Any


def _safeset(obj: Any, key: str, value: Any):
  if hasattr(obj, key):
    if getattr(obj, key) == value:
      return
    else:
      raise AttributeError(f"Attribute '{key}' is already defined for this object.")
  setattr(obj, key, value)
  return

class Palette:
  def __init__(
      self,
      spec: Optional[Any] = None,
      **kwargs
  ):
    spec = spec or {}
    if isinstance(spec, Palette):
      for k, v, in spec.__dict__.items():
        setattr(self, k, v)
    elif isinstance(spec, dict):
      for k, v in spec.items():
        setattr(self, k, v)
    elif spec is not None:
      raise TypeError("Palette positional argument (`spec`) must be a `Palette` object, a `dict`, or `None`.")
    for k, v in kwargs.items():
      setattr(self, k, v)
    self._fill_attrs()
  
  def _fill_attrs(self):
    if 'colorizer' in self:
      colorizer = self.get('colorizer')
      _safeset(self, 'cmap', colorizer.cmap)
      _safeset(self, 'norm', colorizer.norm)
      _safeset(self, 'vmin', colorizer.vmin)
      _safeset(self, 'vmax', colorizer.vmax)
    elif 'norm' in self:
      norm = self.get('norm')
      _safeset(self, 'vmin', norm.vmin)
      _safeset(self, 'vmax', norm.vmax)
      _safeset(self, 'colorizer', mpl.colorizer.Colorizer(self.get('cmap'), norm))
    else:
      _safeset(self, 'norm', mpl.colors.Normalize(vmin=self.get('vmin'), vmax=self.get('vmax')))
      _safeset(self, 'colorizer', mpl.colorizer.Colorizer(self.get('cmap'), self.get('norm')))

  def get(self, key, default=None):
    return getattr(self, key, default)

  def __contains__(self, key):
    return hasattr(self, key)
