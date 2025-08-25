import cartopy.crs as ccrs
import matplotlib as mpl

from typing import Optional, Union, List, Dict, Any

ProjectionType = Optional[Union[ccrs.Projection, List[Optional[ccrs.Projection]]]]
BoundaryType = Optional[Union[mpl.path.Path, List[Optional[mpl.path.Path]]]]
KWType = Optional[Dict[str, Any]]

