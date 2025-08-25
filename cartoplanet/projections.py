import cartopy.crs as ccrs

from cartoplanet import config


body = config['BODY']['body']
R0 = float(config[body]['R0'])
g0 = float(config[body]['g0'])


GLOBE = ccrs.Globe(semimajor_axis=R0, semiminor_axis=R0)
PLATE_CARREE = ccrs.PlateCarree(central_longitude=0.0, globe=GLOBE)
projections = {}
projections.update({
  'PC_NS': PLATE_CARREE,
  'PC_FS': ccrs.PlateCarree(central_longitude=180.0, globe=GLOBE),
  'LAEA_NS': ccrs.LambertAzimuthalEqualArea(central_longitude=0, globe=GLOBE),
  'LAEA_FS': ccrs.LambertAzimuthalEqualArea(central_longitude=180.0, globe=GLOBE),
})
