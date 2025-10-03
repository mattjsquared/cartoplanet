import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartoplanet.layer import Layer, GridLayer, PointLayer, GeometryLayer, SHLayer

def test_layer_basic():
    l = Layer(name='foo', kind='bar', data=123)
    assert l.name == 'foo'
    assert l.kind == 'bar'
    assert l.data == 123

def test_gridlayer_numpy():
    arr = np.arange(6).reshape(2,3)
    x = np.array([10, 20, 30])
    y = np.array([1, 2])
    layer = GridLayer('test', arr, x, y)
    assert layer.data.shape == (2,3)
    assert np.all(layer.coords['x'] == x)
    assert np.all(layer.coords['y'] == y)

def test_gridlayer_xarray():
    arr = xr.DataArray(np.arange(6).reshape(2,3), coords={'lon': [10, 20, 30], 'lat': [1, 2]}, dims=('lat','lon'))
    layer = GridLayer('test', arr, 'lon', 'lat', xname='x', yname='y')
    assert layer.data.shape == (2,3)
    assert np.all(layer.coords['x'] == [10, 20, 30])
    assert np.all(layer.coords['y'] == [1, 2])

def test_gridlayer_coord_grids_numpy():
    arr = np.arange(6).reshape(2,3)
    x = np.array([10, 20, 30])
    y = np.array([1, 2])
    layer = GridLayer('test', arr, x, y)
    X, Y = layer.get_coord_grids()
    assert X.shape == (2,3)
    assert Y.shape == (2,3)
    assert np.all(X[0] == x)
    assert np.all(Y[:,0] == y)

def test_gridlayer_coord_grids_xarray():
    arr = xr.DataArray(np.arange(6).reshape(2,3), coords={'lon': [10, 20, 30], 'lat': [1, 2]}, dims=('lat','lon'))
    layer = GridLayer('test', arr, 'lon', 'lat', xname='x', yname='y')
    X, Y = layer.get_coord_grids()
    assert X.shape == (2,3)
    assert Y.shape == (2,3)
    assert np.all(X[0] == [10, 20, 30])
    assert np.all(Y[:,0] == [1, 2])

def test_gridlayer_coord_grids_xarray_broadcast():
    arr = xr.DataArray(np.arange(6).reshape(2,3), coords={'lon': [10, 20, 30], 'lat': [1, 2]}, dims=('lat','lon'))
    layer = GridLayer('test', arr, 'lon', 'lat', xname='x', yname='y')
    X, Y = layer.get_coord_grids(use_xarray=True)
    assert isinstance(X, xr.DataArray)
    assert isinstance(Y, xr.DataArray)
    assert X.shape == (2,3)
    assert Y.shape == (2,3)

def test_gridlayer_rename_coords():
    arr = np.arange(6).reshape(2,3)
    x = np.array([10, 20, 30])
    y = np.array([1, 2])
    layer = GridLayer('test', arr, x, y)
    layer.xname = 'longitude'
    layer.yname = 'latitude'
    assert 'longitude' in layer.coords
    assert 'latitude' in layer.coords

def test_pointlayer_basic():
    arr = np.array([[1,2],[3,4],[5,6]])
    layer = PointLayer('pt', arr)
    assert np.all(layer.x == [1,3,5])
    assert np.all(layer.y == [2,4,6])
    df = pd.DataFrame({'x':[1,2],'y':[3,4],'value':[5,6]})
    layer2 = PointLayer('pt2', df)
    assert np.all(layer2.x == [1,2])
    assert np.all(layer2.value == [5,6])

def test_geometrylayer_geodataframe():
    gdf = gpd.GeoDataFrame({'geometry':[sgeom.Point(0,0), sgeom.Point(1,1)]})
    layer = GeometryLayer('geom', gdf)
    assert len(layer.geometry) == 2

def test_geometrylayer_shapely():
    pt = sgeom.Point(0,0)
    layer = GeometryLayer('geom', pt)
    assert len(layer.geometry) == 1

def test_shlayer():
    arr = np.arange(10)
    layer = SHLayer('sh', arr)
    try:
        _ = layer.crs
    except AttributeError:
        pass
    else:
        assert False, "SHLayer.crs should raise AttributeError"
