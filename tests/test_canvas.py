import pytest
import matplotlib as mpl
from cartoplanet.canvas import Canvas, Pane
from cartoplanet.projections import PLATE_CARREE, projections
from cartoplanet.boundaries import generate_limb_circle, boundaries

SHOW_FIGS = False
if not SHOW_FIGS:
  mpl.use('Agg')

# Basic Canvas creation
def test_canvas_single_pane():
  canvas = Canvas(nrows=1, ncols=1)
  assert len(canvas.panes) == 1
  assert isinstance(canvas.panes[0], Pane)
  assert isinstance(canvas.panes[0].ax, mpl.axes.Axes)
  assert len(canvas) == 1
  assert canvas[0] is canvas.panes[0]

# Multiple panes
def test_canvas_multiple_panes():
  canvas = Canvas(nrows=2, ncols=2)
  assert len(canvas.panes) == 4
  axes = canvas.get_axes()
  assert len(axes) == 4
  for ax in axes:
    assert isinstance(ax, mpl.axes.Axes)
  # __iter__
  assert list(canvas) == canvas.panes

# Custom projections
def test_canvas_custom_projection():
  canvas = Canvas(nrows=1, ncols=2, projections=[PLATE_CARREE, PLATE_CARREE])
  assert len(canvas.panes) == 2
  for pane in canvas.panes:
    assert pane.projection is not None

# Boundaries
def test_canvas_with_boundaries():
  boundary = generate_limb_circle()
  canvas = Canvas(nrows=1, ncols=1, boundaries=[boundary])
  assert canvas.panes[0].boundary is boundary
  # Boundary validity
  assert hasattr(boundary, 'vertices')
  assert boundary.vertices.shape[1] == 2

# Colorbar axis
def test_canvas_colorbar_axis_horizontal():
  canvas = Canvas(nrows=1, ncols=2, with_cax=True, cax_orientation='horizontal')
  assert canvas.cax is not None
  assert isinstance(canvas.cax, mpl.axes.Axes)

def test_canvas_colorbar_axis_vertical():
  canvas = Canvas(nrows=2, ncols=1, with_cax=True, cax_orientation='vertical')
  assert canvas.cax is not None
  assert isinstance(canvas.cax, mpl.axes.Axes)

# Error handling
def test_invalid_projection_type():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=1, projections=['not_a_projection'])

def test_invalid_boundary_type():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=1, boundaries=['not_a_path'])

def test_invalid_colorbar_orientation():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=1, with_cax=True, cax_orientation='diagonal')

# __getitem__
def test_canvas_getitem():
  canvas = Canvas(nrows=1, ncols=2)
  assert canvas[0] is canvas.panes[0]
  assert canvas[1] is canvas.panes[1]
  assert len(canvas) == 2
  assert list(canvas) == canvas.panes

def test_pane_plot_demo_runs():
  canvas = Canvas(nrows=1, ncols=1)
  pane = canvas.panes[0]
  pane.plot_demo()  # Should not raise

# Canvas.show and save do not error

def test_canvas_show_and_save_runs(tmp_path):
  canvas = Canvas(nrows=1, ncols=1)
  canvas.panes[0].plot_demo()
  canvas.show()  # Should not raise
  out_file = tmp_path / "test_output.png"
  canvas.save(str(out_file))
  assert out_file.exists()

def test_canvas_zero_nrows():
  with pytest.raises(ValueError):
    Canvas(nrows=0, ncols=1)

def test_canvas_zero_ncols():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=0)

def test_canvas_negative_nrows():
  with pytest.raises(ValueError):
    Canvas(nrows=-1, ncols=1)

def test_canvas_negative_ncols():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=-1)

def test_canvas_missing_cax_orientation():
  # Should default to horizontal
  canvas = Canvas(nrows=1, ncols=2, with_cax=True)
  assert canvas.cax is not None

def test_canvas_invalid_cax_orientation():
  with pytest.raises(ValueError):
    Canvas(nrows=1, ncols=2, with_cax=True, cax_orientation='invalid')

def test_canvas_cax_height_difference():
  nrows, ncols = 1, 2
  cax_ratio = 0.05
  canvas_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=True, cax_orientation='horizontal', cax_ratio=cax_ratio)
  canvas_no_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=False)
  fig_cax = canvas_cax.fig
  fig_no_cax = canvas_no_cax.fig
  assert fig_cax.get_figwidth() == fig_no_cax.get_figwidth()
  height_no_cax = fig_no_cax.get_figheight()
  height_cax = fig_cax.get_figheight()
  ax_height = height_no_cax / nrows
  expected_height_cax = height_no_cax + cax_ratio * ax_height
  assert abs(height_cax - expected_height_cax) < 1e-6

def test_canvas_cax_height_difference_vertical():
  nrows, ncols = 2, 1
  cax_ratio = 0.05
  canvas_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=True, cax_orientation='vertical', cax_ratio=cax_ratio)
  canvas_no_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=False)
  fig_cax = canvas_cax.fig
  fig_no_cax = canvas_no_cax.fig
  assert fig_cax.get_figheight() == fig_no_cax.get_figheight()
  width_no_cax = fig_no_cax.get_figwidth()
  width_cax = fig_cax.get_figwidth()
  ax_width = width_no_cax / ncols
  expected_width_cax = width_no_cax + cax_ratio * ax_width
  assert abs(width_cax - expected_width_cax) < 1e-6

def test_canvas_dimensions_unchanged_after_update():
  canvas = Canvas(nrows=1, ncols=2, with_cax=True)
  initial_width = canvas.fig.get_figwidth()
  initial_height = canvas.fig.get_figheight()
  new_proj = [projections['LAEA_NS'], projections['LAEA_FS']]
  new_bound = [boundaries['limb_circle']] * 2
  for i, pane in enumerate(canvas):
    pane.update(projection=new_proj[i], boundary=new_bound[i])
  assert canvas.fig.get_figwidth() == initial_width
  assert canvas.fig.get_figheight() == initial_height

def test_canvas_dimensions_unchanged_after_update_multiple():
  canvas = Canvas(nrows=2, ncols=2, with_cax=True)
  initial_width = canvas.fig.get_figwidth()
  initial_height = canvas.fig.get_figheight()
  new_proj = [projections['LAEA_NS'], projections['LAEA_FS']] * 2
  new_bound = [boundaries['limb_circle']] * 4
  for i, pane in enumerate(canvas):
    pane.update(projection=new_proj[i], boundary=new_bound[i])
  assert canvas.fig.get_figwidth() == initial_width
  assert canvas.fig.get_figheight() == initial_height

def test_ax_dimensions_unchanged_after_update():
  canvas = Canvas(nrows=1, ncols=2, with_cax=True)
  ax_dims_before = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
  new_proj = [projections['LAEA_NS'], projections['LAEA_FS']]
  new_bound = [boundaries['limb_circle']] * 2
  for i, pane in enumerate(canvas):
    pane.update(projection=new_proj[i], boundary=new_bound[i])
  ax_dims_after = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
  assert ax_dims_before == ax_dims_after

def test_ax_dimensions_unchanged_after_update_multiple():
  canvas = Canvas(nrows=2, ncols=2, with_cax=True)
  ax_dims_before = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
  new_proj = [projections['LAEA_NS'], projections['LAEA_FS']] * 2
  new_bound = [boundaries['limb_circle']] * 4
  for i, pane in enumerate(canvas):
    pane.update(projection=new_proj[i], boundary=new_bound[i])
  ax_dims_after = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
  assert ax_dims_before == ax_dims_after

def test_pane_draw_gridlayer():
  from cartoplanet.layer import GridLayer
  import numpy as np
  lon = np.linspace(-180, 180, 10)
  lat = np.linspace(-90, 90, 5)
  Lon, Lat = np.meshgrid(lon, lat)
  Z = np.sin(np.radians(Lat)) * np.cos(np.radians(Lon))
  grid_layer = GridLayer(
    name="demo",
    data=Z,
    x=lat,
    y=lon,
    crs=PLATE_CARREE
  )
  canvas = Canvas(nrows=1, ncols=1)
  pane = canvas.panes[0]
  pane.draw(grid_layer, cmap="viridis")
  pane.ax.set_title('GridLayer draw test')

  # Edge case: zero and negative nrows/ncols
  def test_canvas_zero_nrows():
    with pytest.raises(ValueError):
      Canvas(nrows=0, ncols=1)

  def test_canvas_zero_ncols():
    with pytest.raises(ValueError):
      Canvas(nrows=1, ncols=0)

  def test_canvas_negative_nrows():
    with pytest.raises(ValueError):
      Canvas(nrows=-1, ncols=1)

  def test_canvas_negative_ncols():
    with pytest.raises(ValueError):
      Canvas(nrows=1, ncols=-1)

  # Missing cax orientation (should default or raise)
  def test_canvas_missing_cax_orientation():
    try:
      canvas = Canvas(nrows=1, ncols=2, with_cax=True)
      assert canvas.cax is not None
    except ValueError:
      pass

  # Invalid cax orientation
  def test_canvas_invalid_cax_orientation():
    with pytest.raises(ValueError):
      Canvas(nrows=1, ncols=2, with_cax=True, cax_orientation='invalid')

  # Figure size comparison with/without cax
  def test_canvas_cax_height_difference():
    nrows, ncols = 1, 2
    cax_ratio = 0.05
    canvas_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=True, cax_orientation='horizontal', cax_ratio=cax_ratio)
    canvas_no_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=False)
    fig_cax = canvas_cax.fig
    fig_no_cax = canvas_no_cax.fig
    assert fig_cax.get_figwidth() == fig_no_cax.get_figwidth()
    height_no_cax = fig_no_cax.get_figheight()
    height_cax = fig_cax.get_figheight()
    ax_height = height_no_cax / nrows
    expected_height_cax = height_no_cax + cax_ratio * ax_height
    assert abs(height_cax - expected_height_cax) < 1e-6

  # Figure size comparison with/without vertical cax
  def test_canvas_cax_height_difference_vertical():
    nrows, ncols = 2, 1
    cax_ratio = 0.05
    canvas_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=True, cax_orientation='vertical', cax_ratio=cax_ratio)
    canvas_no_cax = Canvas(nrows=nrows, ncols=ncols, with_cax=False)
    fig_cax = canvas_cax.fig
    fig_no_cax = canvas_no_cax.fig
    # Heights should be equal
    assert fig_cax.get_figheight() == fig_no_cax.get_figheight()
    width_no_cax = fig_no_cax.get_figwidth()
    width_cax = fig_cax.get_figwidth()
    ax_width = width_no_cax / ncols
    expected_width_cax = width_no_cax + cax_ratio * ax_width
    assert abs(width_cax - expected_width_cax) < 1e-6

  # Canvas dimensions remain unchanged after updating boundaries and/or projections
  def test_canvas_dimensions_unchanged_after_update():
    canvas = Canvas(nrows=1, ncols=2, with_cax=True)
    initial_width = canvas.fig.get_figwidth()
    initial_height = canvas.fig.get_figheight()
    # Update projections and boundaries
    from cartoplanet.projections import projections
    from cartoplanet.boundaries import boundaries
    new_proj = [projections['LAEA_NS'], projections['LAEA_FS']]
    new_bound = [boundaries['limb_circle']] * 2
    for i, pane in enumerate(canvas):
      pane.update(projection=new_proj[i], boundary=new_bound[i])
    # Dimensions should remain unchanged
    assert canvas.fig.get_figwidth() == initial_width
    assert canvas.fig.get_figheight() == initial_height

  def test_canvas_dimensions_unchanged_after_update_multiple():
    canvas = Canvas(nrows=2, ncols=2, with_cax=True)
    initial_width = canvas.fig.get_figwidth()
    initial_height = canvas.fig.get_figheight()
    from cartoplanet.projections import projections
    from cartoplanet.boundaries import boundaries
    new_proj = [projections['LAEA_NS'], projections['LAEA_FS']] * 2
    new_bound = [boundaries['limb_circle']] * 4
    for i, pane in enumerate(canvas):
      pane.update(projection=new_proj[i], boundary=new_bound[i])
    assert canvas.fig.get_figwidth() == initial_width
    assert canvas.fig.get_figheight() == initial_height

  # Axes dimensions remain unchanged after updating boundaries and/or projections
  def test_ax_dimensions_unchanged_after_update():
    canvas = Canvas(nrows=1, ncols=2, with_cax=True)
    ax_dims_before = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
    from cartoplanet.projections import projections
    from cartoplanet.boundaries import boundaries
    new_proj = [projections['LAEA_NS'], projections['LAEA_FS']]
    new_bound = [boundaries['limb_circle']] * 2
    for i, pane in enumerate(canvas):
      pane.update(projection=new_proj[i], boundary=new_bound[i])
    ax_dims_after = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
    assert ax_dims_before == ax_dims_after

  def test_ax_dimensions_unchanged_after_update_multiple():
    canvas = Canvas(nrows=2, ncols=2, with_cax=True)
    ax_dims_before = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
    from cartoplanet.projections import projections
    from cartoplanet.boundaries import boundaries
    new_proj = [projections['LAEA_NS'], projections['LAEA_FS']] * 2
    new_bound = [boundaries['limb_circle']] * 4
    for i, pane in enumerate(canvas):
      pane.update(projection=new_proj[i], boundary=new_bound[i])
    ax_dims_after = [(pane.ax.get_position().width, pane.ax.get_position().height) for pane in canvas]
    assert ax_dims_before == ax_dims_after
