import numpy as np
import matplotlib as mpl

def generate_limb_circle():
    """
    Generate a Path representing a great circle (rectangle in lon/lat) boundary for global map axes.
    Returns
    -------
    mpl.path.Path
        Interpolated path for the limb (rectangle/great circle)
    """
    rect = mpl.path.Path([[-90, -90], [90, -90], [90, 90], [-90, 90], [-90, -90]]).interpolated(100)
    return rect


boundaries = {}
boundaries.update({
    'limb_circle': generate_limb_circle(),
})


