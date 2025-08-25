# Entry point for cartoplanet CLI (if needed)

def main(args):
  from cartoplanet.canvas import Canvas
  from cartoplanet.projections import projections
  from cartoplanet.boundaries import boundaries

  canvas = Canvas(
    nrows=args.nrows, ncols=args.ncols,
    with_cax=True
  )
  # canvas[0].plot_demo()
  canvas.save(args.filename[0])
  proj = [projections['LAEA_NS'], projections['LAEA_FS']]*2
  bound = [boundaries['limb_circle'], boundaries['limb_circle']]*2
  for i, pane in enumerate(canvas):
    pane.update(projection=proj[i], boundary=bound[i])
  # canvas[0].plot_demo()
  canvas.save(args.filename[1])

if __name__ == "__main__":
  import argparse, os
  p = argparse.ArgumentParser(description="CartoPlanet CLI")
  p.add_argument("--nrows", type=int, default=1, help="Number of rows")
  p.add_argument("--ncols", type=int, default=2, help="Number of columns")
  img_dir = os.path.dirname(os.path.abspath(__file__)) + "/../img"
  default_filenames = [
    f"{img_dir}/cartoplanet_test.png",
    f"{img_dir}/cartoplanet_test2.png"
  ]
  p.add_argument("--out", type=str, nargs='*', default=default_filenames, help="Output filename(s)")
  main(p.parse_args())
