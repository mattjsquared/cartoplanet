# Entry point for cartoplanet CLI (if needed)

def main():
  from cartoplanet.canvas import Canvas

  canvas = Canvas(nrows=1, ncols=1)
  canvas[0].plot_demo()
  canvas.show()

if __name__ == "__main__":
  main()
