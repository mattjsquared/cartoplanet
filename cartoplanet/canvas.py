# Canvas and Pane classes for figure layout


import matplotlib.pyplot as plt

class Canvas:
  def __init__(self, nrows=1, ncols=1, figsize=(6, 4)):
    self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
    # Flatten axes for easy indexing
    if nrows * ncols == 1:
      self.panes = [Pane(self.axes)]
    else:
      self.panes = [Pane(ax) for ax in self.axes.flat]

  def save(self, filename):
    self.fig.savefig(filename)

  def show(self):
    plt.show()

  def __getitem__(self, idx):
    return self.panes[idx]

class Pane:
  def __init__(self, ax):
    self.ax = ax

  def plot_demo(self):
    self.ax.plot([0, 1], [0, 1])  # Simple demo plot
