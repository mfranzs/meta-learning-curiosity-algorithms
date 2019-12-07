import numpy as np
import matplotlib.pyplot as plt

def scatter_heatmap(x, y, bins=50):
  heatmap, xedges, yedges = np.histogram2d(np.array(x), np.array(y), bins=50)
  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

  plt.clf()
  plt.imshow(heatmap.T, extent=extent, origin='lower')

# Based on, but modified from:
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/scatter_hist.html
def scatter_histogram_histogram(
    x, y, binwidth, xlabel = None, ylabel = None
):
  x, y = np.array(x), np.array(y)

  # definitions for the axes
  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.005

  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]

  # start with a rectangular Figure
  plt.figure(figsize=(8, 8))

  ax_scatter = plt.axes(rect_scatter, xlabel = xlabel, ylabel = ylabel)
  ax_scatter.tick_params(direction='in', top=True, right=True)
  ax_histx = plt.axes(rect_histx)
  ax_histx.tick_params(direction='in', labelbottom=False)
  ax_histy = plt.axes(rect_histy)
  ax_histy.tick_params(direction='in', labelleft=False)

  # the scatter plot:
  ax_scatter.scatter(
      x, y,
      color="blue", label="train", s=.5,)

  # now determine nice limits by hand:
  x_low_lim = np.floor(x.min() / binwidth) * binwidth
  x_high_lim = np.ceil(x.max() / binwidth) * binwidth
  y_low_lim = np.floor(y.min() / binwidth) * binwidth
  y_high_lim = np.ceil(y.max() / binwidth) * binwidth

  print(x_low_lim, x_high_lim, y_low_lim, y_high_lim)

  ax_scatter.set_xlim((x_low_lim, x_high_lim))
  ax_scatter.set_ylim((y_low_lim, y_high_lim))

  x_bins = np.arange(x_low_lim, x_high_lim + binwidth, binwidth)
  y_bins = np.arange(y_low_lim, y_high_lim + binwidth, binwidth)
  ax_histx.hist(x, bins=x_bins)
  ax_histy.hist(y, bins=y_bins, orientation='horizontal')

  ax_histx.set_xlim(ax_scatter.get_xlim())
  ax_histy.set_ylim(ax_scatter.get_ylim())

  plt.show()


def scatter_vertical_histogram(
    x, y, binwidth, xlabel=None, ylabel=None
):
  x, y = np.array(x), np.array(y)

  # definitions for the axes
  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.005

  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]

  # start with a rectangular Figure
  plt.figure(figsize=(8, 8))

  ax_scatter = plt.axes(rect_scatter, xlabel=xlabel, ylabel=ylabel)
  ax_scatter.tick_params(direction='in', top=True, right=True)
  ax_histx = plt.axes(rect_histx)
  ax_histx.tick_params(direction='in', labelbottom=False)
  ax_histy = plt.axes(rect_histy)
  ax_histy.tick_params(direction='in', labelleft=False)

  # the scatter plot:
  ax_scatter.scatter(
      x, y,
      color="blue", label="train", s=.5,)

  # now determine nice limits by hand:
  x_low_lim = np.floor(x.min() / binwidth) * binwidth
  x_high_lim = np.ceil(x.max() / binwidth) * binwidth
  y_low_lim = np.floor(y.min() / binwidth) * binwidth
  y_high_lim = np.ceil(y.max() / binwidth) * binwidth

  print(x_low_lim, x_high_lim, y_low_lim, y_high_lim)

  ax_scatter.set_xlim((x_low_lim, x_high_lim))
  ax_scatter.set_ylim((y_low_lim, y_high_lim))

  x_bins = np.arange(x_low_lim, x_high_lim + binwidth, binwidth)
  y_bins = np.arange(y_low_lim, y_high_lim + binwidth, binwidth)
  ax_histx.hist(x, bins=x_bins)
#   ax_histy.hist(y, bins=y_bins, orientation='horizontal')

  ax_histx.set_xlim(ax_scatter.get_xlim())
  ax_histy.set_ylim(ax_scatter.get_ylim())

  plt.show()
