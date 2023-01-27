import matplotlib.pyplot as plt
import numpy as np

def plt_grid_figure(grid, titles=None, colorbar=True, cmap=None, transpose=False, hspace=-0.4):
    np_grid = np.array(grid).squeeze()
    if len(np_grid.shape) != 4:
        np_grid = np.expand_dims(np_grid, 0)
    if transpose:
        np_grid = np_grid.transpose(1,0,2,3)

    if cmap is None:
        cmap = "bwr"
    nrows, ncols = np_grid.shape[0], np_grid.shape[1]
    im_size = np_grid.shape[2]
    print(np_grid.shape, nrows, ncols)
    fig = plt.figure(figsize=(4/128*im_size*ncols, 5/128*im_size*nrows))
    gridspec = fig.add_gridspec(nrows, ncols, hspace=hspace)
    axes = gridspec.subplots(sharex="col", sharey="row")
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)
    for i, row in enumerate(np_grid):
        for j, img in enumerate(row):
            if j == 0: # assume explain_img is the first thing
                im = axes[i,j].imshow(img, cmap="gray")
            else:
                img_max = np.max(abs(img))
                if cmap != "gray":
                    im = axes[i,j].imshow(img, cmap=cmap, interpolation="nearest", vmax=img_max, vmin=-img_max)
                else:
                    axes[i,j].imshow(img, cmap=cmap)
                if colorbar:
                    plt.colorbar(im, pad=0, fraction=0.048)
            if titles and i == 0:
                axes[i,j].set_title(titles[j])
    plt.show()


def imshow_centered_colorbar(img, cmap, title, colorbar=True):
    heat_max = np.max(abs(img))
    plt.imshow(img, cmap=cmap, vmin=-heat_max, vmax=heat_max)
    if colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
