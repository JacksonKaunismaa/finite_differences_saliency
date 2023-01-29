import matplotlib.pyplot as plt
import numpy as np
import torch

def plt_grid_figure(grid, titles=None, colorbar=True, cmap=None, transpose=False, hspace=-0.4, first_cmap=None):
    #np_grid = np.array(grid).squeeze()
    #if len(np_grid.shape) != 4:
    #    np_grid = np.expand_dims(np_grid, 0)
    #if transpose:
    #    np_grid = np_grid.transpose(1,0,2,3)
    if not isinstance(grid[0], list):
        grid = grid[0]
    if cmap is None:
        cmap = "bwr"
    if transpose:
        ncols, nrows = len(grid), len(grid[0])
    else:
        nrows, ncols = len(grid), len(grid[0]) 
    im_size = grid[0][0].shape[0]
    print(nrows, ncols, im_size)
    fig = plt.figure(figsize=(4/128*im_size*ncols, 5/128*im_size*nrows))
    gridspec = fig.add_gridspec(nrows, ncols, hspace=hspace)
    axes = gridspec.subplots(sharex="col", sharey="row")
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)
    for i, row in enumerate(grid):
        for j, unsqueezed_img in enumerate(row):
            img = unsqueezed_img.squeeze() 
            if transpose:
                idx = (j,i)
            else:
                idx = (i,j)
            if idx[1] == 0: # assume explain_img is the first thing
                if len(img.shape) == 3 and img.shape[2] == 3:
                    im = axes[idx].imshow(img)
                else:
                    if first_cmap is None:
                        first_cmap = "gray"
                    im = axes[idx].imshow(img, cmap=first_cmap)
            else:
                img_max = np.max(abs(img))
                if cmap != "gray":
                    im = axes[idx].imshow(img, cmap=cmap, interpolation="nearest", vmax=img_max, vmin=-img_max)
                else:
                    axes[idx].imshow(img, cmap=cmap)
                if colorbar:
                    plt.colorbar(im, pad=0, fraction=0.048)
            if titles and idx[0] == 0:
                axes[idx].set_title(titles[idx[1]])
    plt.show()


def imshow_centered_colorbar(img, cmap, title, colorbar=True):
    heat_max = np.max(abs(img))
    plt.imshow(img, cmap=cmap, vmin=-heat_max, vmax=heat_max)
    if colorbar:
        plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)


def tensorize(inpt, device, requires_grad=False):
    if len(inpt.shape) == 2: # [size, size]
        inpt = np.expand_dims(np.expand_dims(inpt, 0), 0)
    elif len(inpt.shape) == 3:  # [size, size, C]
        inpt = np.expand_dims(inpt.transpose(2,0,1), 0)
    elif len(inpt.shape) == 4:  # [batch, size, size, 1]
        inpt = inpt.transpose(0,3,1,2)
   
    if inpt.dtype == np.uint8: 
        inpt = inpt.astype(np.float32)/255.
    tensored = torch.tensor(inpt, requires_grad=requires_grad).to(device).float()
    return tensored


def tprint(*args, t=False, **kwargs):
    if t:
        print(*args, **kwargs) 
