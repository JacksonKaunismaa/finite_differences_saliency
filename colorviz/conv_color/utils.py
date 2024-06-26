import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo.eval_frame import OptimizedModule


def get_rank():  # consider making this a decorator (with a parameter that specifies a default return value)
    try:
        return dist.get_rank()
    except RuntimeError:
        return 0

def get_world_size():
    try:
        return dist.get_world_size()
    except RuntimeError:
        return 1
    
def get_raw(net): # slightly awkward way to consistently access underyling Transformer object
    if isinstance(net, OptimizedModule):
        net = net._modules["_orig_mod"]
    if isinstance(net, DDP):
        net = net.module
    return net
    

def remove_borders(ax, borders=None):
    if borders is None:
        borders = ["top", "bottom", "right", "left"]
    for border in borders:
        ax.spines[border].set_visible(False)

def plot_color_classes(dataset, y_scale, labels=False, alpha=0.25):
    color_probe = np.linspace(0, 255, 255)
    color_class = np.asarray([dataset.color_classifier(x) for x in color_probe])
    color_class = color_class/(dataset.num_classes-1)*(y_scale[1]-y_scale[0]) + y_scale[0]

    plt.plot(color_probe, color_class, c="k", alpha=alpha)
    if labels:
        plt.xlabel("Color")
        plt.yticks([0, 1, 2])
        plt.ylabel("Class")


def find_good_ratio(size): # find ratio as close to a square as possible
    start_width = int(np.sqrt(size))
    for width in range(start_width+1, 0, -1):
        if size % width == 0:
            break
    height = size//width
    return min(height, width), max(height, width)  # height, width

def image_grid(img_list, titles=None, force_linear=False):  # plot a grid of images, with img_list being flat
    if force_linear:
        height = len(img_list)
        width = 1
    else:
        height, width = find_good_ratio(len(img_list))
    im_size = img_list[0].shape[0]
    plt.figure(figsize=(4/128*im_size*width, 5/128*im_size*height))
    for i in range(height):
        for j in range(width):
            idx = i*width + j + 1
            plt.subplot(height, width, idx)
            plt.imshow(img_list[idx-1])
            if titles:
                plt.title(titles[idx-1])
            remove_borders(plt.gca())
            plt.xticks([])
            plt.yticks([])


def plt_grid_figure(inpt_grid, col_titles=None, colorbar=True, cmap=None, transpose=False, hspace=-0.4, 
                    first_cmap=None, channel_mode=None, row_titles=None, crop_to=None, zero_centered_cmap=True):
    # plot a grid of images (inpt grid must be in the shape you want the images to display in)
    # inpt_grid.shape == (N_vert, N_horiz, img_height, img_width)
    #np_grid = np.array(grid).squeeze()
    #if len(np_grid.shape) != 4:
    #    np_grid = np.expand_dims(np_grid, 0)
    #if transpose:
    #    np_grid = np_grid.transpose(1,0,2,3)
    if not isinstance(inpt_grid[0], (list, np.ndarray)):
        inpt_grid = [inpt_grid]
    if cmap is None:
        cmap = "bwr"
        

    nrows = len(inpt_grid[0]) if transpose else len(inpt_grid)
    if channel_mode not in ["split", "collapse"] and not isinstance(channel_mode, int) and channel_mode is not None:
        raise ValueError(f"Invalid value for channel_mode: '{channel_mode}'. Must be in ['split', 'collapse', int].")

    if channel_mode == "split":
        grid = [[] for _ in range(nrows)]
        expanded_col_titles = []
        for i, row in enumerate(inpt_grid):
            for j, img in enumerate(row):
                idx = (j,i) if transpose else (i,j)
                # print("on idx", idx, "img_shape", img.shape)
                if idx[1] != 0 and img.ndim == 3:  # not in first column
                    # print("decided to do some expanding, curr_len", len(grid[idx[0]]))
                    expanded_view = [channel_view for channel_view in img.transpose(2,0,1)]
                    # print("expandend_amount", len(expanded_view), len(grid[idx[0]]))
                    grid[idx[0]] += expanded_view  # ie. assume HWC
                    # print("len(grid[idx[0]])", len(grid[idx[0]]))
                    if idx[0] == 0:  # in first row
                        #print("expanding titles also", len(titles), j, len(row))
                        expanded_col_titles += [f"Channel {c} {col_titles[idx[1]] if col_titles else ''}" for c in range(len(expanded_view))]
                else:
                    grid[idx[0]].append(img)
                    if idx[0] == 0 and col_titles:
                        expanded_col_titles.append(col_titles[idx[1]])
        col_titles = expanded_col_titles
    else:
        grid = inpt_grid

    #print([len(x) for x in grid])

    im_size = crop_to if crop_to else grid[0][0].shape[0]

    ncols = len(grid) if transpose and not channel_mode == "split" else len(grid[0])

    #print(expanded_titles, len(expanded_titles))
    print(f"{nrows=}, {ncols=}, {im_size=}")
    fig = plt.figure(figsize=(4/128*im_size*ncols, 5/128*im_size*nrows))
    gridspec = fig.add_gridspec(nrows, ncols, hspace=hspace)
    axes = gridspec.subplots(sharex="col", sharey="row")
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)
    print(f"{axes.shape=}")
    for i, row in enumerate(grid):
        for j, unsqueezed_img in enumerate(row):
            img = unsqueezed_img.squeeze()
            if crop_to is not None:
                # print(img.shape)
                im_actual_size = img.shape[0]
                crop_amt = (im_actual_size - crop_to)//2
                img = img[crop_amt:-crop_amt, crop_amt:-crop_amt]
                # print(img.shape, crop_amt, im_actual_size, i, j)
            idx = (j,i) if transpose and not channel_mode == "split" else (i,j)  # split_channels already accounts for transpose
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            remove_borders(axes[idx])
            if idx[1] == 0: # assume explain_img is the first thing
                if len(img.shape) == 3 and img.shape[2] == 3:
                    # print(img.shape)
                    im = axes[idx].imshow(img)
                else:
                    if first_cmap is None:
                        first_cmap = "gray"
                    im = axes[idx].imshow(img, cmap=first_cmap)
            else:
                if channel_mode == "collapse":
                    img = abs(img).sum(axis=-1)
                elif isinstance(channel_mode, int):
                    img = img[..., channel_mode]

                img_max = np.max(abs(img))
                if cmap != "gray":
                    # print(img.sum())
                    if zero_centered_cmap:
                        im = axes[idx].imshow(img, cmap=cmap, vmax=img_max, vmin=-img_max) # interpolation='nearest'
                    else:
                        im = axes[idx].imshow(img, cmap=cmap)
                else:
                    axes[idx].imshow(img, cmap=cmap)
                if colorbar:
                    plt.colorbar(im, pad=0, fraction=0.048)
            if col_titles and idx[0] == 0:
                axes[idx].set_title(col_titles[idx[1]])
            if row_titles and idx[1] == 0:
                axes[idx].set_ylabel(row_titles[idx[0]])
    #plt.show()


def imshow_centered_colorbar(img, cmap="bwr", title=None, colorbar=True, num_lines=0, line_width=0, ax=None, rm_border=False):
    if ax is None:
        ax = plt.gca()
    heat_max = np.max(abs(img))
    img_width, img_height = img.shape[1], img.shape[0]
    if cmap == "gray":
        im = ax.imshow(img, cmap=cmap, extent=(0,img_width,0,img_height))
    else:
        im = ax.imshow(img, cmap=cmap, vmin=-heat_max, vmax=heat_max, extent=(0,img_width,0,img_height))

    ax.set_xticks([])
    ax.set_yticks([])
    if rm_border:
        remove_borders(ax)

    if colorbar:
        plt.colorbar(im, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    if num_lines or line_width:  # num lines assumes equal in both directions
        if not line_width:
            line_width = img_width // num_lines

        ax.vlines(list(range(line_width, img_width, line_width)), linestyle="dotted",
            ymin=0, ymax=img_height, colors="k")
        ax.hlines(list(range(line_width, img_height, line_width)), linestyle="dotted",
            xmin=0, xmax=img_width, colors="k")
        ax.set_xlim(0,img_width)
        ax.set_ylim(0,img_height)


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
