import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.interpolate import RegularGridInterpolator
import torch
from tqdm import tqdm
from network import correct  # for rate_distribution (reads it as a global variable)
from hooks import AllActivations
import utils
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from mayavi import mlab  # only works when running locally
# PCA Stuff


def find_pca_directions(dataset, sample_size, scales, strides, num_components=4):
# begin by computing pca directions and d_output_d_alphas
    sample = []
    for _ in range(sample_size):
        sample.append(dataset.generate_one()[0])
    sample = np.array(sample).squeeze().astype(np.float32)
    im_size = dataset.size

    if isinstance(strides, int):
        strides = [strides]*len(scales)

    pca_direction_grids = []
    for scale, stride in zip(scales, strides):
        windows = np.lib.stride_tricks.sliding_window_view(sample, (scale,scale), axis=(1,2))
        strided_windows = windows[:, ::stride, ::stride, :]  # [N, H, W, C]

        xs = np.mgrid[scale:im_size:stride]  # technically wrong (but its shape is correct)
        num_grid = xs.shape[0]
        pca_direction_grid = np.zeros((num_grid, num_grid, num_components, scale, scale, dataset.channels))

        pca_fitter = decomposition.PCA(n_components=num_components, copy=False)
        scale_fitter = StandardScaler()
        for i in tqdm(range(num_grid)):
            for j in range(num_grid):
                pca_selection = strided_windows[:, i, j, :]
                flattened = pca_selection.reshape(sample_size, -1)
                normalized = scale_fitter.fit_transform(flattened)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # gives pointless zero-division warnings
                    pca_fitter.fit(normalized)
                for comp in range(num_components):
                    pca_direction_grid[i, j, comp] = pca_fitter.components_[comp].reshape(scale, scale, dataset.channels)

        pca_direction_grids.append(pca_direction_grid.copy())
    return pca_direction_grids

def old_old_pca_direction_grids(model, dataset, target_class, img,
                        sample_size=512, scales=[3,5,9,15],
                        strides=None, pca_direction_grids=None,
                       gaussian=False, device=None):
    # begin by computing pca directions and d_output_d_alphas
    model.eval()
    im_size = dataset.size
    if strides is None:
        strides = scales
    if pca_direction_grids is None:
        pca_direction_grids = find_pca_directions(dataset, sample_size, scales, strides)
    d_out_d_alpha_grids = []
    interpolators = []
    indices_grid = np.mgrid[0:im_size, 0:im_size]

    for s, (scale, stride) in enumerate(zip(scales, strides)):
        index_window = np.lib.stride_tricks.sliding_window_view(indices_grid, (scale,scale), axis=(1,2))
        strided_indices = index_window[:, ::stride, ::stride]

        xs = np.mgrid[0:im_size-scale:stride]
        num_grid = xs.shape[0]
        #print(xs, num_grid)
        d_out_d_alpha_grid = np.zeros((num_grid, num_grid))
        for i in tqdm(range(num_grid)):
            for j in range(num_grid):
                # get pca direction for current patch
                pca_direction = pca_direction_grids[s][i,j,0].transpose(2,0,1)
                indices = strided_indices[:, i, j, ...]  # will have to slice these

                # do d_output_d_alpha computation
                alpha = torch.tensor(0.0, requires_grad=True).to(device)
                direction_tensor = torch.tensor(pca_direction).to(device).float().unsqueeze(0)
                img_tensor = torch.tensor(img.transpose(2,0,1)).to(device).float().unsqueeze(0)
                img_tensor[..., indices[0,0,0]:indices[0,-1,-1]+1, indices[1,0,0]:indices[1,-1,-1]+1] += alpha*direction_tensor
                output = model(img_tensor)[0,target_class]
                d_out_d_alpha = torch.autograd.grad(output, alpha)[0]

                d_out_d_alpha_grid[i,j] = d_out_d_alpha.detach().cpu().numpy()
        d_out_d_alpha_grids.append(d_out_d_alpha_grid.copy())
        interpolators.append(RegularGridInterpolator((xs+scale//2, xs+scale//2), d_out_d_alpha_grid,
                                                     bounds_error=False, fill_value=None))
    #print(d_out_d_alpha_grids[-1], d_out_d_alpha_grids[-1].shape)
    # now, per pixel, interpolate what the d_output_d_alpha value would be if the window
    # were centered at that pixel, then take the max over all possible scales
    saliency_map = np.zeros_like(img)
    scale_wins = [0] * len(scales)
    for i in tqdm(range(im_size)):
        for j in range(im_size):
            best_d_out_d_alpha = 0
            best_scale = -1
            for s in range(len(scales)):
                interp_value = interpolators[s]([i,j])
                if abs(interp_value) >= abs(best_d_out_d_alpha):
                    best_d_out_d_alpha = interp_value
                    best_scale = s
            saliency_map[i,j] = best_d_out_d_alpha
            scale_wins[best_scale] += 1
    print(scale_wins)
    return saliency_map

def pca_direction_grids(model, dataset, target_class, img, scales, pca_direction_grids,
                        strides=None, gaussian=False, device=None, component=0, batch_size=32):
    # begin by computing d_output_d_alphas
    model.eval()
    im_size = dataset.size
    if strides is None:
        strides = scales
    if isinstance(strides, int):
        strides = [strides]*len(pca_direction_grids)

    d_out_d_alpha_grids = []
    interpolators = []
    indices_grid = np.mgrid[0:im_size, 0:im_size]

    stacked_img = np.repeat(np.expand_dims(img, 0), batch_size, axis=0)
    stacked_img = np.transpose(stacked_img, (0, 3, 1, 2)).astype(np.float32) # NCHW format
    img_tensor = dataset.implicit_normalization(torch.tensor(stacked_img).to(device))

    for s, (scale, stride) in enumerate(zip(scales, strides)):
        # centers are [scale//2, ..., im_size-scale//2-1], num_windows = im_size-scale+1
        # the -1 on the upper limit center c.f. the "last index" being im_size-1
        # the num_windows is correct because `(im_size-scale//2-1) - (scale//2) = (im_size-2*(scale-1)/2-1) = im_size-scale`
        # and num elements of the array is last-first+1
        index_windows = np.lib.stride_tricks.sliding_window_view(indices_grid, (scale,scale), axis=(1,2))

        xs = np.mgrid[0:im_size-scale:stride, 0:im_size-scale:stride]  # indexes into pca_direction_grids
        num_grid = xs.shape[1]
        #print(xs, num_grid)
        d_out_d_alpha_grid = np.zeros((num_grid, num_grid))

        strided_indices = xs.transpose(1,2,0).reshape(-1, 2)  # ie should always pass strides=1 pca_directions into this
        unstrided_indices = np.mgrid[:num_grid, :num_grid].transpose(1,2,0).reshape(-1, 2)
        for k in tqdm(range(0, num_grid*num_grid, batch_size)):
            actual_batch_size = min(batch_size, num_grid*num_grid-k)
            batch_locs = strided_indices[k: k+actual_batch_size]
            batch_unstrided_locs = unstrided_indices[k: k+actual_batch_size]  # for indexing into a dense grid (num_grid, num_grid)

            pca_directions = pca_direction_grids[s][batch_locs[:,0], batch_locs[:,1], component]
            batch_window_indices = index_windows[:, batch_locs[:,0], batch_locs[:,1], ...]

            # do d_output_d_alpha computation
            alpha = torch.zeros((actual_batch_size,1,1,1), requires_grad=True).to(device)
            direction_tensor = dataset.implicit_normalization(torch.tensor(pca_directions).to(device).float())
            img_tensor[np.arange(actual_batch_size)[:,None,None], :, batch_window_indices[0], batch_window_indices[1]] += alpha*direction_tensor
            output = model(img_tensor)  # sum since gradient will be back-proped as vector of 1`s

            d_out_d_alpha = torch.autograd.grad(output[:,target_class].sum(), alpha)[0].squeeze()
            model.zero_grad()
            d_out_d_alpha_grid[batch_unstrided_locs[:,0], batch_unstrided_locs[:,1]] = d_out_d_alpha.detach().cpu().numpy()

        d_out_d_alpha_grids.append(d_out_d_alpha_grid.copy())
        # add scale//2 because centers of windows are actually offset by scale//2, and don't directly correspond to indices into
        # pca_direction_grid space
        interpolators.append(RegularGridInterpolator((xs[1,0]+scale//2, xs[1,0]+scale//2), d_out_d_alpha_grid,
                                                     bounds_error=False, fill_value=None))

    # now, per pixel, interpolate what the d_output_d_alpha value would be if the window
    # were centered at that pixel, then take the max over all possible scales
    #print(d_out_d_alpha_grids[-1])
    saliency_map = np.zeros_like(img).astype(np.float32)
    scale_wins = [0] * len(scales)
    for i in tqdm(range(im_size)):
        for j in range(im_size):
            best_d_out_d_alpha = 0
            best_scale = -1
            for s in range(len(scales)):
                interp_value = interpolators[s]([i,j])
                if abs(interp_value) >= abs(best_d_out_d_alpha):
                    best_d_out_d_alpha = interp_value
                    best_scale = s
            saliency_map[i,j] = best_d_out_d_alpha
            scale_wins[best_scale] += 1
    print(scale_wins)
    return saliency_map  # try jacobian with respect to window itself (isnt this just the gradient?)

def visualize_pca_directions(pca_direction_grid_scales, title, scales, lines=True):
    window_shape = pca_direction_grid_scales[0][0,0].shape
    num_components = window_shape[0]
    if len(window_shape) == 4:
        num_channels = window_shape[-1]
    else:
        num_channels = 1
    num_scales = len(pca_direction_grid_scales)
    print("Components:", num_components)
    print("Channels:", num_channels)
    print("Scales:", num_scales)
    plt.figure(figsize=(5*num_scales, 6*num_channels*num_components))
    plt.title(title)
    for component in range(num_components):
        for channel in range(num_channels):
            for i, res in enumerate(pca_direction_grid_scales):
                compressed_results = np.concatenate(np.concatenate(res[:, :, component, :, :, channel], 1), 1)
                subplot_idx = (component*num_scales*num_channels) + (num_scales*channel) + i+1
                plt.subplot(num_channels*num_components, num_scales, subplot_idx)
                line_width = scales[i] if lines else 0
                utils.imshow_centered_colorbar(compressed_results, "bwr",
                                f"Scale {scales[i]} Channel {channel} Component {component}", line_width=line_width)


def generate_many_pca(net, seeds, pca_directions_1_stride, scales, dataset, component=0, batch_size=128,
                      strides=None, skip_1_stride=False, device=None):
    _pca_map_s_strides = []
    _pca_map_1_strides = []
    _grad_maps = []
    _explain_imgs = []
    if strides is None:
        strides = scales
    for seed in seeds:
        np.random.seed(seed)
        generated_img, label, *__ = dataset.generate_one()
        tensored_img = utils.tensorize(generated_img, device=device, requires_grad=True)
        grad_map = torch.autograd.grad(net(tensored_img)[0,label.argmax()], tensored_img)[0]
        pca_map_strided = pca_direction_grids(net, dataset, label.argmax(), generated_img,
                                              scales, pca_directions_1_stride, strides=strides,
                                              device=device, batch_size=batch_size, component=component)
        if not skip_1_stride:
            pca_map_1_stride = pca_direction_grids(net, dataset, label.argmax(), generated_img,
                                                   scales, pca_directions_1_stride, component=component,
                                                   device=device, batch_size=batch_size, strides=1)
        _explain_imgs.append(generated_img)
        _grad_maps.append(grad_map.detach().cpu().squeeze(0).numpy().transpose(1,2,0))
        _pca_map_s_strides.append(pca_map_strided.copy())
        if not skip_1_stride:
            _pca_map_1_strides.append(pca_map_1_stride.copy())
    return _pca_map_s_strides, _pca_map_1_strides, _grad_maps, _explain_imgs



# Finite differences stuff



@torch.no_grad()
def finite_differences(model, dataset, target_class, stacked_img, locations, channel, unfairness, values_prior,
                       num_values, device, baseline_activations):
    largest_slope = np.zeros(stacked_img.shape[0])  # directional finite difference?
    slices = np.index_exp[np.arange(stacked_img.shape[0]), channel, locations[:, 0], locations[:, 1]]
    if values_prior is None:
        values_prior = np.linspace(5, 250, num_values) # uniform distribution assumption
    elif isinstance(values_prior, list):
        values_prior = np.expand_dims(np.asarray(values_prior), 1)
    num_loops = 1 if unfairness == "very unfair" else len(values_prior)
    for i in range(num_loops):
        shift_img = stacked_img.copy()
        # shifting method
        if unfairness in ["fair", "unfair"]:
            shift_img[slices] = values_prior[i]+0.01  # add tiny offset to "guarantee" non-zero shift
        elif unfairness in ["very unfair"]:
            critical_value_dists = shift_img[slices] - values_prior
            closest = np.argmin(abs(critical_value_dists), axis=0) # find closest class boundary
            shift_img[slices] = 0.01 + np.choose(closest, values_prior) - 10*np.sign(np.choose(closest, critical_value_dists))

        actual_diffs = shift_img[slices] - stacked_img[slices]
        img_norm = dataset.implicit_normalization(torch.tensor(shift_img).to(device)) # best is no normalization anyway
        if dataset.num_classes == 2:
            activations = dataset.class_multiplier(target_class)*model(img_norm, logits=True)
        else:
            activations = model(img_norm)[:, target_class]
        activation_diff = (activations - baseline_activations).cpu().numpy().squeeze()
        finite_difference = np.clip(activation_diff/actual_diffs, -30, 30) # take absolute slope
        largest_slope = np.where(abs(finite_difference) > abs(largest_slope), finite_difference, largest_slope)
    return largest_slope

def finite_differences_map(model, dataset, target_class, img, device=None, unfairness="fair", values_prior=None,
                           batch_size=32, num_values=20):
    # generate a saliency map using finite differences method (iterate over colors)
    model.eval()
    im_size = dataset.size
    #img = img.astype(np.float32)/255. # normalization handled later
    indices = np.mgrid[:im_size, :im_size].transpose(1,2,0).reshape(im_size*im_size, -1)
    stacked_img = np.repeat(np.expand_dims(img, 0), batch_size, axis=0)
    stacked_img = np.transpose(stacked_img, (0, 3, 1, 2)).astype(np.float32) # NCHW format
    img_heat_map = np.zeros_like(img).astype(np.float32)

    with torch.no_grad():
        cuda_stacked_img = dataset.implicit_normalization(torch.tensor(stacked_img).to(device))
        if dataset.num_classes == 2:
            baseline_activations = dataset.class_multiplier(target_class)*model(cuda_stacked_img, logits=True)
        else:
            baseline_activations = model(cuda_stacked_img)[:, target_class]
        del cuda_stacked_img

    for channel in range(dataset.channels):
        for k in tqdm(range(0, im_size*im_size, batch_size)):
            actual_batch_size = min(batch_size, im_size*im_size-k)
            locations = indices[k:k+actual_batch_size]
            largest_slopes = finite_differences(model, dataset, target_class, stacked_img[:actual_batch_size],
                                                locations, channel, unfairness, values_prior, num_values, device,
                                                baseline_activations[:actual_batch_size])
            img_heat_map[locations[:,0], locations[:,1], channel] = largest_slopes
    return img_heat_map#.sum(axis=2)  # linear approximation aggregation?


# General visualizations


@torch.no_grad()
def rate_distribution(net, loader, dataset, device=None, buckets=100, critical_values=[]):
    # Plot model error rates as function of color (for greyscale dataset)
    net.eval()
    total = np.zeros((buckets))
    num_correct = np.zeros((buckets))
    num_possible_colors = dataset.color_range[1] - dataset.color_range[0]
    for sample in tqdm(loader):
        imgs = sample["image"].to(device).float()
        labels = sample["label"].to(device).float()
        actual_colors = sample["color"]
        color_indices = (buckets * (actual_colors - dataset.color_range[0]) / num_possible_colors).int().numpy()
        outputs = net(imgs)
        correct_preds = correct(outputs, labels).cpu().numpy()
        for i, color_idx in enumerate(color_indices):
            total[color_idx] += 1
            num_correct[color_idx] += correct_preds[i]

    # Plot results of rate_distribution
    num_wrong = total - num_correct
    width = 0.4
    labels = [int(x) for i, x in enumerate(np.linspace(*dataset.color_range, buckets))]
    plt.bar(labels, num_correct, width, label="correct amount")
    plt.bar(labels, num_wrong, width, bottom=num_correct, label="wrong amount")
    plt.vlines(critical_values, 0, np.max(total), linewidth=0.8,
               colors="r", label="decision boundary",
               linestyles="dashed")
    plt.legend()
    plt.xlabel("Color value")
    plt.show()


@torch.no_grad()
def response_graph(net, dataset, test_index=987_652, use_arcsinh=True, device=None):
    # Plot network output logits as color varies for a specific image
    net.eval() # very important!
    counterfactual_color_values = np.linspace(0, 255, 255)
    responses = []
    for color in counterfactual_color_values:
        np.random.seed(test_index)
        generated_img, lbl, *__ = dataset.generate_one(set_color=color)
        generated_img = utils.tensorize(generated_img, device)
        response = net(generated_img, logits=True).cpu().numpy()
        responses.append(np.squeeze(response))

    responses = np.asarray(responses)
    if use_arcsinh:
        responses = np.arcsinh(responses)

    for output_logit in range(responses.shape[1]):
        plt.plot(counterfactual_color_values, responses[:, output_logit], label=f"class {output_logit}")

    plt.legend()
    plt.xlabel("Color value")
    plt.ylabel("Network output logit")
    plt.vlines([100, 150], np.min(responses), np.max(responses), linewidth=0.8,
               colors="r", label="decision boundary",
               linestyles="dashed") # with .eval() works well



# Mechanistic Interpretation



def compute_profile_plot(profile, dataset):
    avg_line = []
    start = 0
    for c in range(255):
        if np.any(profile[:,0]==c):
            avg_line.append(profile[:,1][profile[:,0]==c].mean())
        else:
            if avg_line:
                avg_line.append(avg_line[-1]) # assume previous value continues
            else:
                start += 1  # or just start elsewhere
    avg_line = np.asarray(avg_line)

    color_values = np.arange(start, 255)
    classes = np.vectorize(dataset.color_classifier)(color_values)/(dataset.num_classes-1)
    return avg_line, color_values, classes, profile


def get_profile_y_lims(profile_plot):
    avg_line, color_values, classes, profile = profile_plot
    max_y = min(max(avg_line)*1.3, max(profile[:,1]))
    min_y = max(min(avg_line)*(1-0.3*np.sign(min(avg_line))), min(profile[:,1]))
    return min_y, max_y


def show_profile_plot(profile_plot, hide_ticks=False, ax=None, rm_border=False, y_lims=None):
    if ax is None:
        ax = plt.gca()

    avg_line, color_values, classes, profile = profile_plot

    if y_lims is None:
        min_y, max_y = get_profile_y_lims(profile_plot)
    else:
        min_y, max_y = y_lims
    scaled_classes = classes*(max_y-min_y) + min_y

    ax.scatter(profile[:,0], profile[:,1], s=0.5, marker="o", alpha=0.08)
    ax.plot(color_values, avg_line, c="r")
    ax.plot(color_values, scaled_classes, c="k", alpha=0.25)

    ax.set_ylim(min_y-(max_y-min_y)*0.01, max_y+(max_y-min_y)*0.01)
    if hide_ticks:
        ax.set_xticks([])
    if rm_border:
        utils.remove_borders(ax, ["top", "bottom", "right"])#, "left"])


def show_profile_plots(color_profile, names, hide_ticks=True, rm_border=True, fixed_height=False, size_mul=1.):
    shape = utils.find_good_ratio(len(names))
    if not isinstance(size_mul, tuple):
        size_mul = (size_mul, size_mul)

    figsize = shape[1]*6*size_mul[1], shape[0]*6*size_mul[0]

    fig = plt.figure(figsize=figsize)#, constrained_layout=True)

    gs = fig.add_gridspec(shape[0], shape[1])
    axes = gs.subplots()
    if not isinstance(axes, np.ndarray):
        axes = np.array(axes)[None, None]
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 0)

    min_min_y = np.inf
    max_max_y = -np.inf
    if fixed_height:
        for name in names:
            min_y, max_y = get_profile_y_lims(color_profile[name])
            min_min_y = min(min_min_y, min_y)
            max_max_y = max(max_max_y, max_y)
        y_lims = (min_min_y, max_max_y)
    else:
        y_lims = None

    for h in range(shape[0]):
        for w in range(shape[1]):
            name = w+h*shape[1]
            show_profile_plot(color_profile[names[name]],
                              hide_ticks=hide_ticks, ax=axes[h,w], rm_border=rm_border, y_lims=y_lims)


@torch.no_grad()
def activation_color_profile(net, loader, dataset, device=None):
    # assume it already has AllActivations hooks on it
    net.eval()
    filter_activations = defaultdict(list) # conv filter: [(color, sum_activations), ...]
    for sample in tqdm(loader):
        imgs = sample["image"].to(device).float()
        #labels = sample["label"].to(device).float()
        colors = sample["color"]
        net(imgs)

        for layer_name, activation in net._features.items():
            for i, color in enumerate(colors):
                # ignore batchnorms, and pre-activation func
                if "act_func" in layer_name: # happens to be identical implementation for fully_connected
                    for channel in range(activation.shape[1]):
                        entry = color.item(), activation[i][channel].sum().item() # could do group conv to give data on input channels?
                        filter_activations[f"{layer_name}_{channel}"].append(entry)
                else:
                    break
    filter_activations = {k:np.asarray(v) for k,v in filter_activations.items()}
    profile_plots = {k:compute_profile_plot(filter_activations[k], dataset) for k in filter_activations.keys()}
    return profile_plots, filter_activations


def show_conv_layer(net, layer_name, shape=None):
    # assume net it already has AllActivations hooks on it
    features = net._features[layer_name].detach().cpu().numpy().squeeze() # CHW
    if shape is None:
        shape = utils.find_good_ratio(features.shape[0])
    height,width = shape
    map_size = features.shape[-1]
    grid_features = features.reshape(height, width, map_size, map_size)
    stacked_features = np.concatenate(np.concatenate(grid_features, 1), 1)
    plt.figure(figsize=(map_size*width/32*6, map_size*height/32*6))
    utils.imshow_centered_colorbar(stacked_features, title=layer_name,
                                   line_width=map_size, colorbar=False, rm_border=True)


def get_weight(net, weight_name):
    if "conv_block" in weight_name:
        infer_weight_name = weight_name.replace("act_func", "conv") # infer desired weights
    else:
        infer_weight_name = weight_name.replace("act_func", "fully_connected") # infer desired weights
    attrs = infer_weight_name.split(".")

    if hasattr(net, "model"):
        curr_module = net.model
    else:
        curr_module = net

    for attr in attrs:
        try:
            curr_module = curr_module[int(attr)]
        except ValueError:
            curr_module = getattr(curr_module, attr)
    return curr_module.weight.detach().cpu().numpy()

def show_fully_connected(net, weight_name):
    pass

def show_conv_weights(net, weight_name, color_profile=None, outer_shape=None, size_mul=6, rm_border=True, fixed_height=False):
    # assume net it already has AllActivations hooks on it
    # implicitly assume that channel-aligned view is salient for the given network
    weights = get_weight(net, weight_name)   # N_out, N_in, K, K
    if outer_shape is None:
        outer_shape = utils.find_good_ratio(weights.shape[0])
    inner_shape = utils.find_good_ratio(weights.shape[1])
    map_size = weights.shape[-1]
    def reshaper(w):
        grid_w = w.reshape(*inner_shape, map_size, map_size)
        return np.concatenate(np.concatenate(grid_w, 1), 1)
    show_weights(weights, weight_name, reshaper, inner_shape, outer_shape, map_size, color_profile, size_mul, rm_border, fixed_height)


def show_fc_conv(net, weight_name="fully_connected.0.act_func", size_mul=1, color_profile=None, rm_border=True,
                 selection=None, fixed_height=False, full_gridspec=False):
    # ie. the first fully connected layer
    fc_weight = get_weight(net, weight_name) # out, in
    if selection:
        fc_weight = fc_weight[selection] # if only want to look at specific logits
    final_img_shape = net.model.final_img_shape  # HWC
    fc_out_shape = utils.find_good_ratio(fc_weight.shape[0])
    conv_shape = utils.find_good_ratio(final_img_shape[2])
    map_size = final_img_shape[0]
    def reshaper(w):
        grid_w = w.reshape(*conv_shape, map_size, map_size)
        if full_gridspec:
            return grid_w
        return np.concatenate(np.concatenate(grid_w, 1), 1)
    show_weights(fc_weight, weight_name, reshaper, conv_shape, fc_out_shape, map_size, color_profile, size_mul, rm_border, fixed_height, selection=selection, full_gridspec=full_gridspec)


def show_fc(net, weight_name, size_mul=1, color_profile=None, rm_border=True, selection=None, fixed_height=False):
    fc_weight = get_weight(net, weight_name) # out, in
    if selection:
        fc_weight = fc_weight[selection] # if only want to look at specific logits
    logits_shape = utils.find_good_ratio(fc_weight.shape[0])
    def reshaper(w):
        return w[None,:]
    show_weights(fc_weight, weight_name, reshaper, (1,1), logits_shape, 1, color_profile, size_mul, rm_border, fixed_height, selection=selection)


def show_weights(weights, weight_name, reshaper, inner_shape, outer_shape, map_size, color_profile, size_mul,
                 rm_border, fixed_height, selection=None, full_gridspec=False):
    if selection is None:
        selection = np.arange(outer_shape[0]*outer_shape[1])
    if not isinstance(size_mul, tuple):
        size_mul = (size_mul, size_mul)


    figsize = map_size*outer_shape[1]*inner_shape[1]/8*size_mul[1], map_size*outer_shape[0]*inner_shape[0]/8*size_mul[0]
    if color_profile:
        figsize = figsize[0], figsize[1]*2
    fig = plt.figure(figsize=figsize)#, constrained_layout=True)
    #plt.suptitle(weight_name)
    if color_profile:
        gs = fig.add_gridspec(outer_shape[0]*2, outer_shape[1], hspace=0)
    else:
        gs = fig.add_gridspec(outer_shape[0], outer_shape[1])
    axes = gs.subplots()
    if not isinstance(axes, np.ndarray):
        axes = np.array(axes)[None, None]
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, 1 if outer_shape[0] == 1 and color_profile else 0)
    #print(axes.shape)

    min_min_y = np.inf
    max_max_y = -np.inf
    if color_profile and fixed_height:
        for item in selection:
            min_y, max_y = get_profile_y_lims(color_profile[f"{weight_name}_{item}"])
            min_min_y = min(min_min_y, min_y)
            max_max_y = max(max_max_y, max_y)
        y_lims = (min_min_y, max_max_y)
    else:
        y_lims = None

    for h in range(outer_shape[0]):
        for w in range(outer_shape[1]):
            channel = w+h*outer_shape[1]
            gs_idx = (h*2,w) if color_profile else (h,w)
            if full_gridspec:
                sub_gs = gs[gs_idx].subgridspec(*inner_shape)
                sub_axes = sub_gs.subplots()
                for i in range(inner_shape[0]):
                    for j in range(inner_shape[1]):
                        utils.imshow_centered_colorbar(reshaper(weights[channel])[i,j], colorbar=False,
                                                       ax=sub_axes[i,j], rm_border=rm_border)
                axes[gs_idx].set_xticks([])
                axes[gs_idx].set_yticks([])
                if rm_border:
                    utils.remove_borders(axes[gs_idx])
            else:
                utils.imshow_centered_colorbar(reshaper(weights[channel]), line_width=map_size if map_size > 1 else 0,
                                        colorbar=False, ax=axes[gs_idx], rm_border=rm_border)
            if color_profile:
                gs_idx = h*2+1, w
                show_profile_plot(color_profile[f"{weight_name}_{selection[channel]}"],
                                  hide_ticks=True, ax=axes[gs_idx], rm_border=True, y_lims=y_lims)


# only works when running locally
def mlab_fc_conv_feature_angles(net, layer_name, num_embed=3, normalize=True):
    # get conv->fc weight/feature angles
    weights = get_weight(net, layer_name)  # (In, Out)
    shape = net.final_img_shape
    shaped_w = weights.reshape(-1, shape[2], shape[0]*shape[1])  # C_out, C_in, HW
    mags = np.linalg.norm(shaped_w, axis=-1)[..., np.newaxis]  # 3, 6, 64
    norm_w = shaped_w/mags if normalize else shaped_w
    dots = np.einsum("kji,lji->jkl", norm_w, norm_w)

    if num_embed not in [2,3]:
        return dots

    def gram_diff(w, desired_grams):
        w_mat = w.reshape(*embed_shape)
        grams = w_mat.dot(w_mat.T)
        return ((grams - desired_grams)**2).sum()
    embed_shape = (norm_w.shape[0], num_embed)
    embedded_weights = np.zeros((norm_w.shape[1], *embed_shape))
    closeness = np.zeros(norm_w.shape[1])
    for i, feats in enumerate(norm_w.transpose(1,0,2)): # 3, 64
        projection = minimize(gram_diff, np.ones(num_embed*embed_shape[0]),
                              args=(dots[i],))
        embedded_weights[i] = projection.x.reshape(*embed_shape)
        closeness[i] = projection.fun

    #grid_shape = utils.find_good_ratio(norm_w.shape[1])
#    colors = ["coral", "forestgreen", "royalblue"]
#    labels = ["class "+x for x in "012"]
    origin = np.zeros(embed_shape[0])
    for w in embedded_weights:
        #lims = np.max(abs(w))
        if num_embed == 3:
            print(w.shape, origin.shape)
            obj = mlab.quiver3d(origin, origin, origin, w[:,0], w[:,1], w[:,2])#, color=colors[k])#, label=labels[k])
            obj.actor.property.interpolation = "phong"
            obj.actor.property.specular = 0.1
            obj.actor.property.specular_power = 5
            return obj
        #else:
        #    ax = fig.add_subplot(gs[i,j])
        #    for k in range(embed_shape[0]):
        #        ax.quiver(0, 0, w[k,0], w[k,1], scale=2.0, color=colors[k], label=labels[k])
        #ax.set_xticklabels([])
        #ax.set_yticklabels([])
        #ax.set_title(f"{int(closeness[feature]*1000.)}")
        #ax.set_xlim(-lims, lims)
        #ax.set_ylim(-lims, lims)
    #fig.legend(*ax.get_legend_handles_labels())
    return dots, embedded_weights


def fc_conv_feature_angles(net, layer_name, num_embed=3, normalize=True):
    # get conv->fc weight/feature angles
    weights = get_weight(net, layer_name)  # (In, Out)
    shape = net.final_img_shape
    shaped_w = weights.reshape(-1, shape[2], shape[0]*shape[1])  # C_out, C_in, HW
    mags = np.linalg.norm(shaped_w, axis=-1)[..., np.newaxis]  # 3, 6, 64
    norm_w = shaped_w/mags if normalize else shaped_w
    dots = np.einsum("kji,lji->jkl", norm_w, norm_w)

    if num_embed not in [2,3]:
        return dots

    def gram_diff(w, desired_grams):
        w_mat = w.reshape(*embed_shape)
        grams = w_mat.dot(w_mat.T)
        return ((grams - desired_grams)**2).sum()
    embed_shape = (norm_w.shape[0], num_embed)
    embedded_weights = np.zeros((norm_w.shape[1], *embed_shape))
    closeness = np.zeros(norm_w.shape[1])
    for i, feats in enumerate(norm_w.transpose(1,0,2)): # 3, 64
        projection = minimize(gram_diff, np.ones(num_embed*embed_shape[0]),
                              args=(dots[i],))
        embedded_weights[i] = projection.x.reshape(*embed_shape)
        closeness[i] = projection.fun

    fig = plt.figure()
    grid_shape = utils.find_good_ratio(norm_w.shape[1])
    gs = fig.add_gridspec(*grid_shape)
    colors = ["coral", "forestgreen", "royalblue"]
    labels = ["class "+x for x in "012"]
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            feature = i*grid_shape[1]+j
            w = embedded_weights[feature]
            lims = np.max(abs(w))
            if num_embed == 3:
                ax = fig.add_subplot(gs[i,j], projection=Axes3D.name)
                ls = LightSource()
                for k in range(embed_shape[0]):
                    obj = ax.quiver(0, 0, 0, w[k,0], w[k,1], w[k,2], color=colors[k], label=labels[k])
                    for seg in obj._segments3d:
                        ls.shade(seg, plt.cm.RdYlBu)
                ax.set_zlim(-lims, lims)
                ax.set_zticklabels([])
            else:
                ax = fig.add_subplot(gs[i,j])
                for k in range(embed_shape[0]):
                    ax.quiver(0, 0, w[k,0], w[k,1], scale=2.0, color=colors[k], label=labels[k])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f"{int(closeness[feature]*1000.)}")
            ax.set_xlim(-lims, lims)
            ax.set_ylim(-lims, lims)
    fig.legend(*ax.get_legend_handles_labels())
    return dots, embedded_weights

def display_fc_conv_grams(dots, selection=None, fig_mul=1.0):
    if selection is not None:
        dots = dots[:, selection][:, :, selection]
    shape = utils.find_good_ratio(dots.shape[0])
    fig = plt.figure(figsize=(6*fig_mul, 6*fig_mul))
    gs = fig.add_gridspec(*shape)
    axes = gs.subplots()
    for i in range(shape[0]):
        for j in range(shape[1]):
            feature = i*shape[1]+j
            utils.imshow_centered_colorbar(dots[feature], cmap="bwr", colorbar=False, rm_border=False, ax=axes[i,j])

# Network training curve plots



def defaultdict_helper():  # to make things pickle-able
    return defaultdict(int)


def final_activation_tracker(net, loss, optimizer, loader, device=None):
    # final hidden layer activations
    net.eval()
    final_fc_name = f"fully_connected.{len(net.fully_connected)-2}."
    post_relu_name = final_fc_name+"act_func"
    pre_relu_name = final_fc_name+"fully_connected"
    debug_net = AllActivations(net)

    pattern_counts = defaultdict(defaultdict_helper)

    post_relu_activations = []
    pre_relu_activations = []
    final_outputs = []
    post_relu_grads = []
    pre_relu_grads = []
    for i, sample in tqdm(enumerate(loader)):
        imgs = sample["image"].to(device).float()
        labels = sample["label"].to(device)
        colors = sample["color"]
        output = debug_net(imgs)
        batch_loss = loss(output, labels)

        post_relu_acts = debug_net._features[post_relu_name]
        pre_relu_acts = debug_net._features[pre_relu_name]
        post_relu_grad = torch.autograd.grad(batch_loss, post_relu_acts)[0]
        nonzero_mask = torch.where(post_relu_acts > 0, 1, 0)
        pre_relu_grad = post_relu_grad*nonzero_mask
        #input("post access")
        post_relu_activations.append(post_relu_acts.detach().cpu().numpy())
        pre_relu_activations.append(pre_relu_acts.detach().cpu().numpy())
        final_outputs.append(output.detach().cpu().numpy())
        post_relu_grads.append(post_relu_grad.detach().cpu().numpy())
        pre_relu_grads.append(pre_relu_grad.detach().cpu().numpy())
        #input("post append")
        #track patterns
        nonzero = nonzero_mask.detach().cpu().numpy()
        for row, color in zip(nonzero, colors):
            pattern = str(row)
            pattern_counts[int(color)][pattern] += 1


    output_sample = np.concatenate(final_outputs, axis=0)
    pre_relu_sample = np.concatenate(pre_relu_activations, axis=0)
    post_relu_sample = np.concatenate(post_relu_activations, axis=0)
    pre_corr = np.corrcoef(output_sample, pre_relu_sample, rowvar=False)[:3, 3:]
    post_corr = np.corrcoef(output_sample, post_relu_sample, rowvar=False)[:3, 3:]
    post_grad_avg = (np.concatenate(post_relu_grads, axis=0)**2).mean(axis=0)
    pre_grad_avg = (np.concatenate(pre_relu_grads, axis=0)**2).mean(axis=0)
    #input("end of stats")
    del imgs, labels, colors
    del debug_net._features
    return pattern_counts, pre_corr, post_corr, pre_grad_avg, post_grad_avg


def plot_results(result, network_name, fig_mul=1.5, size=0.2, alpha=1.0):
    if not isinstance(fig_mul, tuple):
        fig_mul = (fig_mul, fig_mul)
    plt.figure(figsize=(6*fig_mul[0], 6*fig_mul[1]))
    plt.suptitle(network_name)

    plt.subplot(3,2,1)
    plt.title("Log(Loss) Curves")
    plt.plot(np.log(result[0]), label="Valid")
    plt.plot(np.log(result[2]), label="Train")
    plt.xticks([])
    plt.legend()

    plt.subplot(3,2,2)
    plt.xticks([])
    plt.title("Log2(Num unique configurations of output logits used)")
    plt.plot(np.log2(result[-1]))

    colors = ["coral", "forestgreen", "royalblue"]
    np_extra_stats = [np.stack([elem[t] for elem in result[-2]], axis=0) for t in range(1,5)]  # 4,100
    pre_corr, post_corr, pre_grad_avg, post_grad_avg = np_extra_stats
    num_hidden = pre_corr.shape[-1]

    plt.subplot(3,2,3)
    plt.title("Pre-ReLU abs(corr_coef) of logits")
    for i, color in enumerate(colors):
        for logit in range(num_hidden):
            plt.plot(abs(pre_corr[:,i,logit]), c=color, linewidth=size, alpha=alpha)   # 100, 3, 32
    plt.xticks([])

    plt.subplot(3,2,4)
    plt.title("Post-ReLU abs(corr_coef) of logits")
    for i, color in enumerate(colors):
        for logit in range(num_hidden):
            plt.plot(abs(post_corr[:,i,logit]), c=color, linewidth=size, alpha=alpha)
    plt.xticks([])

    plt.subplot(3,2,5)
    plt.title("Mean grad**2 of pre-relu logits")
    for logit in range(num_hidden):
        plt.plot(pre_grad_avg[:,logit], c=colors[-1], linewidth=size, alpha=alpha)  # 100, 32
    plt.xlabel("Epoch")

    plt.subplot(3,2,6)
    plt.title("Mean grad**2 of post-relu logits")
    for logit in range(num_hidden):
        plt.plot(post_grad_avg[:,logit], c=colors[-1], linewidth=size, alpha=alpha)
    plt.xlabel("Epoch")


def count_logit_usage(pattern_counts): # num unique configurations
    if isinstance(pattern_counts, tuple):
        pattern_counts = pattern_counts[0]
    uniq = set()
    for x in pattern_counts.values():
        uniq = uniq.union(set(x.keys()))
    return len(uniq)
