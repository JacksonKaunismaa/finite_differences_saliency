import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.interpolate import RegularGridInterpolator
import torch
import torch.utils.data
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from tensorflow.python.data.ops.dataset_ops import BatchDataset
import cv2


from .hooks import AllActivations
from . import utils
from . import training
#from mayavi import mlab  # only works when running locally
# PCA Stuff

def combine_saliency_and_img(img, saliency, channel=0, do_abs=True, alpha=0.4, gamma=0, method="jet"):
    # combine an image and its saliency map so that they can be simultaneously visualzed
    # it also only takes a certain channel from the saliency map, since they are usually the same anyway
    # does a linear blend, can probably do something better
    saliency = saliency[...,channel]
    if do_abs:
        saliency = abs(saliency)
    centered_saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    if method == "jet":
        saliency_bytes = 255 - cv2.convertScaleAbs(centered_saliency*255)
        saliency_cmapped = cv2.applyColorMap(saliency_bytes, cv2.COLORMAP_JET).astype(img.dtype)/255.
    elif method == "gray":
        saliency_bytes = cv2.convertScaleAbs((centered_saliency**3)*255)
        saliency_cmapped = cv2.cvtColor(saliency_bytes, cv2.COLOR_GRAY2RGB).astype(img.dtype)/255.
    elif method == "bone":
        saliency_bytes = cv2.convertScaleAbs(centered_saliency*255)
        saliency_cmapped = cv2.applyColorMap(saliency_bytes, cv2.COLORMAP_BONE).astype(img.dtype)/255.
    else:
        raise ValueError("Unsupported combining method, must be in ['jet', 'gray', 'bone']")
    blended = cv2.addWeighted(saliency_cmapped, alpha, img, 1-alpha, gamma)
    return blended, saliency_cmapped


def get_sample_for_pca(sample_size, dataset):
    # begin by computing pca directions and d_output_d_alphas
    if isinstance(dataset, torch.utils.data.Dataset):
        sample = []
        for _ in trange(sample_size):
            sample.append(dataset.generate_one()[0])
        print(set(x.shape for x in sample))
        sample = np.asarray(sample).squeeze().astype(np.float32)
    elif isinstance(dataset, BatchDataset):
        batch_size = dataset._batch_size.numpy()
        samples_needed = sample_size // batch_size
        print("samples need", samples_needed)
        sample = [s[0].numpy() for s, i in zip(dataset, range(samples_needed))]  # use zip to limit the iteration length
        print("samples list done")
        sample = np.concatenate(sample).astype(np.float32)
        print("did the sample fine")
    else:
        raise NotImplemented(f"Unknown dataset type {type(dataset)}")
    return sample

def find_pca_directions(dataset, scales, strides, sample=None, num_components=4, split_channels=False):
    if isinstance(sample, int):  # if a sample is not pre-provided, 'sample' is treated as the sample size to use
        sample = get_sample_for_pca(sample, dataset)
    sample_size = sample.shape[0]  # sample is [N, H, W, C]
    im_channels = sample.shape[-1] if sample.ndim == 4 else 1

    if isinstance(strides, int):
        strides = [strides]*len(scales)

    print("Got sample, beginning directions")
    pca_direction_grids = []
    for scale, stride in zip(scales, strides):
        windows = np.lib.stride_tricks.sliding_window_view(sample, (scale,scale), axis=(1,2)) 
        strided_windows = windows[:, ::stride, ::stride, :]  # [N, abs_posx, abs_posy, C?, within_windowx, within_windowy]

        pca_direction_grid = np.zeros((strided_windows.shape[1], strided_windows.shape[2], 
                                       num_components, scale, scale, im_channels))
        pca_fitter = decomposition.PCA(n_components=num_components, copy=False)
        scale_fitter = StandardScaler()
        for i in tqdm(range(strided_windows.shape[1])):
            for j in range(strided_windows.shape[2]):
                for c in range(im_channels):                          
                    if strided_windows.ndim == 6:  # ie. like [N, abs_posx, abs_posy, C, within_windowx, within_windowy]
                        if split_channels:
                            pca_selection = strided_windows[:, i, j, c] # [N, within_windowx, within_windowy]
                        else:
                            pca_selection = strided_windows[:, i, j].transpose(0,2,3,1) #[N, within_windowx, within_windowy, C]
                    elif strided_windows.ndim == 5: # ie. like [N, abs_posx, abs_posy, within_windowx, within_windowy]
                        pca_selection = strided_windows[:, i, j]  # [N, within_windowx, within_windowy]
                    flattened = pca_selection.reshape(sample_size, -1)
                    normalized = scale_fitter.fit_transform(flattened)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # gives pointless zero-division warnings
                        pca_fitter.fit(normalized)
                    for comp in range(num_components):
                        if split_channels:
                            pca_direction_grid[i, j, comp, ..., c] = pca_fitter.components_[comp].reshape(scale, scale)
                        else:
                            pca_direction_grid[i, j, comp] = pca_fitter.components_[comp].reshape(scale, scale, im_channels)
                    if not split_channels:  # if split_channels=False, we don't want to iterate over c
                        break
        pca_direction_grids.append(pca_direction_grid.copy())
    return pca_direction_grids


def pca_direction_grid_saliency(model, dataset, target_class, img, pca_direction_grids,
                        strides=None, gaussian=False, component=0, batch_size=32):
    # begin by computing d_output_d_alphas
    # note: previous version of pca_direction grids required passing scales as a parameter. It was removed 
    # since it can be inferred based on pca_direction_grids.
    # note: this function was renamed to pca_direction_grid_saliency from pca_direction_grids
    # gaussian was supposed to be you apply a gaussian kernel to d_out_d_alpha to blend things a bit better, never implemented
    model.eval()
    im_size = img.shape[0]
    scales = [grid.shape[-2] for grid in pca_direction_grids]
    if strides is None:
        strides = [1]
    if isinstance(strides, int):
        strides = [strides]*len(pca_direction_grids)

    d_out_d_alpha_grids = []
    interpolators = []
    indices_grid = np.mgrid[0:im_size, 0:im_size]

    stacked_img = np.repeat(np.expand_dims(img, 0), batch_size, axis=0)
    stacked_img = np.transpose(stacked_img, (0, 3, 1, 2)).astype(np.float32) # NCHW format
    img_tensor = dataset.implicit_normalization(torch.tensor(stacked_img).to(dataset.cfg.device))

    for s, (scale, stride) in enumerate(zip(scales, strides)):
        # centers are [scale//2, ..., im_size-scale//2-1], num_windows = im_size-scale+1
        # the -1 on the upper limit center c.f. the "last index" being im_size-1
        # the num_windows is correct because `(im_size-scale//2-1) - (scale//2) = (im_size-2*(scale-1)/2-1) = im_size-scale`
        # and num elements of the array is last-first+1
        index_windows = np.lib.stride_tricks.sliding_window_view(indices_grid, (scale,scale), axis=(1,2))

        xs = np.mgrid[0:im_size-scale:stride, 0:im_size-scale:stride]  # indexes into pca_direction_grids
        num_grid = xs.shape[1]
         # stride ratio can be used to account for scenarios where your pca_direction_grid was calculated with strides > 1
         # if we compute the saliency map with a stride >= to the stride of the pca_direction_grid, it is possible to compute
         # the saliency map. However, the stride we use to do the saliency map must be an integer multiple of the stride of
         # the pca direction grid so that we can select from the pca_direction_grid in a strided manner correctly
        stride_ratio = (num_grid*stride) / pca_direction_grids[s].shape[0]
        if not np.isclose(stride_ratio, int(stride_ratio)):
            raise ValueError(f"stride ratio between pca_direction_grids and the image must be an integer, is {stride_ratio}"
                             f" on {scale=}, {stride=}, {s=}, pca_grid_shape={pca_direction_grids[s].shape[0]}, {num_grid*stride=}")
        stride_ratio = int(stride_ratio)

        #print(xs, num_grid)
        d_out_d_alpha_grid = np.zeros((num_grid, num_grid))

        strided_indices = xs.transpose(1,2,0).reshape(-1, 2)
        unstrided_indices = np.mgrid[:num_grid, :num_grid].transpose(1,2,0).reshape(-1, 2)
        for k in tqdm(range(0, num_grid*num_grid, batch_size)):
            actual_batch_size = min(batch_size, num_grid*num_grid-k)
            # strided so that we index the parts of the image in a strided way
            batch_locs = strided_indices[k: k+actual_batch_size]  
            # unstrided so that we index into the contiguous/compressed (values we "skip" while striding aren't included in it) d_out_d_alpha_grid
            batch_unstrided_locs = unstrided_indices[k: k+actual_batch_size]

            pca_directions = pca_direction_grids[s][batch_locs[:,0]//stride_ratio, batch_locs[:,1]//stride_ratio, component]
            batch_window_indices = index_windows[:, batch_locs[:,0], batch_locs[:,1], ...]
            # print(batch_window_indices.shape)
            # print(batch_locs.shape)
            # print(batch_unstrided_locs.shape)
            # print(pca_directions.shape)
            # print(img_tensor[np.arange(actual_batch_size)[:,None,None], :, batch_window_indices[0], batch_window_indices[1]].shape)
            # return


            # do d_output_d_alpha computation
            alpha = torch.zeros((actual_batch_size,1,1,1), requires_grad=True).to(dataset.cfg.device)
            # implicit_normalization to put the pca direction in the same space as the image? (probably wrong?)
            direction_tensor = dataset.implicit_normalization(torch.tensor(pca_directions).to(dataset.cfg.device).float())
            img_tensor[np.arange(actual_batch_size)[:,None,None], :, batch_window_indices[0], batch_window_indices[1]] += alpha*direction_tensor
            output = model(img_tensor)  
            
            # sum since we only need the diagonal elements of the jacobian
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
    saliency_map = np.zeros((img.shape[0], img.shape[1])).astype(np.float32)
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

def visualize_pca_directions(pca_direction_grids, title, lines=True):
    # note: previous version of pca_direction grids required passing scales as a parameter. It was removed 
    # since it can be inferred based on pca_direction_grids
    scales = [grid.shape[-2] for grid in pca_direction_grids]
    window_shape = pca_direction_grids[0][0,0].shape
    num_components = window_shape[0]
    if len(window_shape) == 4:
        num_channels = window_shape[-1]
    else:
        num_channels = 1
    num_scales = len(pca_direction_grids)
    print("Components:", num_components)
    print("Channels:", num_channels)
    print("Scales:", num_scales)
    plt.figure(figsize=(5*num_scales, 6*num_channels*num_components))
    plt.title(title)
    for component in range(num_components):
        for channel in range(num_channels):
            for i, res in enumerate(pca_direction_grids):
                compressed_results = np.concatenate(np.concatenate(res[:, :, component, :, :, channel], 1), 1)
                subplot_idx = (component*num_scales*num_channels) + (num_scales*channel) + i+1
                plt.subplot(num_channels*num_components, num_scales, subplot_idx)
                line_width = scales[i] if lines else 0
                utils.imshow_centered_colorbar(compressed_results, "bwr",
                                f"Scale {scales[i]} Channel {channel} Component {component}", line_width=line_width)


def generate_many_pca(net, seeds, pca_directions_1_stride, scales, dataset, component=0, batch_size=128,
                      strides=None, skip_1_stride=False):
    _pca_map_s_strides = []
    _pca_map_1_strides = []
    _grad_maps = []
    _explain_imgs = []
    if strides is None:
        strides = scales
    for seed in seeds:
        dataset.rng.seed(seed)
        generated_img, label, *__ = dataset.generate_one()
        tensored_img = utils.tensorize(generated_img, device=dataset.cfg.device, requires_grad=True)
        grad_map = torch.autograd.grad(net(tensored_img)[0,label.argmax()], tensored_img)[0]
        pca_map_strided = pca_direction_grids(net, dataset, label.argmax(), generated_img,
                                              scales, pca_directions_1_stride, strides=strides,
                                              batch_size=batch_size, component=component)
        if not skip_1_stride:
            pca_map_1_stride = pca_direction_grids(net, dataset, label.argmax(), generated_img,
                                                   scales, pca_directions_1_stride, component=component,
                                                   batch_size=batch_size, strides=1)
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
        if dataset.cfg.num_classes == 2:
            activations = dataset.class_multiplier(target_class)*model(img_norm, logits=True)
        else:
            activations = model(img_norm)[:, target_class]
        activation_diff = (activations - baseline_activations).cpu().numpy().squeeze()
        finite_difference = np.clip(activation_diff/actual_diffs, -30, 30) # take absolute slope
        largest_slope = np.where(abs(finite_difference) > abs(largest_slope), finite_difference, largest_slope)
    return largest_slope

def finite_differences_map(model, dataset, target_class, img, unfairness="fair", values_prior=None,
                           batch_size=32, num_values=20):
    # generate a saliency map using finite differences method (iterate over colors)
    model.eval()
    im_size = img.shape[0]
    #img = img.astype(np.float32)/255. # normalization handled later
    indices = np.mgrid[:im_size, :im_size].transpose(1,2,0).reshape(im_size*im_size, -1)
    stacked_img = np.repeat(np.expand_dims(img, 0), batch_size, axis=0)
    stacked_img = np.transpose(stacked_img, (0, 3, 1, 2)).astype(np.float32) # NCHW format
    img_heat_map = np.zeros_like(img).astype(np.float32)

    with torch.no_grad():
        cuda_stacked_img = dataset.implicit_normalization(torch.tensor(stacked_img).to(dataset.cfg.device))
        if dataset.cfg.num_classes == 2:
            baseline_activations = dataset.class_multiplier(target_class)*model(cuda_stacked_img, logits=True)
        else:
            baseline_activations = model(cuda_stacked_img)[:, target_class]
        del cuda_stacked_img

    for channel in range(dataset.cfg.channels):
        for k in tqdm(range(0, im_size*im_size, batch_size)):
            actual_batch_size = min(batch_size, im_size*im_size-k)
            locations = indices[k:k+actual_batch_size]
            largest_slopes = finite_differences(model, dataset, target_class, stacked_img[:actual_batch_size],
                                                locations, channel, unfairness, values_prior, num_values,
                                                baseline_activations[:actual_batch_size])
            img_heat_map[locations[:,0], locations[:,1], channel] = largest_slopes
    return img_heat_map#.sum(axis=2)  # linear approximation aggregation?


# General visualizations


@torch.no_grad()
def rate_distribution(net, dataset, buckets=100, critical_values=[]):
    # Plot model error rates as function of color (for greyscale dataset)
    net.eval()
    total = np.zeros((buckets))
    num_correct = np.zeros((buckets))
    num_possible_colors = dataset.cfg.color_range[1] - dataset.cfg.color_range[0]
    for sample in tqdm(dataset.dataloader()):
        imgs = sample["image"].to(dataset.cfg.device).float()
        labels = sample["label"].to(dataset.cfg.device).float()
        actual_colors = sample["color"]
        color_indices = (buckets * (actual_colors - dataset.cfg.color_range[0]) / num_possible_colors).int().numpy()
        outputs = net(imgs)
        correct_preds = training.correct(outputs, labels).cpu().numpy()
        for i, color_idx in enumerate(color_indices):
            total[color_idx] += 1
            num_correct[color_idx] += correct_preds[i]

    # Plot results of rate_distribution
    num_wrong = total - num_correct
    width = 0.4
    labels = [int(x) for i, x in enumerate(np.linspace(*dataset.cfg.color_range, buckets))]
    plt.bar(labels, num_correct, width, label="correct amount")
    plt.bar(labels, num_wrong, width, bottom=num_correct, label="wrong amount")
    plt.vlines(critical_values, 0, np.max(total), linewidth=0.8,
               colors="r", label="decision boundary",
               linestyles="dashed")
    plt.legend()
    plt.xlabel("Color value")
    plt.show()


@torch.no_grad()
def response_graph(net, dataset, test_index=987_652, title=None, selection=None, img=None, class_select=None, use_arcsinh=True, no_plot=False, batch_size=64):
    # Plot network output logits as color varies for a specific image
    # class_select can be specified as a bool, indicating you want to take lbl.argmax() each time (and that you aren't passing in an img with selection)
    # or it can be an integer, indicating you want to look at that specific logit each time (and you have passed in the same img, with selection)
    net.eval()
    counterfactual_color_values = np.linspace(0, 255, 255)
    responses = []
    actual_selection = 0
    if img is not None:
        img = img.repeat(batch_size,1,1,1)
        actual_selection = len(selection)
        #print(actual_selection)
    else:
        batch_size = 1
    num_changed = 0  # what pct of the time do the changes to the pixels change the actual classification
    for color_idx in range(0, len(counterfactual_color_values), batch_size):
        actual_batch_size = min(batch_size, len(counterfactual_color_values)-color_idx)
        color_slice = slice(color_idx, color_idx+actual_batch_size)
        color = counterfactual_color_values[color_slice]

        if selection is None:
            dataset.rng.seed(test_index)
            generated_img, lbl, *__ = dataset.generate_one(set_color=color[0])
            img = utils.tensorize(generated_img, dataset.cfg.device)
        else:
            img[:actual_batch_size, 0, selection[:,0], selection[:,1]] = torch.tensor(color).to(dataset.cfg.device).float().unsqueeze(1)
            #plt.imshow(img[50].detach().cpu().numpy().squeeze(), cmap="gray")
        response = net(img, logits=True).cpu().numpy().squeeze()
        if class_select is not None:
            for elem,clr in zip(response, color):
                if dataset.color_classifier(clr) == elem.argmax():
                    num_changed += 1
                if isinstance(class_select, bool):
                    responses.append(np.expand_dims(elem[lbl.argmax()], 0))
                else:
                    responses.append(np.expand_dims(elem[class_select], 0))
        else:
            responses.append(response)

    responses = np.asarray(responses)
    if no_plot:
        return num_changed, actual_selection, responses

    if use_arcsinh:
        responses = np.arcsinh(responses)

    for output_logit in range(responses.shape[1]):
        plt.plot(counterfactual_color_values, responses[:, output_logit], label=f"class {output_logit}")

    plt.legend()
    plt.xlabel("Color value")
    plt.ylabel("Network output logit")
    #plt.vlines([100, 150], np.min(responses), np.max(responses), linewidth=0.8,
    #           colors="r", label="decision boundary",
    #           linestyles="dashed") # with .eval() works well
    utils.plot_color_classes(dataset, (np.min(responses), np.max(responses)), alpha=1.0)
    if title is not None:
        plt.title(title)
    return num_changed, actual_selection, responses



# Mechanistic Interpretation



def compute_profile_plot(profile, dataset):
    avg_line = []
    stds = []
    start = 0
    for c in range(255):
        if np.any(profile[:,0]==c):
            selection = profile[:,1][profile[:,0]==c]
            avg_line.append(selection.mean())
            stds.append(selection.std())

        else:
            if avg_line:
                avg_line.append(avg_line[-1]) # assume previous value continues
            else:
                start += 1  # or just start elsewhere
    avg_line = np.asarray(avg_line)

    color_values = np.arange(start, 255)
    classes = np.vectorize(dataset.color_classifier)(color_values)/(dataset.cfg.num_classes-1)
    return avg_line, color_values, classes, profile, stds


def get_profile_y_lims(profile_plot): # assumes y values are all positive
    avg_line, color_values, classes, profile, stds = profile_plot

    max_y = min(max(avg_line)*(1+0.3*np.sign(max(avg_line))), max(profile[:,1]))
    min_y = max(min(avg_line)*(1-0.3*np.sign(min(avg_line))), min(profile[:,1]))
    return min_y, max_y


def show_profile_plot(profile_plot, hide_ticks=False, hide_y_ticks=False, ax=None, rm_border=False, y_lims=None, bands=True):
    if ax is None:
        ax = plt.gca()

    avg_line, color_values, classes, profile, stds = profile_plot

    if y_lims is None:
        min_y, max_y = get_profile_y_lims(profile_plot)
    else:
        min_y, max_y = y_lims
    scaled_classes = classes*(max_y-min_y) + min_y

    if bands:
        colors = {class_value: clr for class_value,clr in zip(sorted(list(set(classes))), ["g", "m", "y"])}
        class_names = {class_value: f"Class {name}" for class_value,name in zip(sorted(list(set(classes))), range(500))}
        seen_colors = set()
        start_x = 0
        curr_value = classes[0]
        for pos_now, class_val in enumerate(classes):
            if class_val != curr_value:
                lbl = None
                if curr_value not in seen_colors:
                    lbl = class_names[curr_value]
                    seen_colors.add(curr_value)
                ax.fill_betweenx((min_y, max_y), start_x, pos_now, color=colors[curr_value], alpha=0.2, label=lbl)
                curr_value = class_val
                start_x = pos_now
        ax.fill_betweenx((min_y, max_y), start_x, pos_now+10, color=colors[curr_value], alpha=0.2)
        plt.legend()

    else:
        ax.plot(color_values, scaled_classes, alpha=0.6, c="c")
    ax.scatter(profile[:,0], profile[:,1], s=0.5, marker="o", alpha=0.08, c="m")
    #ax.fill_between(color_values, avg_line-stds, avg_line+stds, color="k", alpha=0.5)
    ax.plot(color_values, avg_line, c="r")

    ax.set_ylim(min_y-(max_y-min_y)*0.01, max_y+(max_y-min_y)*0.01)
    if hide_ticks:
        ax.set_xticks([])
    if hide_y_ticks:
        ax.set_yticks([])
        
    if rm_border:
        if hide_y_ticks:
            utils.remove_borders(ax, ["top", "bottom", "right", "left"])
        else:
            utils.remove_borders(ax, ["top", "bottom", "right"])#, "left"])
        


def show_profile_plots(color_profile, names, hide_ticks=True, rm_border=True, fixed_height=False, size_mul=1.):
    if isinstance(names, str):  # ie. expand names based on root name
        names = [k for k in color_profile if names in k]

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
            if w == 0:
                axes[h,w].set_ylabel("Average channel activation")
            if h == shape[0]-1 and w==shape[1]//2:
                axes[h,w].set_xlabel("Image Intensity")
            name = w+h*shape[1]
            show_profile_plot(color_profile[names[name]], hide_y_ticks=(w!=0 and rm_border), bands=True,
                              hide_ticks=hide_ticks, ax=axes[h,w], rm_border=rm_border, y_lims=y_lims)


@torch.no_grad()
def activation_color_profile(net, dataset):
    # assume it already has AllActivations hooks on it
    net.eval()
    filter_activations = defaultdict(list) # conv filter: [(color, sum_activations), ...]
    for sample in tqdm(dataset.dataloader()):
        imgs = sample["image"].to(dataset.cfg.device).float()
        #labels = sample["label"].to(device).float()
        colors = sample["color"]
        net(imgs)

        for layer_name, activation in net._features.items():
            layer_type = layer_name.split(".")[-1]  # truly horrible code
            # if layer_type not in ["fully_connected", "batch_norm1", "batch_norm2"]:  # look at fc logits, and post-batchnorms
            #     continue
            for b, color in enumerate(colors):  # b corresponds to batch dimension
                for channel in range(activation.shape[1]):  # for fc layers, this gives the value per logit (mean of 1 value)
                    entry = color.item(), activation[b][channel].mean().item() # could do group conv to give data on input channels?
                    filter_activations[f"{layer_name}_{channel}"].append(entry)  # should be mean so that activation doesn't scale with map size

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


def get_weight(net, weight_name, merge_batchnorms=False):
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
        parent_mod = curr_module
        try:
            curr_module = curr_module[int(attr)]
        except ValueError:
            curr_module = getattr(curr_module, attr)
    if merge_batchnorms:  # combine the convolution with the scaling effect of the batchnorm (still ignores biases)
        bn = getattr(parent_mod, attr.replace("conv", "batch_norm"))
        #print("parent", parent_mod, "parent")
        #print(bn)
        scale_factor = bn.weight / torch.sqrt(bn.running_var)
        #print(scale_factor.shape)
        return (curr_module.weight * scale_factor[:, None, None, None]).detach().cpu().numpy()
    return curr_module.weight.detach().cpu().numpy()

def show_fully_connected(net, weight_name):
    pass

def show_conv_weights(net, weight_name, color_profile=None, outer_shape=None, size_mul=6, 
                      rm_border=True, fixed_height=False, full_gridspec=False, merge_batchnorms=True,
                      show_scale=True):
    # assume net it already has AllActivations hooks on it
    # implicitly assume that channel-aligned view is salient for the given network
    weights = get_weight(net, weight_name, merge_batchnorms=merge_batchnorms)   # N_out, N_in, K, K
    if outer_shape is None:
        outer_shape = utils.find_good_ratio(weights.shape[0])
    inner_shape = utils.find_good_ratio(weights.shape[1])
    map_size = weights.shape[-1]
    def reshaper(w):
        grid_w = w.reshape(*inner_shape, map_size, map_size)
        if full_gridspec:
            return grid_w
        #print("gw", grid_w.shape)
        return np.concatenate(np.concatenate(grid_w, 1), 1)
    show_weights(weights, weight_name, reshaper, inner_shape, outer_shape, map_size, color_profile, 
                 size_mul, rm_border, fixed_height, show_scale=show_scale, full_gridspec=full_gridspec)


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
                 rm_border, fixed_height, selection=None, full_gridspec=False, show_scale=False):
    if selection is None:
        selection = np.arange(outer_shape[0]*outer_shape[1])
    if not isinstance(size_mul, tuple):
        size_mul = (size_mul, size_mul)


    figsize = map_size*outer_shape[1]*inner_shape[1]/8*size_mul[1], map_size*outer_shape[0]*inner_shape[0]/10*size_mul[0]
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
                if len(sub_axes.shape) == 1:
                    sub_axes = np.expand_dims(sub_axes, 0)
                for i in range(inner_shape[0]):
                    for j in range(inner_shape[1]):
                        #print(weights.shape, weights[channel].shape, "reshaper(w[c])")
                        utils.imshow_centered_colorbar(reshaper(weights[channel])[i,j], colorbar=show_scale,
                                                    ax=sub_axes[i,j], rm_border=False)
                axes[gs_idx].set_xticks([])
                axes[gs_idx].set_yticks([])
                if rm_border:
                    utils.remove_borders(axes[gs_idx])
            else:
                utils.imshow_centered_colorbar(reshaper(weights[channel]), line_width=map_size if map_size > 1 else 0,
                                        colorbar=show_scale, ax=axes[gs_idx], rm_border=rm_border)
            if color_profile:
                gs_idx = (h*2+1,w) 
                if w == 0:
                    axes[gs_idx].set_ylabel("Response")
                axes[gs_idx].set_xlabel("Image Intensity")
                show_profile_plot(color_profile[f"{weight_name}_{selection[channel]}"],
                                  hide_ticks=True, ax=axes[gs_idx], rm_border=True, y_lims=y_lims)


# only works when running locally (and should probably just do it manually instead with like surfaces and a bunch of math)
#def mlab_fc_conv_feature_angles(net, layer_name, num_embed=3, normalize=True):
#    # get conv->fc weight/feature angles
#    weights = get_weight(net, layer_name)  # (In, Out)
#    shape = net.final_img_shape
#    shaped_w = weights.reshape(-1, shape[2], shape[0]*shape[1])  # C_out, C_in, HW
#    mags = np.linalg.norm(shaped_w, axis=-1)[..., np.newaxis]  # 3, 6, 64
#    norm_w = shaped_w/mags if normalize else shaped_w
#    dots = np.einsum("kji,lji->jkl", norm_w, norm_w)
#
#    if num_embed not in [2,3]:
#        return dots
#
#    def gram_diff(w, desired_grams):
#        w_mat = w.reshape(*embed_shape)
#        grams = w_mat.dot(w_mat.T)
#        return ((grams - desired_grams)**2).sum()
#    embed_shape = (norm_w.shape[0], num_embed)
#    embedded_weights = np.zeros((norm_w.shape[1], *embed_shape))
#    closeness = np.zeros(norm_w.shape[1])
#    for i, feats in enumerate(norm_w.transpose(1,0,2)): # 3, 64
#        projection = minimize(gram_diff, np.ones(num_embed*embed_shape[0]),
#                              args=(dots[i],))
#        embedded_weights[i] = projection.x.reshape(*embed_shape)
#        closeness[i] = projection.fun
#
#    #grid_shape = utils.find_good_ratio(norm_w.shape[1])
#    colors = ["coral", "forestgreen", "royalblue"]
#    labels = ["class "+x for x in "012"]
#    origin = np.zeros(embed_shape[0])
#    for w in embedded_weights:
#        #lims = np.max(abs(w))
#        if num_embed == 3:
#            print(w.shape, origin.shape)
#            obj = mlab.quiver3d(origin, origin, origin, w[:,0], w[:,1], w[:,2])#, color=colors[k])#, label=labels[k])
#            obj.actor.property.interpolation = "phong"
#            obj.actor.property.specular = 0.1
#            obj.actor.property.specular_power = 5
#            return obj
#        #else:
#        #    ax = fig.add_subplot(gs[i,j])
#        #    for k in range(embed_shape[0]):
#        #        ax.quiver(0, 0, w[k,0], w[k,1], scale=2.0, color=colors[k], label=labels[k])
#        #ax.set_xticklabels([])
#        #ax.set_yticklabels([])
#        #ax.set_title(f"{int(closeness[feature]*1000.)}")
#        #ax.set_xlim(-lims, lims)
#        #ax.set_ylim(-lims, lims)
#    #fig.legend(*ax.get_legend_handles_labels())
#    return dots, embedded_weights


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



# Region Importance



@torch.no_grad()
def random_pixels_response(net, dataset, num_pixels, img_id=987_652, one_class=True, no_plot=False, batch_size=128):
    net.eval() # very important!
    dataset.rng.seed(img_id)
    generated_img, lbl, color, size, pos  = dataset.generate_one()
    tensor_img = utils.tensorize(generated_img, device=dataset.cfg.device)

    im_size = generated_img.shape[0]
    possible_pixels = np.mgrid[:im_size, :im_size].transpose(1,2,0).reshape(im_size*im_size, -1)
    selected_pixels = possible_pixels[np.random.choice(len(possible_pixels), num_pixels, replace=False)]
    num_inside = 0
    for p in selected_pixels:
        if np.linalg.norm(p-pos) < size:
            num_inside += 1
    if not no_plot:
        print(f"Percent of random inside circle: {num_inside/num_pixels*100.}")
    class_select = lbl.argmax() if one_class else None
    return response_graph(net, dataset, title="Randomly selected pixels",
                   selection=selected_pixels, img=tensor_img, class_select=class_select, use_arcsinh=False, no_plot=no_plot, batch_size=batch_size)


@torch.no_grad()
def circle_pixels_response(net, dataset, num_pixels, width, img_id=987_652, outer=True, one_class=True, no_plot=False, batch_size=128):
    # random selected pixels must be inside circle, but within "width" of the edge if outer==True
    # but greater than "width" from the edge if outer==False
    net.eval() # very important!
    dataset.rng.seed(img_id)  # generate image
    generated_img, lbl, color, size, pos = dataset.generate_one()
    tensor_img = utils.tensorize(generated_img, device=dataset.cfg.device)

    size = size[0]
    num_angles = 500
    num_radii = 50
    angle = np.linspace(0, 2*np.pi, num_angles)
    if width is None:
        radii = np.linspace(0, size, num_radii)
    elif outer:
        radii = np.linspace(size-width, size, num_radii)
        circle_area = size**2
        area_pct = (circle_area - (size-width)**2) / circle_area
        if not no_plot:
            print(f"Only using points within {width} pixels of the outer edge ({area_pct*100.:.2f}% of circle area)")
    else:
        circle_area = size**2
        area_pct = (size-width)**2 / circle_area
        if not no_plot:
            print(f"Only using points further than {width} pixels from the outer edge ({area_pct*100.:.2f}% of circle area)")
        radii = np.linspace(0, size-width, num_radii)
    possible_indices = np.zeros((num_radii*num_angles, 2))
    possible_indices[:,0] = (pos[0][0] + np.cos(angle)*radii[:,None]).flat
    possible_indices[:,1] = (pos[0][1] + np.sin(angle)*radii[:,None]).flat
    possible_indices = np.round(possible_indices).astype(np.int64)  # all possible pixels we can select (cant do np.unique because slow
    im_size = generated_img.shape[0]
    zero_img = np.zeros((im_size, im_size))
    #print(possible_indices.shape)
    zero_img[possible_indices[:,0], possible_indices[:,1]] = 1
    possible_pixels = np.nonzero(zero_img)
    #print(len(possible_pixels[0]), len(possible_pixels[1]))
    #print(len(possible_pixels[0]))
    selected_indices = np.random.choice(len(possible_pixels[0]), min(num_pixels, len(possible_pixels[0])), replace=False)
    selected_pixels = np.zeros((len(selected_indices), 2))
    selected_pixels[:,0] = possible_pixels[0][selected_indices]
    selected_pixels[:,1] = possible_pixels[1][selected_indices]
    #print(selected_pixels.shape)

    class_select = lbl.argmax() if one_class else None
    return response_graph(net, dataset, title="Pixels inside circle",
                   selection=selected_pixels, img=tensor_img, class_select=class_select, use_arcsinh=False, no_plot=no_plot, batch_size=batch_size)


def both_pixels_response(net, dataset, num_pixels, width, one_class, img_id=987_652, outer=True, no_plot=False, batch_size=128):
    if not no_plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
    res1 = circle_pixels_response(net, dataset, num_pixels, width, one_class=one_class, img_id=img_id, outer=outer, no_plot=no_plot, batch_size=batch_size)
    if not no_plot:
        plt.subplot(1,2,2)
    res2 = random_pixels_response(net, dataset, num_pixels, img_id=img_id, one_class=one_class, no_plot=no_plot, batch_size=batch_size)
    return res1, res2


def region_importance_exp(net, dataset, num_pixels, width, batch_size=128, runs=1000, verbose=False):
    changed_rand = changed_inner = changed_outer = 0  # num times that the pixels selected actually altered the class prediction
    total_rand = total_inner = total_outer = 0 # num pixels actually selected/changed
    run_iter = tqdm(range(runs)) if verbose else range(runs)
    for _ in run_iter:
        img_id = np.random.randint(0,500_000)
        inner_res, rand_res = both_pixels_response(net, dataset, num_pixels, width, img_id=img_id,
                         one_class=True, outer=False, no_plot=True, batch_size=batch_size)
        outer_res, rand_res = both_pixels_response(net, dataset, num_pixels, width, img_id=img_id,
                         one_class=True, outer=True, no_plot=True, batch_size=batch_size)
        changed_rand += rand_res[0]
        changed_inner += inner_res[0]
        changed_outer += outer_res[0]
        total_rand += rand_res[1]
        total_inner += inner_res[1]
        total_outer += outer_res[1]
    total_amt = runs*255
    _pct = lambda c: c/total_amt*100.
    if verbose:
        print("Inner circle:", _pct(changed_inner))
        print("Outer circle:", _pct(changed_outer))
        print("Random baseline:", _pct(changed_rand))
    return _pct(changed_inner), total_inner/runs, _pct(changed_outer), total_outer/runs, _pct(changed_rand), total_rand/runs


def region_importance(net, dataset, batch_size=128, runs=1000):
    widths = list(range(3,20,3))
    #pixel_nums = list(range(10,100,10))
    #pixel_nums.extend(list(range(100,1000,100)))
    #pixel_nums.extend(list(range(1000,10000,1000)))
    full_results = []
    for w in tqdm(widths):
        full_results.append(region_importance_exp(net, dataset, 1, w, batch_size=batch_size, runs=runs))
    full_results = np.asarray(full_results)
    return full_results, widths


def plot_region_importance(full_results, widths, pixel_nums):
    colors = ["coral", "forestgreen", "royalblue"]
    plt.figure(figsize=(15,15))
    types = ["Inner", "Outer", "Rand"]
    shape = utils.find_good_ratio(len(widths))
    for i, w in enumerate(widths):
        plt.subplot(*shape, i+1)
        plt.xscale("log")
        plt.xlabel("Log10(num_pixels)")
        plt.ylabel("%Altered")
        plt.title(f"Width {w}")
        for j, (pixel_type, color) in enumerate(zip(types,colors)):
            plt.scatter(full_results[i, :, j*2+1], full_results[i, :, j*2], color=color, label=pixel_type)
        if i//shape[1] == 0 and i%shape[1] == shape[1]-1:
            plt.legend()



# Network training curve plots



def defaultdict_helper():  # to make things pickle-able
    return defaultdict(int)


def final_activation_tracker(net, loss, dataset):
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
    for i, sample in tqdm(enumerate(dataset.dataloader())):
        imgs = sample["image"].to(dataset.cfg.device).float()
        labels = sample["label"].to(dataset.cfg.device)
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
