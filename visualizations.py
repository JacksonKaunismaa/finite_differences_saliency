import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.interpolate import RegularGridInterpolator
import torch
from tqdm import tqdm
from network import correct  # for rate_distribution (reads it as a global variable)
import utils
from utils import *
import matplotlib.pyplot as plt
import warnings


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
    T = False
    for scale, stride in zip(scales, strides):
        windows = np.lib.stride_tricks.sliding_window_view(sample, (scale,scale), axis=(1,2))
        strided_windows = windows[:, ::stride, ::stride, :]  # [N, H, W, C]
                
        xs = np.mgrid[scale:im_size:stride]
        num_grid = xs.shape[0]
        pca_direction_grid = np.zeros((num_grid, num_grid, num_components, scale, scale, dataset.channels))
        
        pca_fitter = decomposition.PCA(n_components=scale**2, copy=False)
        scale_fitter = StandardScaler()
        for i in tqdm(range(num_grid)):
            for j in range(num_grid):
                # find pca direction for current patch
                #if i == 0 and j == 7 and scale == 3:
                #    T = True
                pca_selection = strided_windows[:, i, j, :]
                flattened = pca_selection.reshape(sample_size, -1)
                normalized = scale_fitter.fit_transform(flattened)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # gives pointless zero-division warnings
                    pca_fitter.fit(normalized)
                #if pca_fitter.components_[0].sum() < 0:
                #    T = True
                #tprint("pca_selection", pca_selection, pca_selection.shape, t=T)
                #tprint("flattened", flattened, flattened.shape, t=T)
                #tprint(pca_fitter.components_[0], pca_fitter.components_[0].shape, t=T)
                #tprint(pca_fitter.components_[1], t=T)
                #tprint(i,j)
                #if T:
                #    return
                for comp in range(num_components):
                    pca_direction_grid[i, j, comp] = pca_fitter.components_[comp].reshape(scale, scale, dataset.channels)

        pca_direction_grids.append(pca_direction_grid.copy())    
    return pca_direction_grids
    
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
        index_windows = np.lib.stride_tricks.sliding_window_view(indices_grid, (scale,scale), axis=(1,2))        

        xs = np.mgrid[scale:im_size:stride, scale:im_size:stride]
        num_grid = xs.shape[0]
        d_out_d_alpha_grid = np.zeros((num_grid, num_grid))
        
        strided_indices = xs.transpose(1,2,0).reshape(-1, 2)  # ie should always pass strides=1 pca_directions into this
        unstrided_indices = np.mgrid[:num_grid, :num_grid].transpose(1,2,0).reshape(-1, 2)
        for k in tqdm(range(0, num_grid*num_grid, batch_size)):
            actual_batch_size = min(batch_size, num_grid*num_grid-k)
            batch_locs = strided_indices[k: k+actual_batch_size]  # for indexing into a full image (im_size, im_size), which we do with strides
            batch_unstrided_locs = unstrided_indices[k: k+actual_batch_size]  # for indexing into a dense grid (num_grid, num_grid)

            pca_directions = pca_direction_grids[s][batch_locs[:,0], batch_locs[:,1], component]
            batch_window_indices = index_windows[:, batch_locs[:,0], batch_locs[:,1], ...]

            # do d_output_d_alpha computation
            alpha = torch.zeros((actual_batch_size,1,1,1), requires_grad=True).to(device)
            direction_tensor = dataset.implicit_normalization(torch.tensor(pca_directions).to(device).float())
            img_tensor[np.arange(actual_batch_size)[:,None,None], :, window_indices[0], window_indices[1]] += alpha*direction_tensor
            output = model(img_tensor)  # sum since gradient will be back-proped as vector of 1`s

            d_out_d_alpha = torch.autograd.grad(output[:,target_class].sum(), alpha)[0].squeeze()
            model.zero_grad()
            d_out_d_alpha_grid[batch_unstrided_locs[:,0], batch_unstrided_locs[:,1]] = d_out_d_alpha.detach().cpu().numpy()
        
        d_out_d_alpha_grids.append(d_out_d_alpha_grid.copy())
        interpolators.append(RegularGridInterpolator((xs[0], xs[1]), d_out_d_alpha_grid, 
                                                     bounds_error=False, fill_value=None))
    # now, per pixel, interpolate what the d_output_d_alpha value would be if the window
    # were centered at that pixel, then take the max over all possible scales
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
                    best_scale = scale
            saliency_map[i,j] = best_d_out_d_alpha
            scale_wins[s] += 1
    print(scale_wins)
    return saliency_map  # try jacobian with respect to window itself (isnt this just the gradient?)

def visualize_pca_directions(pca_direction_grid_scales, title, scales, component=0, lines=True):
    window_shape = pca_direction_grid_scales[0][0,0].shape
    if len(window_shape) == 4:
        num_channels = window_shape[-1]
    else:
        num_channels = 1
    num_scales = len(pca_direction_grid_scales)
    plt.figure(figsize=(5*num_scales, 6*num_channels))
    plt.title(title)
    for channel in range(num_channels):
        for i, res in enumerate(pca_direction_grid_scales):
            compressed_results = np.concatenate(np.concatenate(res[:, :, component, :, :, channel], 1), 1)
            img_size = compressed_results.shape[0]
            #print("doing", i, channel, "drawing", list(range(scales[i], img_size, scales[i])))
            plt.subplot(num_channels, num_scales, num_scales*channel+i+1)
            imshow_centered_colorbar(compressed_results, "bwr", f"Scale {scales[i]} Channel {channel}")
            if lines:
                plt.vlines(list(range(scales[i], img_size, scales[i])), linestyle="dotted",
                        ymin=0, ymax=img_size, colors="k")
                plt.hlines(list(range(scales[i], img_size, scales[i])), linestyle="dotted",
                        xmin=0, xmax=img_size, colors="k")
                plt.xlim(0,img_size)
                plt.ylim(0,img_size)



# Finite differences stuff



@torch.no_grad()
def finite_differences(model, dataset, target_class, stacked_img, locations, channel, 
        unfairness, values_prior, num_values, device, baseline_activations):
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
            activations = class_multiplier*model(img_norm, logits=True)
        else:
            activations = model(img_norm)[:, target_class]
        activation_diff = (activations - baseline_activations).cpu().numpy().squeeze()
        finite_difference = np.clip(activation_diff/actual_diffs, -30, 30) # take absolute slope
        largest_slope = np.where(abs(finite_difference) > abs(largest_slope), finite_difference, largest_slope)
    return largest_slope

def finite_differences_map(model, dataset, target_class, img, 
        device=None, unfairness="fair", values_prior=None, batch_size=32, num_values=20):
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
            class_multiplier = 1 if target_class == 1 else -1
            baseline_activations = class_multiplier*model(cuda_stacked_img, logits=True)
        else:
            baseline_activations = model(cuda_stacked_img)[:, target_class]
        del cuda_stacked_img

    for channel in range(dataset.channels):
        for k in tqdm(range(0, im_size*im_size, batch_size)):
            actual_batch_size = min(batch_size, im_size*im_size-k)
            locations = indices[k:k+actual_batch_size]
            largest_slopes = finite_differences(model, dataset, target_class, stacked_img[:actual_batch_size],
                    locations, channel, unfairness, values_prior, num_values, device, baseline_activations[:actual_batch_size])
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
        generated_img = tensorize(generated_img, device)
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
