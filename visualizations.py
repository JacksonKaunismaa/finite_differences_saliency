import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.interpolate import RegularGridInterpolator
import torch
from tqdm import tqdm
from network import correct  # for rate_distribution (reads it as a global variable)
import matplotlib.pyplot as plt


# PCA Stuff


def find_pca_directions(dataset, sample_size, scales, strides):
# begin by computing pca directions and d_output_d_alphas
    sample = []
    for _ in range(sample_size):
        sample.append(dataset.generate_one()[0])
    sample = np.array(sample).squeeze()
    im_size = dataset.size
    
    if isinstance(strides, int):
        strides = [strides]*len(scales)
    
    pca_direction_grids = []
    for scale, stride in zip(scales, strides):
        windows = np.lib.stride_tricks.sliding_window_view(sample, (scale,scale), axis=(1,2))
        strided_windows = windows[:, ::stride, ::stride, ...]
                
        xs = np.mgrid[scale:im_size:stride]
        num_grid = xs.shape[0]
        pca_direction_grid = np.zeros((num_grid, num_grid, scale, scale))
        
        pca_fitter = decomposition.PCA(n_components=scale**2, copy=False)
        scale_fitter = StandardScaler()
        for i in tqdm(range(num_grid)):
            for j in range(num_grid):
                # find pca direction for current patch
                pca_selection = strided_windows[:, i, j, ...]
                flattened = pca_selection.reshape(sample_size, -1)
                normalized = scale_fitter.fit_transform(flattened)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # gives pointless zero-division warnings
                    pca_fitter.fit(normalized)
                pca_direction = pca_fitter.components_[0].reshape(scale,scale)
                                
                pca_direction_grid[i,j,...] = pca_direction
        pca_direction_grids.append(pca_direction_grid.copy())    
    return pca_direction_grids
    
def pca_direction_grids(model, dataset, target_class, img, 
                        sample_size=512, scales=None, 
                        strides=None, pca_direction_grids=None,
                       gaussian=False, device=None):
    # begin by computing pca directions and d_output_d_alphas
    model.eval()
    im_size = dataset.size
    if scales is None:
        scales = default_scales
    if strides is None:
        strides = default_scales
    if isinstance(strides, int):
        strides = [strides]*len(pca_direction_grids)
    if pca_direction_grids is None:
        pca_direction_grids = find_pca_directions(dataset, sample_size, scales, strides)
    d_out_d_alpha_grids = []
    interpolators = []
    indices_grid = np.mgrid[0:im_size, 0:im_size]

    for s, (scale, stride) in enumerate(zip(scales, strides)):
        index_window = np.lib.stride_tricks.sliding_window_view(indices_grid, (scale,scale), axis=(1,2))        
        strided_indices = index_window[:, ::stride, ::stride]
        
        xs = np.mgrid[scale:im_size:stride]
        num_grid = xs.shape[0]
        d_out_d_alpha_grid = np.zeros((num_grid, num_grid))
        for i in tqdm(range(num_grid)):
            for j in range(num_grid):
                # get pca direction for current patch
                pca_direction = pca_direction_grids[s][i,j]
                indices = strided_indices[:, i, j, ...]  # will have to slice these

                # do d_output_d_alpha computation
                alpha = torch.tensor(0.0, requires_grad=True).to(device)
                direction_tensor = torch.tensor(pca_direction).to(device).float()
                img_tensor = torch.tensor(img.transpose(2,0,1)).to(device).float().unsqueeze(0)
                img_tensor[..., indices[0,0,0]:indices[0,-1,-1]+1, indices[1,0,0]:indices[1,-1,-1]+1] += alpha*direction_tensor
                output = model(img_tensor)[0,target_class]
                d_out_d_alpha = torch.autograd.grad(output, alpha)[0]
                # try guided backprop
                d_out_d_alpha_grid[i,j] = d_out_d_alpha.detach().cpu().numpy()
        d_out_d_alpha_grids.append(d_out_d_alpha_grid.copy())
        interpolators.append(RegularGridInterpolator((xs, xs), d_out_d_alpha_grid, 
                                                     bounds_error=False, fill_value=None))
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
                    best_scale = scale
            saliency_map[i,j] = best_d_out_d_alpha
            scale_wins[s] += 1
    print(scale_wins)
    return saliency_map  # try jacobian with respect to window itself (isnt this just the gradient?)

def visualize_pca_directions(pca_direction_grid_scales):
    plt.figure(figsize=(len(pca_direction_grid)*4, 12))
    for i, res in enumerate(pca_directions_grid_scales):
        compressed_results = np.concatenate(np.concatenate(res, 1), 1)
        plt.subplot(1,len(pca_directions_grid_scales),i+1)
        if i == 0:
            plt.title("Strided windows")
        plt.imshow(compressed_results, cmap="gray")


# Finite differences stuff



@torch.no_grad()
def finite_differences(model, dataset, target_class, stacked_img, locations, channel, 
        unfairness, values_prior, num_values, device):
    cuda_stacked_img = torch.tensor(stacked_img).to(device)
    if dataset.num_classes == 2:
        class_multiplier = 1 if target_class == 1 else -1
        baseline_activations = class_multiplier*model(cuda_stacked_img, logits=True)
    else:
        baseline_activations = model(cuda_stacked_img)[:, target_class]
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
        img_norm = torch.tensor(shift_img).to(device) # best is no normalization anyway
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
    values_x = np.repeat(np.arange(im_size), im_size)
    values_y = np.tile(np.arange(im_size), im_size)
    indices = np.stack((values_x, values_y), axis=1)
    stacked_img = np.repeat(np.expand_dims(img, 0), batch_size, axis=0)
    stacked_img = np.transpose(stacked_img, (0, 3, 1, 2)).astype(np.float32) # NCHW format
    img_heat_map = np.zeros_like(img)
    for channel in range(dataset.channels):
        for k in tqdm(range(0, im_size*im_size, batch_size)):
            actual_batch_size = min(batch_size, im_size*im_size-k+batch_size)
            locations = indices[k:k+batch_size]
            largest_slopes = finite_differences(model, dataset, target_class, 
                    stacked_img, locations, channel, unfairness, values_prior, num_values, device)
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
        generated_img = np.expand_dims(generated_img, 0).transpose(0, 3, 1, 2)
        generated_img = torch.tensor(generated_img).to(device).float()
        response = net(torch.tensor(generated_img).to(device).float(), logits=True).cpu().numpy()
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
