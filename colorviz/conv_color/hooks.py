import torch
import torch.nn as nn
import time

def traverse_modules(func, mod, curr_name="", **kwargs):
    has_sub_mods = False
    for name, sub_mod in mod.named_children():
        traverse_modules(func, sub_mod, curr_name=f"{curr_name}{name}.", **kwargs)
        has_sub_mods = True
    if not has_sub_mods: # if no submodules, it is an actual operation
        func(mod, curr_name, **kwargs)


class HookAdder(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.handles = []
        self.hooks_exist = False
        self.verbose = False
        for k,v in kwargs.items():
            setattr(self, k, v)

    def setup_hooks(self):
        if self.hooks_exist:
            return
        def add_hook(mod, curr_name, _self):
            for hook_type, hook_func in zip(_self.hook_types, _self.hook_funcs):
                mod_hook = getattr(mod, hook_type)
                generated_hook = getattr(_self, hook_func)(curr_name[:-1])
                if _self.verbose:
                    print("Adding", hook_func, "to", curr_name[:-1], "on", hook_type)
                if generated_hook:
                    _self.handles.append(mod_hook(generated_hook))
        traverse_modules(add_hook, self.model, _self=self)
        self.hooks_exist = True

    def clean_up(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.hooks_exist = False

    def forward(self, *args, preserve_hooks=False):
        self.setup_hooks()
        try:
            result = self.model(*args)
        finally:
            if not preserve_hooks:
                self.clean_up()
        return result




class ProfileExecution(HookAdder):
    def __init__(self, model, **kwargs):
        # in order for this hacky mess to actually work, you need to profile_net.benchmark() before doing .forward()
        self.hook_types = ["register_forward_hook"]
        self.hook_funcs = ["benchmark_hook"]

        self.gamma = 0.99  # can override these with kwargs
        self.verbose = False
        self.profile = True  # use to toggle whether profiling happens

        super().__init__(model)
        self.prev_time = 0
        self.stats = {}

    def benchmark(self, point=None): # not thread safe at all
        if not self.profile:
            return
        if point is not None:
            torch.cuda.synchronize()
            time_taken = time.perf_counter() - self.prev_time
            if point not in self.stats:
                self.stats[point] = [time_taken, 0]  # avg_time, num_times
            self.stats[point][1] += 1
            self.stats[point][0] = self.stats[point][0]*self.gamma + time_taken*(1-self.gamma)
            if self.verbose:
                print(f"took {time_taken} to reach {point}, ewma={self.stats[point]}")
        self.prev_time = time.perf_counter()

    def __repr__(self):
        sum_avgs = sum([x[0] for x in self.stats.values()])
        sum_time = sum([x[0]*x[1] for x in self.stats.values()])
        ret_str = "point\tpct_avg\tpct_cumulative"
        for point,stat in self.stats.items():
            ret_str += f"\n{point}\t{stat[0]/sum_avgs*100.:.3f}%\t{stat[0]*stat[1]/sum_time*100.:.3f}%"
        return ret_str

    def benchmark_hook(self, name):
        def fn(layer, inpt, outpt):
            self.benchmark(name)
        return fn

class AllActivations(HookAdder):
    def __init__(self, model, track_grads=None, **kwargs):
        self._features = {}
        self._grads = {}
        self.hook_types = ["register_forward_hook"]
        self.hook_funcs = ["save_activations_hook"]
        if track_grads:
            self.track_grads = track_grads  # list of names
            self.hook_types.append("register_backward_hook")
            self.hook_funcs.append("save_grad_hook")
        super().__init__(model, **kwargs)

    def save_activations_hook(self, name):
        def fn(layer, inpt, output):
            self._features[name] = output
        return fn

    def save_grad_hook(self, name):
        if name in self.track_grads:
            def fn(layer, grad_in, grad_out):
                self._grads[name] = grad_out[0].detach().cpu().numpy()
            return fn
        return None


# adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py
class GuidedBackprop(HookAdder):
    def __init__(self, model, exceptions=None, **kwargs):
        self.forward_relu_outputs = []
        self.hook_types = ["register_backward_hook", "register_forward_hook"]
        self.hook_funcs = ["relu_backward_hook_creater", "relu_forward_hook_creater"]
        self.exceptions = []
        if exceptions:
            self.exceptions = exceptions
        super().__init__(model, **kwargs)

    def relu_backward_hook_creater(self, name):
        #print("potentiall adding to", name)
        def _relu_backward_hook(module, grad_in, grad_out):
            if not isinstance(module, (nn.ReLU, nn.SiLU)) or name in self.exceptions:  # only consider ReLUs
                return
            #print("guided_backprop on", name)
            #print("R^{l+1} is", grad_in[0], abs(grad_in[0]).max())
            corresponding_forward_output = self.forward_relu_outputs[-1]
            #print("activations were",  corresponding_forward_output)
            # technicall this still works since it is the output of a ReLU
            #corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = torch.clamp(grad_in[0], min=0.0)
            modified_grad_out[corresponding_forward_output < 0] = 0
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)
        return _relu_backward_hook

    def relu_forward_hook_creater(self, name):
        def _relu_forward_hook(module, inpt, outpt):
            if not isinstance(module, (nn.ReLU, nn.SiLU)) or name in self.exceptions:
                return
            self.forward_relu_outputs.append(outpt)
        return _relu_forward_hook

def set_initializers(mod, gain):
    has_sub_mods = False
    for name, sub_mod in mod.named_children():
        set_initializers(sub_mod, gain)
        has_sub_mods = True
    if not has_sub_mods: # if no submodules, it is an actual operation
        if hasattr(mod, "weight"):
            try:
                torch.nn.init.xavier_normal_(mod.weight, gain=gain)
            except ValueError:
                pass
                #print(mod)
