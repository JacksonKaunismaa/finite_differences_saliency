import torch
import torch.nn as nn

class HookAdder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.handles = []
        self.hooks_exist = False
        
    def setup_hooks(self):
        if self.hooks_exist:
            return
        def recurse_modules(mod, curr_name=""):
            has_sub_mods = False
            for name, sub_mod in mod.named_children():
                recurse_modules(sub_mod, f"{curr_name}{name}.")
                has_sub_mods = True
            if not has_sub_mods: # if no submodules, it is an actual operation
                for hook_type, hook_func in zip(self.hook_types, self.hook_funcs):
                    mod_hook = getattr(mod, hook_type)
                    generated_hook = getattr(self, hook_func)(curr_name[:-1])
                    self.handles.append(mod_hook(generated_hook))
        recurse_modules(self.model)
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
    def __init__(self, model):
        self.hook_types = ["register_forward_hook"]
        self.hook_funcs = ["benchmark_hook"]
        super().__init__(model)
    
    def benchmark_hook(self, name):
        def fn(layer, inpt, outpt):
            benchmark(name, verbose=False)
        return fn

class AllActivations(HookAdder):
    def __init__(self, model):
        self._features = {}
        self.hook_types = ["register_forward_hook"]
        self.hook_funcs = ["save_activations_hook"]
        super().__init__(model)
            
    def save_activations_hook(self, name):
        def fn(layer, inpt, output):
            self._features[name] = output
        return fn


# adapted from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py
class GuidedBackprop(HookAdder):
    def __init__(self, model, exceptions=None):
        self.forward_relu_outputs = []
        self.hook_types = ["register_backward_hook", "register_forward_hook"]
        self.hook_funcs = ["relu_backward_hook_creater", "relu_forward_hook_creater"]
        self.exceptions = []
        if exceptions:
            self.exceptions = exceptions
        super().__init__(model)
    
    def relu_backward_hook_creater(self, name):
        #print("potentiall adding to", name)
        def _relu_backward_hook(module, grad_in, grad_out):
            if not isinstance(module, nn.ReLU) or name in self.exceptions:  # only consider ReLUs
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
            if not isinstance(module, nn.ReLU) or name in self.exceptions:
                return
            self.forward_relu_outputs.append(outpt)
        return _relu_forward_hook
