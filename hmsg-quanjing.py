import torch
from torch.nn import Conv2d
import torch.nn.functional as F
import weakref

class AspectAwareTilingNode_Final:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "enable": (["enable", "disable"], {"default": "enable"}),
                "target_ratio": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1}),
            },
        }
    
    CATEGORY = "image/postprocessing"
    RETURN_TYPES = ("MODEL", "VAE")
    FUNCTION = "process"

    def process(self, model, vae, enable, target_ratio):
        # 使用弱引用避免循环依赖
        model_ref = weakref.proxy(model)
        vae_ref = weakref.proxy(vae)
        
        safe_model = self.SafeModelWrapper(model_ref, target_ratio)
        safe_vae = self.SafeVAEWrapper(vae_ref, target_ratio)
        
        if enable == "enable":
            safe_model.enable_tiling()
            safe_vae.enable_tiling()
        else:
            safe_model.disable_tiling()
            safe_vae.disable_tiling()
            
        return (safe_model, safe_vae)

    class SafeModelWrapper(torch.nn.Module):
        def __init__(self, orig_model, ratio):
            super().__init__()
            self._orig_model = orig_model
            self._ratio = ratio
            self._patches = orig_model.patches.copy() if hasattr(orig_model, 'patches') else []
            self._modified_layers = {}

        def forward(self, x, *args, **kwargs):
            # 保持原始前向传播
            return self._orig_model(x, *args, **kwargs)

        def enable_tiling(self):
            def _mod_layer(layer):
                if isinstance(layer, Conv2d) and id(layer) not in self._modified_layers:
                    # 保存原始参数
                    self._modified_layers[id(layer)] = {
                        'padding': layer.padding,
                        'padding_mode': layer.padding_mode,
                        'forward': layer._conv_forward
                    }
                    # 应用新参数
                    h_pad = layer.padding[0]
                    w_pad = int(h_pad * self._ratio)
                    layer.padding = (h_pad, w_pad)
                    layer.padding_mode = 'circular'
                    layer._conv_forward = self.create_custom_forward(layer, h_pad, w_pad)

            self._orig_model.model.apply(_mod_layer)

        def disable_tiling(self):
            for layer_id, params in self._modified_layers.items():
                layer = next((l for l in self._orig_model.model.modules() if id(l) == layer_id), None)
                if layer:
                    layer.padding = params['padding']
                    layer.padding_mode = params['padding_mode']
                    layer._conv_forward = params['forward']
            self._modified_layers.clear()

        @staticmethod
        def create_custom_forward(layer, h_pad, w_pad):
            def _custom_forward(_, input, weight, bias):
                padded = F.pad(input, (w_pad, w_pad, h_pad, h_pad), mode='circular')
                return F.conv2d(padded, weight, bias, 
                              layer.stride, (0, 0),
                              layer.dilation, layer.groups)
            return _custom_forward.__get__(layer, Conv2d)

        def __getattr__(self, name):
            """安全属性访问"""
            if name in self.__dict__:
                return self.__dict__[name]
            try:
                return getattr(self._orig_model, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    class SafeVAEWrapper:
        def __init__(self, orig_vae, ratio):
            self._orig_vae = orig_vae
            self._ratio = ratio
            self._modified = False

        def encode(self, x):
            return self._orig_vae.encode(x)

        def decode(self, z):
            return self._orig_vae.decode(z)

        def enable_tiling(self):
            if not self._modified:
                self._apply_vae_tiling(enable=True)
                self._modified = True

        def disable_tiling(self):
            if self._modified:
                self._apply_vae_tiling(enable=False)
                self._modified = False

        def _apply_vae_tiling(self, enable):
            def _mod_layer(layer):
                if isinstance(layer, Conv2d):
                    if enable:
                        layer.padding_mode = 'circular'
                        layer.padding = (layer.padding[0], int(layer.padding[1] * self._ratio))
                    else:
                        layer.padding_mode = 'zeros'
                        layer.padding = (layer.padding[0], int(layer.padding[1] / self._ratio))

            self._orig_vae.first_stage_model.apply(_mod_layer)

        def __getattr__(self, name):
            """安全代理到原始VAE"""
            try:
                return getattr(self._orig_vae, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

NODE_CLASS_MAPPINGS = {"AspectAwareTiling": AspectAwareTilingNode_Final}
NODE_DISPLAY_NAME_MAPPINGS = {"AspectAwareTiling": "📏 Ratio-Aware Tiling"}