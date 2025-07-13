print("[custom_hooks.py] ✅ custom_hooks.py loaded")
from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            runner.eval_model.load_state_dict(runner.model.state_dict())

'''
@HOOKS.register_module()
class FreezeModulesHook(Hook):
    def __init__(self, frozen_modules):
        self.frozen_modules = frozen_modules

    def get_nested_attr(self, obj, attr_path):
        for attr in attr_path.split('.'):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
            return obj

    def before_train_epoch(self, runner):
        print("[FreezeModulesHook] ✅ before_train_epoch triggered")
        model = runner.model.module if getattr(runner.model, 'module', None) else runner.model
        for module_name in self.frozen_modules:
            module = self.get_nested_attr(model, module_name)
            if module is None:
                print(f"[FreezeModulesHook] Warning: Module '{module_name}' not found.")
                continue
            for param in module.parameters():
                param.requires_grad = False
            print(f"[FreezeModulesHook] ✅ Module '{module_name}' frozen.")
        for name, param in model.named_parameters():
            print(f"[Contains] {name}")
            if param.requires_grad:
                print(f"[Trainable ✅] {name}")
                
'''

@HOOKS.register_module()
class FreezeAllButSegHead(Hook):
    def __init__(self, frozen_modules):
        self.frozen_modules = frozen_modules

    def before_train_epoch(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        for name, param in model.named_parameters():
            if 'seg_decoder' in name:  # use broader match
                param.requires_grad = True
                print(f"[FreezeAllButSegHead] 🔓 Kept trainable: {name}")
            else:
                param.requires_grad = False
                print(f"[FreezeAllButSegHead] ✅ Frozen: {name}")