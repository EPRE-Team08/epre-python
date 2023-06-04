import torch
from model import BrandsDetector
from torch.utils.mobile_optimizer import optimize_for_mobile

CKPT = r"logs\lightning_logs\version_0\checkpoints\epoch=199-step=200.ckpt"


model = BrandsDetector.load_from_checkpoint(CKPT)
model.eval()
model.to("cpu")

example = torch.rand(1, 3, 224, 224).to("cpu")
traced_script_module = torch.jit.trace(model.model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("model_cpu.ptl")

