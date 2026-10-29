"""Small trainable part of a pinned ControlNet, with strict partial weights."""

from pathlib import Path
import torch

PREFIXES = ("controlnet_cond_embedding.", "controlnet_down_blocks.", "controlnet_mid_block.")


def configure_trainable(controlnet, unet):
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    modules = [controlnet.controlnet_cond_embedding, *controlnet.controlnet_down_blocks,
               controlnet.controlnet_mid_block]
    for module in modules:
        module.float().requires_grad_(True)
    # Diffusers checkpointing runs in train mode; the pinned networks have no active dropout.
    for network in (controlnet, unet):
        if any(isinstance(m, torch.nn.Dropout) and m.p for m in network.modules()):
            raise ValueError("Frozen-backbone checkpointing expects dropout=0")
        network.train()
        network.enable_gradient_checkpointing()
    return [p for p in controlnet.parameters() if p.requires_grad]


def partial_state(controlnet):
    return {name: value.detach().cpu().float().contiguous()
            for name, value in controlnet.state_dict().items() if name.startswith(PREFIXES)}


def apply_state(controlnet, state):
    expected = partial_state(controlnet)
    if set(state) != set(expected):
        raise ValueError("Incomplete/unexpected ControlNet adapter keys")
    for name, value in state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid adapter tensor: {name}")
    with torch.no_grad():
        target = controlnet.state_dict()
        for name, value in state.items():
            target[name].copy_(value.to(target[name]))


def save_adapter(controlnet, path):
    from safetensors.torch import save_file
    state = partial_state(controlnet)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Refuse to save nonfinite ControlNet adapter weights")
    save_file(state, str(path))


def load_adapter(controlnet, path):
    from safetensors.torch import load_file
    apply_state(controlnet, load_file(str(path), device="cpu"))


def noise_prediction(controlnet, unet, noisy, timesteps, text, condition):
    down, mid = controlnet(noisy, timesteps, encoder_hidden_states=text,
                           controlnet_cond=condition, return_dict=False)
    # Keep autograd through the frozen UNet: it carries the adapter's gradient.
    return unet(noisy, timesteps, encoder_hidden_states=text,
                down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
