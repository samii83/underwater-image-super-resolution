"""Minimal Gradio web demo for Underwater Image Super-Resolution."""

import os
import time
from PIL import Image
import gradio as gr
import torch

from underwater_sr.model import SRCNN
from underwater_sr.utils import get_device, pil_to_tensor, tensor_to_pil

# Cache models in memory
_MODELS = {}


def load_cached_model(scale_factor: int, checkpoint_path: str = None, device: str = "auto") -> tuple:
    dev = get_device(device)
    key = (scale_factor, checkpoint_path, str(dev))

    if key not in _MODELS:
        model = SRCNN(scale_factor=scale_factor).to(dev)
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=dev))
            msg = f"Loaded checkpoint: {checkpoint_path}"
        else:
            msg = "Using initial weights (untrained model for demonstration)"
        model.eval()
        _MODELS[key] = (model, dev, msg)

    return _MODELS[key]


def predict(input_img: Image.Image, scale_factor: int, checkpoint_path: str):
    if input_img is None:
        return None, "Please upload an image."

    start_t = time.time()
    ckpt = checkpoint_path.strip() if checkpoint_path else None
    model, device, ckpt_msg = load_cached_model(scale_factor, ckpt)

    orig_w, orig_h = input_img.size
    lr_tensor = pil_to_tensor(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_tensor = model(lr_tensor)

    sr_img = tensor_to_pil(sr_tensor.squeeze(0))
    elapsed = (time.time() - start_t) * 1000.0

    status = (
        f"Status: Success\n"
        f"Input Size: {orig_w}x{orig_h}\n"
        f"Output Size: {sr_img.width}x{sr_img.height} ({scale_factor}x)\n"
        f"Device: {device}\n"
        f"Latency: {elapsed:.1f} ms\n"
        f"Model: {ckpt_msg}"
    )
    return sr_img, status


def create_ui():
    with gr.Blocks(title="Underwater Image Super-Resolution") as demo:
        gr.Markdown(
            "# 🌊 Underwater Image Super-Resolution (SRCNN)\n"
            "Enhance and super-resolve low-resolution underwater imagery using a deep convolutional neural network."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="pil",
                    label="Input Underwater Image",
                    value="sample_images/underwater_sample.png" if os.path.exists("sample_images/underwater_sample.png") else None,
                )
                scale_slider = gr.Radio([2, 4], value=4, label="Upscaling Factor")
                ckpt_input = gr.Textbox(
                    label="Checkpoint Path (Optional)",
                    placeholder="e.g. checkpoints/best_srcnn_x4.pth",
                    value="",
                )
                btn = gr.Button("Super-Resolve Image", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Super-Resolved Output")
                status_box = gr.Textbox(label="Inference Details", lines=6, interactive=False)

        btn.click(
            fn=predict,
            inputs=[input_image, scale_slider, ckpt_input],
            outputs=[output_image, status_box],
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch()

