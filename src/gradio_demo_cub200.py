import argparse
import gc

import gradio as gr
import torch
from PIL import Image
from gradio_imageslider import ImageSlider

from dreamcreature.pipeline import create_args, load_pipeline

MAPPING = {
    'body': 0,
    'tail': 1,
    'head': 2,
    'wing': 4,
    'leg': 6
}

ID2NAME = open('data/cub200_2011/class_names.txt').readlines()
ID2NAME = [line.strip() for line in ID2NAME]

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='sd15-cub200-sup')
opt = parser.parse_args()

OUTPUT_DIR = opt.output_dir


def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        pad_w = 0
        pad_h = (w - h) // 2
        new_image.paste(image, (0, pad_h))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        pad_w = (h - w) // 2
        pad_h = 0
        new_image.paste(image, (pad_w, 0))
        return new_image


def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, seed):
    args = create_args(OUTPUT_DIR)
    if 'dpo' in OUTPUT_DIR:
        args.unet_path = "mhdang/dpo-sd1.5-text2image-v1"

    pipe = load_pipeline(args, torch.float16, 'cuda')
    pipe = pipe.to(torch.float16)

    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    for k, v in MAPPING.items():
        if f'<{k}:' in prompt:
            prompt = prompt.replace(f'<{k}:', f'<{v}:')

    pipe.v1 = False
    image = pipe(prompt,
                 negative_prompt=negative_prompt, generator=generator,
                 num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale).images[0]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return image


with gr.Blocks(title="DreamCreature") as demo:
    with gr.Column():
        with gr.Row():
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", value="a photo of a <body:1>")
                negative_prompt = gr.Textbox(label="Negative Prompt",
                                             value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
                num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Num Inference Steps")
                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
                seed = gr.Number(label="Seed", value=42)
                button = gr.Button()

            output_images = ImageSlider(show_label=False)

    button.click(fn=generate_images,
                 inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale,
                         seed], outputs=[output_images], show_progress=True)

demo.queue().launch(inline=False, share=True, debug=True)
