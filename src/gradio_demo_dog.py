import gc
import os
import shutil

import gradio as gr
import requests
import torch
import traceback
import argparse

from dreamcreature.pipeline import create_args, load_pipeline


def download_file(url, local_path):
    if os.path.exists(local_path):
        return

    with requests.get(url, stream=True) as r:
        with open(local_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # Example usage


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='dreamcreature-sd1.5-dog')
parser.add_argument('--checkpoint', default='checkpoint-150000')
opt = parser.parse_args()

model_name = opt.model_name
checkpoint_name = opt.checkpoint

repo_url = f"https://huggingface.co/kamwoh/{model_name}/resolve/main"
file_url = repo_url + f"/{checkpoint_name}/pytorch_model.bin"
local_path = f"{model_name}/{checkpoint_name}/pytorch_model.bin"
os.makedirs(f"{model_name}/{checkpoint_name}", exist_ok=True)
download_file(file_url, local_path)

file_url = repo_url + f"/{checkpoint_name}/pytorch_model_1.bin"
local_path = f"{model_name}/{checkpoint_name}/pytorch_model_1.bin"
download_file(file_url, local_path)

OUTPUT_DIR = model_name

args = create_args(OUTPUT_DIR)
if 'dpo' in OUTPUT_DIR:
    args.unet_path = "mhdang/dpo-sd1.5-text2image-v1"

pipe = load_pipeline(args, torch.float16, 'cuda')
pipe = pipe.to(torch.float16)

pipe.verbose = True
pipe.v1 = False
pipe.num_k_per_part = 120

MAPPING = {
    'eye': 0,
    'neck': 2,
    'ear': 3,
    'body': 4,
    'leg': 5,
    'nose': 6,
    'forehead': 7
}

ID2NAME = open('data/dogs/class_names.txt').readlines()
ID2NAME = [line.strip() for line in ID2NAME]


def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, seed):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(int(seed))

    part2id = []

    tokens = prompt.split(' ')
    for t in tokens:
        for k, v in MAPPING.items():
            if t.startswith(f'<{k}:') and (t.endswith('>') or t.endswith('>,') or t.endswith('>.')):
                i, b = t.split(':')
                bnum = ''.join([char for char in b if char.isdigit()])
                clsid = abs(int(bnum))
                part2id.append(f'{k}: {ID2NAME[clsid]}')
                break

    for k, v in MAPPING.items():
        if f'<{k}:' in prompt:
            prompt = prompt.replace(f'<{k}:', f'<{v}:')
        if f'<{k}:' in negative_prompt:
            negative_prompt = negative_prompt.replace(f'<{k}:', f'<{v}:')

    try:
        images = pipe(prompt,
                      negative_prompt=negative_prompt, generator=generator,
                      num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                      num_images_per_prompt=num_images).images
    except Exception as e:
        raise gr.Error(f"Probably due to the prompt have invalid input, please follow the instruction. "
                       f"The error message: {e}")
    finally:
        gc.collect()
        torch.cuda.empty_cache()

    return images, '; '.join(part2id)


with gr.Blocks(title="DreamCreature") as demo:
    with gr.Row():
        gr.Markdown(
            """
            # DreamCreature (Stanford Dogs)
            To create your own creature, you can type:
            
            `"a photo of a <nose:id> <ear:id> dog"` where `id` ranges from 0~119 (120 classes corresponding to Stanford Dogs)
            
            For instance `"a photo of a <nose:2> <ear:112> dog"` using head of `maltese dog (2)` and wing of `cardigan (112)`
            
            Please see `id` in https://github.com/kamwoh/dreamcreature/blob/master/src/data/dogs/class_names.txt
            
            (Experimental) You can also use two parts together such as:
            
            `"a photo of a <nose:1> <nose:112> dog"` mixing head of `maltese dog (2)` and `spotted cardigan (112)`
            
            The current available parts are: `eye`, `neck`, `ear`, `body`, `leg`, `nose` and `forehead`
            
            """)
    with gr.Column():
        with gr.Row():
            with gr.Group():
                prompt = gr.Textbox(label="Prompt", value="a photo of a <nose:2> <ear:112> dog")
                negative_prompt = gr.Textbox(label="Negative Prompt",
                                             value="blurry, ugly, duplicate, poorly drawn, deformed, mosaic")
                num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Num Inference Steps")
                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
                num_images = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Number of Images")
                seed = gr.Number(label="Seed", value=42)
                button = gr.Button()

            with gr.Column():
                output_images = gr.Gallery(columns=4, label='Output')
                markdown_labels = gr.Markdown("")

    button.click(fn=generate_images,
                 inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images,
                         seed], outputs=[output_images, markdown_labels], show_progress=True)

demo.queue().launch(inline=False, share=False, debug=True, server_name='0.0.0.0')
