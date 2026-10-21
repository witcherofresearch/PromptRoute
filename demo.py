
from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes



import numpy as np
import random
import torch
import spaces

from PIL import Image
#from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwen_vl_utils import process_vision_info



import os

from huggingface_hub import  hf_hub_download

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

seafoam = Seafoam()
CSS = """
.fillable { max-width: 960px !important; }
"""


def update_textbox(selected_items):
    # Join the selected list of strings into a comma-separated string
    return ", ".join(selected_items)


pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipe.to('cuda')
pipe.set_progress_bar_config(disable=None)


'''
pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',
        origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
        download_source='huggingface',origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=None,
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
    download_source='huggingface',origin_file_pattern="processor/"),
)
'''





qwenstyle= hf_hub_download(repo_id="witcherderivia/PromptRoute", filename="diffusers_promptroute_qie2511_bf16.safetensors")
speedup = hf_hub_download(repo_id="Tele-AI/TeleStyleV2", filename="QIE-2511-Lightning-4steps-V1.0-bf16.safetensors")



pipe.load_lora_weights(
    qwenstyle,adapter_name='style'
)


pipe.load_lora_weights(
    speedup,adapter_name='dmd'
)

pipe.set_adapters(["style", "dmd",], adapter_weights=[1.0, 1.0])
pipe.fuse_lora(adapter_names=["style", "dmd"], lora_scale=1.0)
pipe.unload_lora_weights()






dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"




MAX_SEED = np.iinfo(np.int32).max


#@spaces.GPU(size="xlarge")


def infer(
    content_ref,
    style_ref,
    prompt,
    seed=123,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    minedge=1024,
    progress=gr.Progress(track_tqdm=True),
    checkbox=[],
    
):
    
    

    

    
    content_text_input='describe main objects (fewer than 3) with separated words, each word is separated by comma,  the total number of words is strictly fewer than 3'
    style_text_input='describe only the artistic style, material, stroke in 5 words, not objects.'
    #pipe.text_encoder.eval()
    content_prompt=''
    style_prompt=''

    
    

    

    if content_ref is not None:
        content_ref=Image.fromarray(content_ref)
        content_messages = [
        {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": content_ref,
                    },
                    {"type": "text", "text": content_text_input},
                ],
            }
        ]
        content_text = pipe.processor.apply_chat_template(
            content_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(content_messages)
        inputs = pipe.processor(
            text=[content_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        generated_ids = pipe.text_encoder.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        content_prompt = pipe.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"content_prompt={content_prompt}")
    if style_ref is not None:
        style_ref=Image.fromarray(style_ref)
        style_messages = [
        {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": style_ref,
                    },
                    {"type": "text", "text": style_text_input},
                ],
            }
        ]
        style_text = pipe.processor.apply_chat_template(
            style_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(style_messages)
        inputs = pipe.processor(
            text=[style_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        generated_ids = pipe.text_encoder.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        style_prompt = pipe.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"style_prompt={style_prompt}")
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    
    
    
    
    sw,sh,w,h=0,0,0,0
    if content_ref:
        w,h=content_ref.size



        #minedge=1024
        if w>h:
            r=w/h
            h=minedge
            w=int(h*r)-int(h*r)%16
            
        else:
            r=h/w
            w=minedge
            h=int(w*r)-int(w*r)%16
    if style_ref:
        sw,sh=style_ref.size
        if sw>sh:
            r=sw/sh
            sh=minedge
            sw=int(sh*r)-int(sh*r)%16
            
        else:
            r=sh/sw
            sw=minedge
            sh=int(sw*r)-int(sw*r)%16

    if "Compose the pictures" in prompt:
        
        #minedge=1024
        if content_ref:
            w,h=content_ref.size
            r=w/h
            h=minedge
            w=int(h*r)-int(h*r)%16
        #content_ref=content_ref.resize((w,h))
        #minedge=1024
        if style_ref:
            sw,sh=style_ref.size
            r=sw/sh
            sh=minedge
            sw=int(sh*r)-int(sh*r)%16

    
    
    
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale},")
    
    if content_ref and style_ref:
        images = [
            content_ref.resize((w, h)),
            style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    elif content_ref:
        images = [
            content_ref.resize((w, h)),
            #style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    elif style_ref:
        images = [
            #content_ref.resize((w, h)),
            style_ref.resize((sw, sh)) ,
            #style_ref.resize((minedge, minedge)) ,
        ]
    
    if "infer with content prompt" in checkbox and content_prompt not in prompt:
        prompt=','.join([prompt,content_prompt])
    if "infer with style prompt" in checkbox and style_prompt not in prompt:
        prompt=','.join([prompt,style_prompt])
    if "infer with content prompt" not in checkbox and content_prompt in prompt:
        prompt=prompt.replace(content_prompt.strip(','),'')
    if "infer with style prompt" not in checkbox and style_prompt in prompt:
        prompt=prompt.replace(style_prompt.strip(),'')
    prompt=prompt.strip(',')
    print(f"Calling pipeline with prompt: '{prompt}'")
    if "Compose the pictures" in prompt:
            w=int(16/9*minedge)
            w=w-w%16
            h=minedge
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_guidance_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": true_guidance_scale,
        "num_images_per_prompt": 1,
        "width": w or sw,
        "height": h or sh, 
    }
    with torch.inference_mode():
        image = pipe(**inputs)
    image = image.images[0]
    
    

    
    




    return image, seed, content_prompt, style_prompt, prompt

# --- Examples and UI Layout ---
examples = []



_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">PromptRoute</h1>
    
</div>


<p style="font-size: 1rem; margin-bottom: 1.5rem;">Paper: <a href='' target='_blank'>PromptRouter: Routing Different Image Customization Tasks via Simple Prompt</a> | Codes: <a href='https://github.com/witcherofresearch/PromptRoute' target='_blank'>GitHub</a>
</p>
<p style="font-size: 1rem; margin-bottom: 1.5rem;">Please light a star for the <a href='https://github.com/witcherofresearch/PromptRoute' target='_blank'>Code</a> if you find the demo useful, thank you.</p>

<p style="font-size: 1rem; margin-bottom: 1.5rem;">In this demo, we question whether there exist any conflicts when learning multi-ref subject-driven image generation and content-preserving style transfer given content ref and style ref with a unified model. The conclusion is no and our PromptRoute's solution is simple: we attach "Compose the pictures" for multi-ref subject generation, and "Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1." for content-preserving style transfer during training and inference. Such simple prompts distinguish the tasks and eliminate the confusion. </p> 

<p style="font-size: 1rem; margin-bottom: 1.5rem;">Caution: If you choose infer with content/style prompt, pls do not forget to clean the prompt box when you run a new  inference. The results of multi-ref subject generation are all 16:9 in this demo. </p>

<p style="font-size: 1rem; margin-bottom: 1.5rem;">If you encounter an Error with this demo, the most possible reason is ZeroGPU out-of-memory and the solution is to decrease the Min Edge of the generated image from 1024 to a lower value.  </p>
'''  

with gr.Blocks() as demo:

    with gr.Column(elem_id="col-container"):
        
        gr.Markdown(_HEADER_)
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    content_ref = gr.Image(label="content ref", type="numpy", )
                    style_ref = gr.Image(label="style ref", type="numpy", )
                    #print(f"type(content_ref)={type(content_ref)}")
                    
                #input_images = gr.Gallery(label="Input Images", show_label=False, type="pil", interactive=True)
            

            result = gr.Image(label="Result", show_label=True, type="pil")
            #result = gr.Gallery(label="Result", show_label=True, type="pil")
        with gr.Column():
            
            checkbox=gr.CheckboxGroup(["infer with content prompt", "infer with style prompt"], label="Prompt Enhancer", )
            content_prompt=gr.Text(
                    label="Content Reference Prompt",
                    show_label=True,
                    container=True,
            )
            style_prompt=gr.Text(
                    label="Style Reference Prompt",
                    show_label=True,
                    container=True,
            )
            prompt = gr.Text(
                    label="Prompt",
                    value='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    show_label=True,
                    placeholder='Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.',
                    container=True,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=True):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=123,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=False)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="CFG should be 1.0",
                    minimum=0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps should be 4",
                    minimum=1,
                    maximum=50,
                    step=1,
                    value=4,
                )
                
                minedge = gr.Slider(
                    label="Min Edge of the generated image",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1024,
                )
        
        with gr.Row(), gr.Column():
            gr.Markdown("## Examples")
            gr.Markdown("Some of these references are example refs from Kling Image 3.0 Doc. For those cases, our model is obviouly superior than Kling Image 3.0.")
            default_prompt=''
            gr.Examples(examples=[
                ['./qwenstyleref/man.png','./qwenstyleref/woman.png','Compose the pictures. a man and a woman  are fighting with light sabres. They are facing the camera.',123,False,1.0,4,1024,[]], 
                ['./qwenstyleref/kling_ref1.webp','./qwenstyleref/kling_ref0.webp','Compose the pictures. a girl and a woman  are fighting with swords, they are looking at the camera',123,False,1.0,4,1024,[]],               
                ['./qwenstyleref/kling_ref0.webp','./qwenstyleref/kling_ref1.webp','Compose the pictures. a woman and a girl  are fighting with swords, they are looking at the camera',123,False,1.0,4,832,[]],               
                ['./qwenstyleref/kling_ref0.webp','./qwenstyleref/kling_ref1.webp','Style Transfer',123,False,1.0,4,864,[]],
                ['./qwenstyleref/kling_ref0.webp','./qwenstyleref/kling_ref1.webp','',123,False,1.0,4,832,[]],
                ],
                inputs=[content_ref,
                    style_ref,
                    prompt,
                    seed,
                    randomize_seed,
                    true_guidance_scale,
                    num_inference_steps,
                    minedge,
                    checkbox
                    ], 
                outputs=[result, seed, content_prompt, style_prompt,prompt], 
                fn=infer, 
                cache_examples=False
                )      
                
                
                
                

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)
        

    gr.on(
        triggers=[run_button.click],
        fn=infer,
        inputs=[
            content_ref,
            style_ref,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            minedge,
            checkbox,
            
        ],
        outputs=[result, seed, content_prompt, style_prompt,prompt],
    )

    
    

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0',theme=seafoam,
                css=CSS,)
'''
['./qwenstyleref/pulpfiction_2.jpg','./qwenstyleref/styleref=6_style_ref.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/110.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/romanholiday_1.jpg','./qwenstyleref/s0099____1113_01_query_1_img_000146_1682705733350_08158389675901344.jpg.jpg',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/125.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/fallenangle.jpg','./qwenstyleref/styleref=s0038.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/styleref=s0572.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/startrooper1.jpg','./qwenstyleref/david-face-760x985.jpg','Style Transfer Figure  1 into marble material.',123,False,1.0,4,1024,[]],
                ['./qwenstyleref/startrooper1.jpg','./qwenstyleref/125.png',default_prompt, 123,False,1.0,4,1024,[]],
                ['./qwenstyleref/possession.png','./qwenstyleref/s0026____0907_01_query_0_img_000194_1682674358294_041656249089406583.jpeg.jpg',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/styleref=0_content_ref.png','./qwenstyleref/Jotarokujo.webp',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/wallstreet1.jpg','./qwenstyleref/034.png',default_prompt,123,False,1.0,4,1024,[]],
                ['./qwenstyleref/bird.jpeg','./qwenstyleref/styleref=s0539.png',default_prompt,123,False,1.0,4,1024,[]],
'''
