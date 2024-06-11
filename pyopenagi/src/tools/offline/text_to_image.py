from diffusers import AutoPipelineForText2Image
import torch

from ..base import BaseHuggingfaceTool

class SDXLTurbo(BaseHuggingfaceTool):
    def __init__(self):
        super().__init__()
        self.pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

    def run(self, params):
        prompt = params["prompt"]
        self.pipe.to("cuda")
        image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return image
