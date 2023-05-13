"""
Copyright 2023 Yingqiang Ge

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import os

from langchain.agents import initialize_agent, Tool, load_tools
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.tools import BaseTool

os.environ["SERPAPI_API_KEY"] = "YOUR SEARCH API KEY"
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
os.environ['TRANSFORMERS_CACHE'] = "YOUR CACHE DIRECTORY TO STORE HUGGINGFACE TRANSFORMERS"

# import github_models.colorization.colorizers as colorizers
# from github_models.colorization.colorizers import *

from torchvision import transforms
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DetrImageProcessor, 
    DetrForObjectDetection,
    ViTFeatureExtractor, 
    ViTForImageClassification,
    AutoImageProcessor, 
    Swin2SRForImageSuperResolution,
    set_seed,
    ViltProcessor, 
    ViltForQuestionAnswering,
    VisionEncoderDecoderModel,
    pipeline, 
    BlipProcessor, 
    BlipForConditionalGeneration, 
    BlipForQuestionAnswering,
    DetrImageProcessor, 
    DetrForObjectDetection,
)

from diffusers import StableDiffusionPipeline

import models.github_models.colorization.colorizers as colorizers
from models.github_models.colorization.colorizers import *

import torch
import os
from runpy import run_path
from skimage import img_as_ubyte
import uuid

import cv2

from PIL import Image, ImageDraw, ImageOps, ImageFont

import torchvision.transforms as transforms
from torchvision.utils import save_image

import warnings
warnings.filterwarnings('ignore')

import gc

try:
    import langchain as lc

    LANGCHAIN_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    LANGCHAIN_INSTALLED = False

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)


class BaseModel:
    def __init__(self, device, model_name):
        pass
    
    def run(self, input):
        pass
    
    # Optional langchain functionalities
    @property
    def langchain(self) -> "langchain.agents.Tool":  # type: ignore
        if not LANGCHAIN_INSTALLED:
            raise ModuleNotFoundError(
                "langchain must be installed to access langchain tool"
            )
        return lc.agents.Tool(  # type: ignore
            name=self.name, func=self.run, description=self.description
        )

    
def save_string_to_file(input_string):
    file_path = "./travel_plan.txt"
    with open(file_path, 'a') as file:
        file.write(input_string+"\n")
        return file_path
    
WriteFileTool = Tool(
    name = "WriteFile",
    func=save_string_to_file,
    description="""useful for when you need to write a string to a file. 
                The input to this tool should be a string, representing the file to write the string.
                For example, "Hello, world!" would be the input."""
    )

    
# class WriteFile(BaseModel):
#     name = "WriteFile",
#     description="""useful for when you need to write a string to a file. 
#                 The input to this tool should be a string, representing the file needed to be saved.
#                 For example, "Hello, world!" would be the input."""
    
#     def __init__(self, device, model_name=None):
#         print(f"Initializing Text2Image to {device}")
#         self.device = device
#         self.model_name = model_name
    
#     def run(self, input):
#         file_path = "./travel_plan.txt"
#         with open(file_path, 'a') as file:
#             file.write(input)
#             return file_path
    
#     # Optional langchain functionalities
#     @property
#     def langchain(self) -> "langchain.agents.Tool":  # type: ignore
#         if not LANGCHAIN_INSTALLED:
#             raise ModuleNotFoundError(
#                 "langchain must be installed to access langchain tool"
#             )
#         return lc.agents.Tool(  # type: ignore
#             name=self.name, func=self.run, description=self.description
#         )
    
    
class Text2Image(BaseModel):
    name="Generate Image From User Input Text"
    description="useful when you want to generate an image from a user input text and save it to a file. " \
                "like: generate an image of an object or something, or generate an image that includes some objects. " \
                "The input to this tool should be a string, representing the text used to generate image. "
    def __init__(self, device, model_name="runwayml/stable-diffusion-v1-5"):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_name, torch_dtype=self.torch_dtype)
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, ' \
                        'fewer digits, cropped, worst quality, low quality'

        self.name="Generate Image From User Input Text"
        self.description="useful when you want to generate an image from a user input text and save it to a file. " \
                         "like: generate an image of an object or something, or generate an image that includes some objects. " \
                         "The input to this tool should be a string, representing the text used to generate image. "
    
    def run(self, text):
        self.pipe.to(self.device)
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        prompt = text + ', ' + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        self.pipe.to('cpu')
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename

    
class ImageCaptioning(BaseModel):
    name="ImageCaptioning"
    description="useful when you want to know what is inside the photo. It receives image_path as input. "\
             "The input to this tool should be a string, representing the image_path. "
    # description="An image captioner. Use this to create a caption for an image. " \
    #         "Input will be a path to an image file. " \
    #         "The output will be a caption of that image."
            
    def __init__(self, device, model_name="Salesforce/blip-image-captioning-base"):
        print(f"Initializing ImageCaptioning to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=self.torch_dtype)

    def run(self, image_path):
        self.model.to(self.device)
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        self.model.to('cpu')
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions

    
class ImageClassification(BaseModel):
    name="ImageClassification"
    description="useful when you want to know the class of the image. It receives image_path as input. "\
             "The input to this tool should be a string, representing the image_path. "
            
    def __init__(self, device, model_name='google/vit-base-patch16-224'):
        print(f"Initializing ImageClassification to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path, torch_type=self.torch_dtype)
        self.classifier = ViTForImageClassification.from_pretrained(self.model_path).to(self.device)

    def run(self, image_path):
        self.classifier.to(self.device)
        inputs = self.feature_extractor(Image.open(image_path), return_tensors="pt").to(self.device, self.torch_dtype)
        logits = self.classifier(**inputs).logits
        predicted_class_idx = logits.argmax(1)[0].item()
        predicted_class = self.classifier.config.id2label[predicted_class_idx]
        self.classifier.to('cpu')
        print(f"\nProcessed ImageClassification, Input Image: {image_path}, Output Text: {predicted_class}")
        return predicted_class

    
class ImageColorization(BaseModel):
    name="ImageColorization"
    description="useful when you want to colorize a photo. It receives image_path as input. "\
             "The input to this tool should be a string, representing the image_path. "
            
    def __init__(self, device, model_name=None):
        print(f"Initializing ImageColorization to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.colorizer= colorizers.siggraph17()
        self.img_transform = transforms.Compose([
            transforms.PILToTensor(),
        ])
    def run(self, image_path):
        self.colorizer.to(self.device)
        img = self.img_transform(Image.open(image_path)).to(self.device).permute(1,2,0).cpu().numpy()
        # print(img.shape)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        tens_l_rs = tens_l_rs.to(self.device)

        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel
        img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
        out_img = postprocess_tens(tens_l_orig, self.colorizer(tens_l_rs).cpu())

        norm_out_img = cv2.normalize(out_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_out_img = norm_out_img.astype(np.uint8)
        
        # print(colored_img.shape)
        colored_img = Image.fromarray(norm_out_img)
        
        colored_img_path = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        colored_img.save(colored_img_path)
        self.colorizer.to('cpu')
        print(
            f"\nProcessed ImageColorization, Input Text: {image_path}, Output Image: {colored_img_path}")
        return colored_img_path
    
    
class ObjectDetection(BaseModel):
    name="ObjectDetection"
    description="useful when you want to detect the objects in a photo. It receives image_path as input. "\
             "The input to this tool should be a string, representing the image_path. "
            
    def __init__(self, device, model_name="facebook/detr-resnet-101"):
        print(f"Initializing Object Detection to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.detector = DetrForObjectDetection.from_pretrained(self.model_name)

    def run(self, image_path):
        self.detector.to(self.device)
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        output = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            output += self.detector.config.id2label[label.item()]
            output += ", "
        predicted_results = output[:-2]
        self.detector.to('cpu')
        
        return predicted_results

# TODO: solve the cuda out of memory problem
class ImageSuperResolution(BaseModel):
    name="ImageSuperResolution"
    description="useful when you want to create a high-resolution image from a low-resolution image. It receives image_path as input. "\
             "The input to this tool should be a string, representing the image_path. "
            
    def __init__(self, device, model_name="caidas/swin2SR-classical-sr-x2-64"):
        print(f"Initializing ImageSuperResolution to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(self.model_name)
        self.img_transform = transforms.Compose([
            transforms.Resize((8,8)),
            transforms.PILToTensor(),
        ])

    def run(self, image_path):
        self.model.to(self.device)
        img = Image.open(image_path)
        inputs = self.processor(img, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=0, destination=-1)
        output = (output * 255.0).round().astype(np.uint8)
        reformed_img = Image.fromarray(output)
        
        reformed_img_path = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        reformed_img.save(reformed_img_path)
        self.model.to('cpu')
        print(
            f"\nProcessed ImageColorization, Input Text: {image_path}, Output Image: {colored_img_path}")
        return reformed_img_path

    
# TODO input requires both image and text
class VisualQuestionAnswering(BaseModel):
    name="VisualQuestionAnswering"
    description=""
            
    def __init__(self, device, model_name="caidas/swin2SR-classical-sr-x2-64"):
        pass
    
    def run(self, image_path):
        pass
    

class SentimentAnalysis(BaseModel):
    name="SentimentAnalysis"
    description="useful when you want to analyze the sentiment of a sentence. It receives sentence as input. "\
             "The input to this tool should be a sentence string. "
            
    def __init__(self, device, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Initializing SentimentAnalysis to {device}")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def run(self, sentence):
        self.model.to(self.device)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)

        # Get the outputs from the model
        outputs = self.model(**inputs)

        # Get the logits from the outputs
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get the most likely label and score
        predicted_label_id = torch.argmax(probabilities, 1)[0]#.item()
        predicted_label = self.model.config.id2label[predicted_label_id.item()]
        self.model.to('cpu')
        print(
            f"\nProcessed SentimentAnalysis, Input Text: {sentence}, Output Text: {predicted_label}")
        return predicted_label
        

# TODO: input requires two text sequences
class QuestionAnswering(BaseModel):
    name="QuestionAnswering"
    description=""
            
    def __init__(self, device, model_name="distilbert-base-cased-distilled-squad"):
        pass

    def run(self, sentence):
        pass


    
class TextSummarization(BaseModel):
    name="TextSummarization"
    description="useful when you want to summarize a sentence or a paragraph. It receives text as input. "\
             "The input to this tool should be a string. "
            
    def __init__(self, device, model_name="facebook/bart-large-cnn"):
        print(f"Initializing TextSummarization to {device}")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def run(self, sentence):
        self.model.to(self.device)
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)

        # Get the outputs from the model
        summary_ids = self.model.generate(**inputs)
        summary_text = self.tokenizer.decode(summary_ids[0]).strip("</s>")
        
        self.model.to('cpu')
        return summary_text

    
class TextGeneration(BaseModel):
    name="TextGeneration"
    description="useful when you want to generate a sentence. It receives text as input. "\
             "The input to this tool should be a string. "
            
    def __init__(self, device, model_name="bigscience/bloom-560m"):
        print(f"Initializing TextGeneration to {device}")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.generator = AutoModelForCausalLM.from_pretrained(self.model_name)

    def run(self, sentence):
        self.generator.to(self.device)
        
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)

        # Get the outputs from the model
        outputs = self.generator.generate(**inputs, min_length=5, max_length=30, pad_token_id=50256)

        # Decode the outputs to get the generated text
        generated_s = self.tokenizer.decode(outputs[0])
        
        self.generator.to('cpu')
        
        return generated_s
    

class MachineTranslation(BaseModel):
    name="MachineTranslation"
    description="useful when you want to translate a sentence. It receives text as input. "\
             "The input to this tool should be a string. "
            
    def __init__(self, device, model_name="t5-base"):
        print(f"Initializing MachineTranslation to {device}")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def run(self, sentence):
        self.translator.to(self.device)
        
        text = "Translate English to German: " + sentence

        # Encode the input with the tokenizer
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)

        # Get the outputs from the model
        translated_ids = self.translator.generate(**inputs, min_length=5, max_length=1000)

        # Decode the outputs to get the translated text
        translated_text = self.tokenizer.decode(translated_ids[0]).strip("<pad></s>")
        
        self.translator.to('cpu')
        
        return translated_text
    

class FillMask(BaseModel):
    name="FillMask"
    description="useful when you want to fill the sentence at the masked position. It receives text as input. "\
             "The input to this tool should be a string. "
            
    def __init__(self, device, model_name="distilbert-base-uncased"):
        print(f"Initializing FillMask to {device}")
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

    def run(self, sentence):
        self.model.to(self.device)
        
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)

        outputs = self.model(**inputs)

        # Get the logits from the outputs
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Get the top 5 tokens and their probabilities for the masked position
        masked_index = inputs.input_ids[0].tolist().index(self.tokenizer.mask_token_id)
        top_tokens = torch.topk(probabilities[masked_index], k=1)

        # Decode the tokens to get the words
        word = self.tokenizer.convert_ids_to_tokens(top_tokens.indices)
        completed_text = sentence.replace(self.tokenizer.mask_token, word[0])
        
        self.model.to('cpu')
        
        return completed_text

if __name__ == "__main__":
    from langchain.memory import ConversationBufferMemory

    from gradio_tools.tools import (StableDiffusionTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool,
                                    TextToVideoTool)

    llm = OpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # tools = [Text2Image("cuda:0").langchain, ImageCaptioning("cuda:1").langchain]

    # Image classification
    # tools = [ImageClassification("cuda:0").langchain]

    # Image Colorization
    # tools = [ImageColorization("cuda:7").langchain]

    # Object Detection
    tools = [MachineTranslation("cuda:4").langchain]

    agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

    # output = agent.run(input=("Please tell me the class of the photo whose path is image/3a2e1da9.png")) # image classification
    # output = agent.run(input=("Please help me colorize image whose path is image/3a2e1da9.png")) # image colorization
    # output = agent.run(input=("Please help me detect objects in the image whose path is image/3a2e1da9.png")) # object detection
    # output = agent.run(input=("Please help me convert the low-resolution image whose path is image/2.jpg to a high-resolution image")) # image super resolution
    output = agent.run(input=("Please help me translate this sentence: you are my son")) # object detection