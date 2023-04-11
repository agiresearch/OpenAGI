""" 
This code mainly works on combine model squence and execute the solution.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/09"
__license__ = "GPLv3"
__version__ = "0.0.1"



import github_models.colorization.colorizers as colorizers
from github_models.colorization.colorizers import *
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
    VisionEncoderDecoderModel
)

from diffusers import StableDiffusionPipeline


import torch
import os
from runpy import run_path
from skimage import img_as_ubyte

import cv2

import warnings
warnings.filterwarnings('ignore')

import gc


set_seed(42)


class SeqCombine:
    def __init__(self, device_list=['cpu']):
        self.device_list = device_list
        
        self.img_classifier_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224', device_map = 'auto')
        self.img_classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')#.to(self.device)
        self.img_classifier.eval()

        self.colorizer= colorizers.siggraph17().eval()#.to(self.device)
        

        self.object_detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")
        self.object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101")#.to(self.device)
        self.object_detector.eval()
        
        self.image_super_resolution_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        self.image_super_resolution_model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        self.image_super_resolution_model.eval()
        
        self.image_caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")#.to(self.device)
        self.image_caption_model.eval()
        self.image_caption_feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.image_caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.text_to_image_generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")#, torch_dtype=torch.float16)#.to(self.device)
        def dummy(images, **kwargs):
            return images, False
        self.text_to_image_generator.safety_checker = dummy
        self.text_to_image_generator.enable_attention_slicing()
        
        self.sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_analysis_module = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_analysis_module.eval()
        
        self.question_answering_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.question_answerer = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
        self.question_answerer.eval()
        
        self.summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.summarizer.eval()
        
        
        self.text_generation_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.text_generation_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.text_generator = AutoModelForCausalLM.from_pretrained("gpt2")
        self.text_generator.eval()
        
        self.vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model.eval()
        
        
        self.img_transform = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.PILToTensor(),
                                         ])
        
        #load debluring and denoising models
        parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
        weights = os.path.join('Restormer','Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring.pth')
        load_arch = run_path(os.path.join('github_models','Restormer','basicsr', 'models', 'archs', 'restormer_arch.py'))
        self.image_deblurring_model = load_arch['Restormer'](**parameters)
        #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        #model.to(device)
        checkpoint = torch.load(weights)
        self.image_deblurring_model.load_state_dict(checkpoint['params'])
        self.image_deblurring_model.eval()
        
        parameters = {'inp_channels':3, 'out_channels':3, 'dim':48, 'num_blocks':[4,6,6,8], 'num_refinement_blocks':4, 'heads':[1,2,4,8], 'ffn_expansion_factor':2.66, 'bias':False, 'LayerNorm_type':'WithBias', 'dual_pixel_task':False}
        weights = os.path.join('Restormer','Denoising', 'pretrained_models', 'real_denoising.pth')
        parameters['LayerNorm_type'] =  'BiasFree'
        load_arch = run_path(os.path.join('github_models','Restormer','basicsr', 'models', 'archs', 'restormer_arch.py'))
        self.image_denoising_model = load_arch['Restormer'](**parameters)
        #device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        #model.to(device)
        checkpoint = torch.load(weights)
        self.image_denoising_model.load_state_dict(checkpoint['params'])
        self.image_denoising_model.eval()


        self.translation_tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.translator = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        self.translator.eval()
        
        self.unmask_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.unmasker = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        self.unmasker.eval()
        
        self.module2function_dict = {
                                    "Image Classification": ["self.image_classification", self.img_classifier], \
                                    "Colorization": ["self.image_colorization", self.colorizer], \
                                    "Object Detection": ["self.image_object_detect", self.object_detector], \
                                    "Image Deblurring": ["self.image_deblurring", self.image_deblurring_model], \
                                    "Image Denoising": ["self.image_denoising", self.image_denoising_model], \
                                    "Image Super Resolution": ["self.image_super_resolution", self.image_super_resolution_model], \
                                    "Image Captioning": ["self.image_caption", self.image_caption_model], \
                                    "Text to Image Generation": ["self.text_to_image_generation", self.text_to_image_generator], \
                                    "Visual Question Answering": ["self.vqa", self.vqa_model], \
                                    "Sentiment Analysis": ["self.sentiment_analysis", self.sentiment_analysis_module], \
                                    "Question Answering": ["self.question_answering", self.question_answerer], \
                                    "Text Summarization": ["self.text_summarization", self.summarizer], \
                                    "Text Generation": ["self.text_generation", self.text_generator], \
                                    "Machine Translation": ["self.machine_translation", self.translator], \
                                    "Fill Mask": ["self.fill_mask", self.unmasker], \
                                    }
                                    
    
    def construct_module_seq(self, generated_module_seq):
        module_list = generated_module_seq.split(",")
        self.module_function_list = []
        self.module_list = []
        self.used_device_list = []
        i = 0
        cur_device = self.device_list[i]
        
        for module in module_list:
            module = module.strip()
            temp_values = self.module2function_dict[module]
            temp_m = temp_values[1]

            if cur_device != "cpu": 
                if torch.cuda.mem_get_info(cur_device)[0]/1024**3 >= 3:
                    temp_m = temp_m.to(cur_device)
                    self.used_device_list.append(cur_device)
                else:
                    i += 1 
                    cur_device = self.device_list[i]
                    temp_m = temp_m.to(cur_device)
                    self.used_device_list.append(cur_device)
            else:
                temp_m = temp_m.to(cur_device)
                self.used_device_list.append(cur_device)


            temp_f = eval(temp_values[0])
                
            
            self.module_function_list.append(temp_f)
            self.module_list.append(temp_m)
            
    
    def run_module_seq(self, input_data):
        temp = input_data
        for i,m in enumerate(self.module_function_list):
            temp = m(temp, self.used_device_list[i])
        return temp
    
    def close_module_seq(self):
        for m in self.module_list:
            m = m.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        gc.collect()
        return
        
        
    def image_classification(self, imgs, device):
        img_classifier_inputs = self.img_classifier_feature_extractor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            img_classifier_outputs = self.img_classifier(**img_classifier_inputs)
        img_classifier_logits = img_classifier_outputs.logits
        # img_classifier_logits.shape
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = img_classifier_logits.argmax(1)#.item()
        predicted_class = [self.img_classifier.config.id2label[i.item()] for i in predicted_class_idx]
        
        return predicted_class
    
    def image_colorization(self, imgs, device):
        temp_imgs = []
        for img in imgs:
            img = img.permute(1,2,0).cpu().numpy()
            (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
            tens_l_rs = tens_l_rs.to(device)

            # colorizer outputs 256x256 ab map
            # resize and concatenate to original L channel
            img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
            out_img = postprocess_tens(tens_l_orig, self.colorizer(tens_l_rs).cpu())

            norm_out_img = cv2.normalize(out_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            norm_out_img = norm_out_img.astype(np.uint8)

            # colorized_img = Image.fromarray(norm_out_img,'RGB')
            # temp_imgs.append(torch.from_numpy(np.array(colorized_img)).permute(2,0,1))
            temp_imgs.append(torch.from_numpy(norm_out_img).permute(2,0,1))
        return temp_imgs
    
    def image_object_detect(self, imgs, device):
        imgs = torch.stack(imgs)
        object_detector_inputs = self.object_detector_processor(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            object_detector_outputs = self.object_detector(**object_detector_inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([[object_detector_inputs['pixel_values'].shape[2],\
                                      object_detector_inputs['pixel_values'].shape[3]] \
                                     for i in range(object_detector_inputs['pixel_values'].shape[0])]).to(device)
        results = self.object_detector_processor.post_process_object_detection(object_detector_outputs, target_sizes=target_sizes, threshold=0.9)

        predicted_results = []
        for r in results:
            output = ""
            for score, label, box in zip(r["scores"], r["labels"], r["boxes"]):
                output += self.object_detector.config.id2label[label.item()]
                output += ", "
            predicted_results.append(output[:-2])
        
        return predicted_results
    
    def image_caption(self, imgs, device):
        """
        input:
            imgs: list of image tensors
        output:
            preds: list of strings
        """
        max_length = 40
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        pixel_values = self.image_caption_feature_extractor(images=imgs, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            output_ids = self.image_caption_model.generate(pixel_values, **gen_kwargs)

        preds = self.image_caption_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

    def text_to_image_generation(self, prompts, device):
        with torch.no_grad():
            images = self.text_to_image_generator(prompts).images
            
        return [self.img_transform(im) for im in images]
        
    def image_super_resolution(self, imgs, device):
        """
        imgs: can be list of Images or list of image tensor (3,H,W) or list of image array (3,H,W), 
              while we fix it to be list of image tensor
        output: numpy.array (3,H,W,B)
        res: list of image tensor where each element is (3,H,W)
        """
        batch_size = len(imgs)
        inputs = torch.stack(imgs).permute(0,2,3,1)
        inputs = self.image_super_resolution_processor(inputs, return_tensors="pt").to(device)

        # forward pass
        with torch.no_grad():
            outputs = self.image_super_resolution_model(**inputs)

        reformed_outputs = []
        for i in range(batch_size):
            output_ = outputs.reconstruction.data[i]
            output = output_.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.moveaxis(output, source=0, destination=-1)
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
            reformed_outputs.append(torch.from_numpy(output).permute(2,0,1))
        return reformed_outputs
    
    def vqa(self, input_data, device):
        """
        input:
            imgs: list of image tensors
            questions: list of strings
        output:
            answers: list of strings
        """
        imgs = input_data[1]
        questions = list(input_data[0])
        
        encoding = self.vqa_processor(imgs, questions, return_tensors="pt", padding=True).to(device)
        # forward pass
        with torch.no_grad():
            outputs = self.vqa_model(**encoding)
        logits = outputs.logits
        idxs = torch.argmax(logits, 1)
        answers = [self.vqa_model.config.id2label[idx.item()] for idx in idxs]

        return answers
    
    def sentiment_analysis(self, sentences, device):
        """
        input:
            sentences: list of strings
        output:
            predicted_labels: list of strings
        """
        inputs = self.sentiment_analysis_tokenizer(sentences, return_tensors="pt", padding=True).to(device)

        # Get the outputs from the model
        with torch.no_grad():
            outputs = self.sentiment_analysis_module(**inputs)

        # Get the logits from the outputs
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get the most likely label and score
        predicted_label_ids = torch.argmax(probabilities, 1)#.item()
        predicted_labels = [self.sentiment_analysis_module.config.id2label[i.item()] for i in predicted_label_ids]
        
        return predicted_labels
       
    
    def question_answering(self, input_data, device):
        """
        input:
            questions: list of strings
            contexts: list of strings
        output:
            results: list of strings
        """
        questions = list(input_data[1])
        contexts = list(input_data[0])
        batch_size = len(questions)
        
        inputs = self.question_answering_tokenizer(questions, contexts, return_tensors="pt", padding=True).to(device)

        # Get the outputs from the model
        with torch.no_grad():
            outputs = self.question_answerer(**inputs)

        # Get the start and end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Get the most likely start and end indices
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)

        # Get the answer span from the inputs
        results = []
        for i in range(batch_size):
            answer_ids = inputs["input_ids"][i][start_index:end_index+1]
            answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
            answer_text = tokenizer.convert_tokens_to_string(answer_tokens)

            # Print the answer
            results.append(answer_text)
            
        return results
    
    
    def text_summarization(self, text, device):
        """
        input:
            text: list of strings
        output:
            summary_text: list of strings
        """
        inputs = self.summarization_tokenizer(text, return_tensors="pt", padding=True).to(device)

        # Get the outputs from the model
        with torch.no_grad():
            outputs = self.summarizer.generate(**inputs)
        summary_text = [self.summarization_tokenizer.decode(summary_ids).strip("</s>") for summary_ids in outputs]
        
        return summary_text
    
    
    def machine_translation(self, sentences, device):
        """
        input:
            sentences: list of Englisth strings
        output:
            translated_text: list of German strings
        """
        text = ["translate English to German: " + s for s in sentences]

        # Encode the input with the tokenizer
        inputs = self.translation_tokenizer(text, return_tensors="pt", padding=True).to(device)

        # Get the outputs from the model
        with torch.no_grad():
            outputs = self.translator.generate(**inputs, min_length=5, max_length=1000)

        # Decode the outputs to get the translated text
        translated_text = [self.translation_tokenizer.decode(translated_ids).strip("<pad></s>") for translated_ids in outputs]

        return translated_text
        
    def text_generation(self, sentences, device):
        """
        input:
            sentences: list of strings
        output:
            generated_text: list of strings
        """
        res = []
        for s in sentences:
            # Encode the input with the tokenizer
            inputs = self.text_generation_tokenizer(s, return_tensors="pt", padding=True).to(device)

            # Get the outputs from the model
            with torch.no_grad():
                outputs = self.text_generator.generate(**inputs, min_length=5, max_length=30, pad_token_id=50256)

            # Decode the outputs to get the generated text
            generated_s = self.text_generation_tokenizer.decode(outputs[0])
            res.append(generated_s)
        return res
        
    
    def fill_mask(self, sentences, device):
        """
        input:
            sentences: list of strings with "[MASK]"
        output:
            results: lsit of strings
        """
        batch_size = len(sentences)
        inputs = self.unmask_tokenizer(sentences, return_tensors="pt", padding=True).to(device)

        # Get the outputs from the model
        with torch.no_grad():
            outputs = self.unmasker(**inputs)

        # Get the logits from the outputs
        logits = outputs.logits

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        results = []
        for i in range(batch_size):
            # Get the top 5 tokens and their probabilities for the masked position
            masked_index = inputs.input_ids[i].tolist().index(self.unmask_tokenizer.mask_token_id)
            top_tokens = torch.topk(probabilities[i][masked_index], k=1)

            # Decode the tokens to get the words
            word = self.unmask_tokenizer.convert_ids_to_tokens(top_tokens.indices)
            completed_text = sentences[i].replace(self.unmask_tokenizer.mask_token, word[0])
            results.append(completed_text)
            
        return results
       
    
    def image_deblurring(self, imgs, device):
        restoreds = []
        with torch.no_grad():
            img_multiple_of = 8
            
            for cur in imgs:

                h = cur.shape[1]
                w = cur.shape[2]
                img = cur.contiguous().view(h, w, 3)

                input_ = img.float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

                height,width = input_.shape[2], input_.shape[3]
                H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
                padh = H-height if height%img_multiple_of!=0 else 0
                padw = W-width if width%img_multiple_of!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                restored = self.image_deblurring_model(input_)

                restored = torch.clamp(restored, 0, 1)

                restored = restored[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0]).reshape((3,h,w))

                restored = torch.from_numpy(restored)
                restoreds.append(restored)
            #print(restored)
            return restoreds
    
    def image_denoising(self,imgs, device):
        restoreds = []
        img_multiple_of = 8
        with torch.no_grad():
            for cur in imgs:
                
                h = cur.shape[1]
                w = cur.shape[2]
                img = cur.contiguous().view(h, w, 3)
            
                input_ = img.float().div(255.).permute(2,0,1).unsqueeze(0).to(device)
               

                height,width = input_.shape[2], input_.shape[3]
                H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
                padh = H-height if height%img_multiple_of!=0 else 0
                padw = W-width if width%img_multiple_of!=0 else 0
                input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

                restored = self.image_denoising_model(input_)

                restored = torch.clamp(restored, 0, 1)

                restored = restored[:,:,:height,:width]

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = img_as_ubyte(restored[0]).reshape((3,h,w))

                restored = torch.from_numpy(restored)
                restoreds.append(restored)
                #print(restored)
            return restoreds
        