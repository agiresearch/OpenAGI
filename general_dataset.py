""" 
This code is used to load augmented data samples.

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

__authors__ = "Jianchao Ji, Yingqiang Ge"
__copyright__ = "Copyright 2023, OpenAGI"
__date__ = "2023/04/09"
__license__ = "GPLv3"
__version__ = "0.0.1"



from torchvision import transforms
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import copy 

class GeneralDataset(Dataset):
    def __init__(self, folder_name, folder_path=None):
        self.folder_name = folder_name
        
        dir_path = folder_path+'{}/'.format(str(self.folder_name))
        
        #used to load the images 
        def image_loader(path):
            images = []
            file_names = os.listdir(path)
            file_len = len(file_names)
            for file in range(file_len):
                temp = Image.open(path + str(file)+".jpg")
                keep = temp.copy()
                images.append(keep)
                temp.close()
            
            return images
    
        #used to load the txt files
        def text_loader(path):
            text = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    text.append(line)
            f.close()
            return text
        
        #used to select image_loader or text loader
        def file_loader(path):
            files = []
            
            file_names = os.listdir(path)
            for file in file_names:
                if file[-3:] == 'txt':
                    files.append(text_loader(path+file))
                else:
                    files.append(image_loader(path+file+'/'))
                    
            return files
        
        input_path = dir_path + 'inputs/'
        output_path = dir_path + 'outputs/'
        
        test_path = os.listdir(output_path)
        

        
        self.input_files = file_loader(input_path)
        self.output_files = file_loader(output_path)
        



    def __len__(self):
        return len(self.input_files[0])

    def __getitem__(self, idx):
        
        #transform the images to embeddings
        transform_img = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.CenterCrop(256),
                                        transforms.PILToTensor(),
                                         ])
        
        input_files = []
        output_files = []
        
        #add the files(txt or images) to the input and output
        for contents in self.input_files:
            content = contents[idx]
            if isinstance(content,str):
                input_files.append(content)
            else:
                input_files.append(transform_img(content))
                
                
        for contents in self.output_files:
            content = contents[idx]
            if isinstance(content,str):
                output_files.append(content)
            else:
                output_files.append(transform_img(content.convert('RGB')))
                
                
        return {'input':input_files, 'output':output_files}