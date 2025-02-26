from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import contextlib


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class FeatureExtractor(ABC):
	def __init__(self):
		pass
	

	def extract_features_from_image(self, image_path):
		image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
		return self._extract_features(image)


	def extract_features(self, image_dir):
		features = []
		paths = []
		subdir_nb = []

		subdirs = sorted(os.listdir(image_dir))
		
		for subdir_name in tqdm(subdirs, desc=f"Processing subdirectories"):
			subdir_path = os.path.join(image_dir, subdir_name)

			if not os.path.isdir(subdir_path):  
				continue

			files = sorted(os.listdir(subdir_path))
			subdir_nb.append(len(files))

			for i in tqdm(range(0, len(files) - 2, 3), desc=f"Processing {subdir_name}", leave=False):
				img_paths = [os.path.join(subdir_path, files[i + j]) for j in range(3)]
				images = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) for path in img_paths]
				panorama = np.hstack(images)
								
				img_features = self._extract_features(panorama)

				if img_features is not None:
					features.append(img_features)

					file_root, ext = os.path.splitext(img_paths[0])
					trimmed_path = file_root[:-2] + ext 
					paths.append(trimmed_path)  

		return features, paths, subdir_nb


	@abstractmethod
	def _extract_features(self, img_path):
		pass


class RootSIFT(FeatureExtractor):
	def __init__(self, k):
		super().__init__()

		# Initialize the SIFT feature extractor
		self.extractor = cv.SIFT_create(k)
		
	def _extract_features(self, image):
		image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
		_, descs = self.extractor.detectAndCompute(image, None)
		if descs is None or len(descs) == 0:
			return None
		
		# Apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		eps = 1e-7

		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
	
		return descs


class VGGNet(FeatureExtractor):
    def __init__(self):
        super().__init__()

        self.input_shape = (400, 1920)
        self.pooling = 'max'
        
        # Load pre-trained VGG16 model without fully connected layers
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(self.model.features.children()))  # Remove FC layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device).eval()

        # Define transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def _extract_features(self, image):
        # Convert image to tensor and move to GPU
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Extract features with torch.no_grad() to save memory
        with torch.no_grad():
            features = self.model(image)
        
        # Convert features to numpy and normalize
        features = features.cpu().numpy().flatten()
        norm_features = features / np.linalg.norm(features)

        return norm_features

