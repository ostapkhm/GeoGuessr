from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from PIL import Image


import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights



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
	def _extract_features(self, image):
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


class ViT(FeatureExtractor):
	def __init__(self):
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Initialize ViT model
		self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

		# Create a modified forward method to get embeddings instead of classification
		self.original_forward = self.model.forward
		self.model.forward = self._forward_features

		self.model.eval()
		self.model.to(self.device)

		# ViT specific transforms
		self.transform = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
		])

	def _forward_features(self, x):
		# Process input
		x = self.model._process_input(x)
		n = x.shape[0]

		# Add class token
		cls_token = self.model.class_token.expand(n, -1, -1)
		x = torch.cat([cls_token, x], dim=1)
		x = self.model.encoder(x)

		# Return CLS token embedding
		return x[:, 0]


	@torch.no_grad()
	def _extract_features(self, image):
		width = image.shape[1] // 3

		# Split image into 3 equal parts
		image_parts = [
			image[:, :width],  # Left
			image[:, width:2 * width],  # Center
			image[:, 2 * width:3 * width]  # Right
		]

		images = torch.stack([self.transform(Image.fromarray(part)) for part in image_parts]).to(self.device)
		features = self.model(images)
		features = features.view(features.shape[0], -1).cpu().numpy()

		norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)
		final_features = np.concatenate(norm_features, axis=0)

		return final_features


	def __del__(self):
		# Restore original forward method when object is destroyed
		if hasattr(self, 'original_forward'):
			self.model.forward = self.original_forward
