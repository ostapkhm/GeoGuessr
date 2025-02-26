from abc import ABC, abstractmethod

import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import contextlib

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


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

		self.input_shape = (224, 224, 3)
		self.weight = 'imagenet'
		self.pooling = 'max'
		self.model = VGG16(weights = self.weight, 
							input_shape = (self.input_shape[0], 
							self.input_shape[1], self.input_shape[2]), 
							pooling = self.pooling, 
							include_top = False)
						

	def _extract_features(self, image):
		# Resize image
		image = cv.resize(image, (self.input_shape[0], self.input_shape[1]))
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)

		# Suppress output during inference
		with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
			features = self.model.predict(image)

		norm_features = features[0] / np.linalg.norm(features[0])
		return norm_features

