import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


class RootSIFT:
	def __init__(self, k):
		# Initialize the SIFT feature extractor
		self.extractor = cv.SIFT_create(k)
		
	def detect_and_compute(self, image, eps=1e-7):
		(kps, descs) = self.extractor.detectAndCompute(image, None)
		if len(kps) == 0:
			return ([], None)
		
		# Apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
	
		return (kps, descs)


class FeatureExtractor:
	def __init__(self, n_features):
		self.extractor_ = RootSIFT(n_features)
	

	def extract_features(self, image_dir):
		descriptors = []
		paths = []
		subdir_nb = []

		subdirs = sorted(os.listdir(image_dir))
		
		for subdir_name in tqdm(subdirs, desc="Processing subdirectories"):
			subdir_path = os.path.join(image_dir, subdir_name)

			if not os.path.isdir(subdir_path):  
				continue

			files = sorted(os.listdir(subdir_path))
			subdir_nb.append(len(files))

			for i in tqdm(range(0, len(files) - 2, 3), desc=f"Processing {subdir_name}", leave=False):
				img_paths = [os.path.join(subdir_path, files[i + j]) for j in range(3)]
				images = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2GRAY) for path in img_paths]
				panorama = np.hstack(images)
								
				_, img_des = self.extractor_.detect_and_compute(panorama)

				if img_des is not None:
					descriptors.append(img_des)

					file_root, ext = os.path.splitext(img_paths[0])
					trimmed_path = file_root[:-2] + ext 
					paths.append(trimmed_path)  

		return descriptors, paths, subdir_nb
 