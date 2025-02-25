from DataScraper import DataScraper
from FeatureExtractor import FeatureExtractor
from FeatureExtractor import RootSIFT
from VLAD import VLAD
from DistrictParser import DistrictParser

import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np
from dotenv import load_dotenv
import argparse


def compose_panorama(root, ext):
    img_paths = [f"{root}_{i}{ext}" for i in range(3)]
    images = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) for path in img_paths]

    return np.hstack(images)


def compare_images(query_img_path, db_image_path):
    # Compare two images given teir path

    query_img = cv.cvtColor(cv.imread(query_img_path), cv.COLOR_BGR2RGB)
    db_img_root, db_img_ext = os.path.splitext(db_image_path)
    db_panorama = compose_panorama(db_img_root, db_img_ext)

    parts = db_img_root.split('_')
    latitude = float(parts[3])
    longitude = float(parts[4])

    fig, axes = plt.subplots(2, 1, figsize=(20, 8))
    axes[0].imshow(query_img, cmap='gray')
    axes[0].set_title("Query image")
    axes[0].axis("off")
    
    axes[1].imshow(db_panorama, cmap='gray')
    axes[1].set_title("DB image. latitude={}, longtitude={}".format(latitude, longitude))
    axes[1].axis("off")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="VLAD Image Retrieval using RootSIFT")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path to store images/db")
    parser.add_argument("--query_image_path", type=str, required=True, help="Path to the query image (the image to compare with others in the database)")
    parser.add_argument("--n_features", type=int, default=500, help="Max number of features to extract from query image")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    db_path = os.path.join(args.data_dir, 'db')
    vlad = VLAD(db_path, verbose=args.verbose)

    if vlad.database_exists():
        vlad.load()

    else:
        print("Creating new DB")
        print("Specify number of clusters")
        k = 512
        print(k)

        print("Specify number of features")
        features_nb = 3000
        print(features_nb)

        vlad.create(k)

        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        secret = os.getenv("GOOGLE_SECRET_KEY")

        if not api_key or not secret:
            raise ValueError("API key or Secret key is missing. Set them in a .env file.")
    
        if args.verbose:
            print("API Key and Secret loaded successfully.")
    
        centeres, r_mins, lengths, n_points = DistrictParser.get_districts()
        output_dir = os.path.join(args.data_dir, "Images")

        # data_scraper = DataScraper(api_key, secret)
        # data_scraper.retrieve_panoramas(r_mins, lengths, n_points, size, centeres, output_dir)

        ft_extractor = FeatureExtractor(features_nb)
        descriptors, paths, subdir_nb = ft_extractor.extract_features(output_dir)
    
        if args.verbose:
            print("Total images:", np.sum(subdir_nb))

        if args.verbose:
            print("Total descriptors:", len(np.vstack(descriptors)))
        
        vlad.fit(descriptors, paths)

    if args.verbose:
        print("Querying an image...")

    query_image = cv.imread(args.query_image_path, cv.IMREAD_GRAYSCALE)
    _, query_descriptors = RootSIFT(args.n_features).detect_and_compute(query_image)
    best_match_path = vlad.predict(query_descriptors)
    
    if args.verbose:
        print("Retrieved image path:", best_match_path)
    
    compare_images(args.query_image_path, best_match_path)


if __name__ == "__main__":
    main()
