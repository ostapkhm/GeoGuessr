from DataScraper import DataScraper
from ImageRetrieval import VLAD, NNet
from utils import DistrictParser, get_boolean_input, get_positive_int_input

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
    # Compare two images given their paths

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
    parser = argparse.ArgumentParser(description="Image Retrieval CLI")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path to resources")
    parser.add_argument("--query_image_path", type=str, required=True, help="Path to the query image")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    secret = os.getenv("GOOGLE_SECRET_KEY")

    if not api_key or not secret:
        raise ValueError("API key or Secret key is missing. Set them in a .env file.")

    if args.verbose:
        print("API Key and Secret loaded successfully.")

    use_existing = get_boolean_input("Do you want to use existing images? (yes/no): ")

    if use_existing:
        output_dir = input("Enter the path to the folder containing existing images: ").strip()
        if not os.path.isdir(output_dir):
            raise ValueError("Invalid folder path. Please provide an existing directory.")
    else:
        centeres, r_mins, lengths, n_points = DistrictParser.get_districts()
        output_dir = input("Enter the folder path to store new images: ").strip()
        os.makedirs(output_dir, exist_ok=True)

        data_scraper = DataScraper(api_key, secret)
        data_scraper.retrieve_panoramas(r_mins, lengths, n_points, centeres, output_dir)

    create = get_boolean_input("Would you like to create a new database? (yes/no): ")

    print("Choose retrieval method:\n - NN (Neural Network-based)\n - VLAD (Vector of Locally Aggregated Descriptors)")
    while True:
        method = input("Enter method (NN/VLAD): ").strip().lower()
        if method in ["nn", "vlad"]:
            break
        print("Invalid input. Please enter 'NN' or 'VLAD'.")

    db_paths = os.path.join(args.data_dir, 'dbs')
    os.makedirs(db_paths, exist_ok=True)

    # Initialize and create the database
    if method == "nn":
        db_path = os.path.join(db_paths, 'db_nn')
        retriever = NNet(db_path, create, verbose=args.verbose)
    else:
        db_path = os.path.join(db_paths, 'db_vlad')
        k = get_positive_int_input("Specify k", default=512)
        features_nb = get_positive_int_input("Specify number of features", default=3000)
        retriever = VLAD(db_path, create, k, features_nb, verbose=args.verbose)

    if create:
        retriever.fit(output_dir)
    else:
        print("Loading already existing DB...")

    # Query image retrieval
    if args.verbose:
        print(f"Querying an image using {method.upper()}...")

    best_match_path = retriever.predict(args.query_image_path)

    if args.verbose:
        print("Retrieved image path:", best_match_path)

    compare_images(args.query_image_path, best_match_path)


if __name__ == "__main__":
    main()
