from DataScraper import DataScraper
from ImageRetrieval import VLAD, NNet
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


def get_boolean_input(prompt):
    """Helper function to get a yes/no input from the user."""
    while True:
        response = input(prompt).strip().lower()
        if response in ["yes", "y", "1"]:
            return True
        elif response in ["no", "n", "0"]:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def get_positive_int_input(prompt, default):
    """Helper function to get a positive integer input from the user."""
    while True:
        value = input(f"{prompt} (default: {default}): ").strip()
        if not value:  # Use default if input is empty
            return default
        if value.isdigit() and int(value) > 0:
            return int(value)
        else:
            print("Invalid input. Please enter a positive integer.")


def main():
    parser = argparse.ArgumentParser(description="Image Retrieval CLI")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory path to resources")
    parser.add_argument("--query_image_path", type=str, required=True, help="Path to the query image (the image to compare with others in the database)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()

    output_dir = os.path.join(args.data_dir, "Images")

    
    create = get_boolean_input("Would you like to create a new database? (yes/no): ")

    # Ask user to choose retrieval method
    while True:
        method = input("Choose retrieval method (NN/VLAD): ").strip().lower()
        if method in ["nn", "vlad"]:
            break
        print("Invalid input. Please enter 'NN' or 'VLAD'.")

    # Initialize and create the database
    if method == "nn":
        db_path = os.path.join(args.data_dir, 'db_nn')
        retriever = NNet(db_path, create, verbose=args.verbose)
    else:
        db_path = os.path.join(args.data_dir, 'db_vlad')
        k = get_positive_int_input("Specify k", default=512)
        features_nb = get_positive_int_input("Specify number of features", default=3000)
        retriever = VLAD(db_path, create, k, features_nb, verbose=args.verbose)

    if create:
        retriever.fit(output_dir)
    else:
        print("Loading already existing DB")

    # Query image retrieval
    if args.verbose:
        print(f"Querying an image using {method.upper()}...")

    best_match_path = retriever.predict(args.query_image_path)

    if args.verbose:
        print("Retrieved image path:", best_match_path)

    compare_images(args.query_image_path, best_match_path)


if __name__ == "__main__":
    main()
