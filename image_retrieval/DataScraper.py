import hashlib
import hmac
import base64
import urllib.parse as urlparse
import requests
import time

from PIL import Image
from io import BytesIO

import numpy as np
import folium
import os


class PanoUnavailableError(Exception):
    """ 
        Exception raised when panorama is not available at the given coordinates. 
    """
    pass

class ImageRetrievalError(Exception):
    """ 
        Exception raised when there is an error retrieving the image. 
    """
    pass


class DataScraper:
    def __init__(self, api_key, secret):
        self.api_key_ = api_key
        self.secret = secret
    
    
    @staticmethod
    def to_geodesic(pts, center):
        # Convert Cartesian coordinates to geodesic coordinates.

        R = 6378137
        pts = np.copy(pts)

        if np.ndim(pts) == 1:
            pts = np.array([pts])

        lat = np.degrees(pts[:, 1] / R)
        long = np.degrees(pts[:, 0] / R) / np.cos(np.radians(center[0]))

        return np.vstack([lat, long]).T + center
    

    @staticmethod
    def distance(pt1, pt2):
        # Calculate distance between two points
        # which have geodesic coordinates using haversine formula.

        R = 6371000
        pt1_r = np.radians(pt1)
        pt2_r = np.radians(pt2)

        dphi = pt2_r[0] - pt1_r[0]
        dlam = pt2_r[1] - pt1_r[1]

        a = np.sin(dphi/2)**2 + np.cos(pt1_r[0]) * np.cos(pt2_r[0]) * np.sin(dlam / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c
    

    def draw_map(self, coordinates, center, length, map):
        # Plot coordinates on a map.

        length /= 2
        corners = np.array([[-length, -length],
                            [length, -length],
                            [length, length],
                            [-length, length],
                            [-length, -length]])
            
        square_coords = self.to_geodesic(corners, center)

        folium.PolyLine(
            locations=square_coords,
            color="red",
            weight=5,
            opacity=0.7
        ).add_to(map)

        # Add each coordinate as a marker on the map
        for idx in range(len(coordinates)):
            lat, long = coordinates[idx, 0], coordinates[idx, 1]
            folium.Marker([lat, long]).add_to(map)

        return map
    

    def sign_url(self, input_url=None):
        # Sign a request URL with a URL signing secret.

        if not input_url or not self.secret:
            raise Exception("Both input_url and secret are required")

        url = urlparse.urlparse(input_url)
        url_to_sign = url.path + "?" + url.query

        # Decode the private key into its binary format
        decoded_key = base64.urlsafe_b64decode(self.secret)

        # Create a signature using the private key and the URL-encoded string using HMAC SHA1.
        signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

        # Encode the binary signature into base64 for use within a URL
        encoded_signature = base64.urlsafe_b64encode(signature.digest())
        original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

        return original_url + "&signature=" + encoded_signature.decode()


    def get_pano_coordinates(self, coordinate, radius=50):
        # Check if a panorama exists within a 50-meter radius of a given coordinate. 
        # If found, return its coordinates. Otherwise, return None.
        
        # Get pano metadata 
        metadata_url = (
            f"https://maps.googleapis.com/maps/api/streetview/metadata"
            f"?location={coordinate[0]},{coordinate[1]}&source=outdoor"
            f"&radius={radius}&key={self.api_key_}"
        )
        
        signed_url = self.sign_url(metadata_url)

        current_delay = 0.5
        max_delay = 10

        while current_delay < max_delay:
            try:
                response = requests.get(signed_url, timeout=5)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying...")
            else:
                json_response = response.json()

                if json_response["status"] == "OK":
                    coordinates = json_response["location"]
                    return coordinates["lat"], coordinates["lng"]
                
                elif json_response["status"] == "ZERO_RESULTS":
                    return None
            
            time.sleep(current_delay)
            current_delay *= 2
        
        return None
    

    def sample_coordinates_around(self, center, length, n_points, r_min):
        # Sample coordinates within a square region defined by the center coordinate and side length.

        sample = np.arange(-length/2 + r_min, length/2 - r_min, 2*r_min)
        xv, yv = np.meshgrid(sample, -sample)

        ys = yv.flatten()
        xs = xv.flatten()

        candidate_coordinates = self.to_geodesic(np.vstack((xs, ys)).T, center)
        
        valid_coords = []
        for coordinate in candidate_coordinates:
            pano_coord = self.get_pano_coordinates(coordinate, r_min)
            if pano_coord is not None:
                distance_between = self.distance(coordinate, pano_coord)

                if distance_between <= r_min:
                    valid_coords.append(pano_coord)
                    if len(valid_coords) == n_points:
                        break

        valid_coords = np.array(valid_coords)
        return np.round(valid_coords, 6)

    
    def retrieve_image(self, coordinate, heading, size=(640, 400), pitch=0.0):
        # Retrieve a Google Street View image based on the provided parameters.
        # Raises:
        #     PanoUnavailableError: If panorama is not available at the given coordinates.
        #     ImageRetrievalError: If there is an error retrieving the image.
        
        
        pano_coord = self.get_pano_coordinates(coordinate)
        if pano_coord is None:
            raise PanoUnavailableError("Panorama is not available at the given coordinates.")


        # Format the URL with the input parameters
        image_url = (
            f"https://maps.googleapis.com/maps/api/streetview?"
            f"size={size[0]}x{size[1]}&location={pano_coord[0]},{pano_coord[1]}"
            f"&heading={heading}&pitch={pitch}&fov=120&source=outdoor&key={self.api_key_}"
        )

        signed_url = self.sign_url(image_url)
        
        current_delay = 0.5
        max_delay = 5

        while current_delay < max_delay:
            try:
                response = requests.get(signed_url, timeout=5)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}. Retrying...")
            else:
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    return image

                elif response.status_code in {400, 404}:
                    raise ImageRetrievalError(f"Client error: {response.status_code} - {response.reason}")
            
            time.sleep(current_delay)
            current_delay *= 2 

        raise ImageRetrievalError("Too many retry attempts.")


    def retrieve_panorama(self, coordinates, output_dir, center, length, size=(640, 400), pitch=0):
        # Retrieve a set of 360-degree panoramas at the given coordinates and save the images to a specified directory.
        # The function saves images in a structured subdirectory based on the center coordinate and region length.
        
        headings = np.array([0, 120, 240])
        
        subdirectory_name = f"{center[0]}_{center[1]}_{length}"
        subdirectory_path = os.path.join(output_dir, subdirectory_name)
        os.makedirs(subdirectory_path, exist_ok=True)
        
        for coordinate in coordinates:
            for i in range (0, headings.shape[0]):
                try:
                    image = self.retrieve_image(coordinate, headings[i], size, pitch)
                    image_path = (
                        "{dir_path}/image_{lat}_{lon}_{heading_id}.jpg"
                    ).format(dir_path=subdirectory_path, lat=coordinate[0], lon=coordinate[1], heading_id=i)

                    image.save(image_path)

                except (ImageRetrievalError, PanoUnavailableError) as e:
                    print(f"Skipping image at {coordinate} due to error: {e}")

    
    def retrieve_panoramas(self, r_mins, lengths, n_points, centeres, output_dir, size=(640, 400)):
        # Retrieve panoramas at multiple centers and save the images and map to the specified output directory.
        os.makedirs(output_dir, exist_ok=True)

        map = folium.Map(location=centeres[0], zoom_start=14)
        all_coordinates = []

        for idx in range(len(centeres)):
            r_min = r_mins[idx]
            length = lengths[idx]
            points_nb = n_points[idx]
            center = centeres[idx]

            coordinates = self.sample_coordinates_around(center, length, points_nb, r_min)
            map = self.draw_map(coordinates, center, length, map)
            all_coordinates.append(coordinates)
        
        map_path = os.path.join(output_dir, "output_map.html")
        map.save(map_path)

        for idx in range(len(centeres)):
            coordinates = all_coordinates[idx]
            length = lengths[idx]
            center = centeres[idx]

            self.retrieve_panorama(coordinates, size, output_dir, center, length)
