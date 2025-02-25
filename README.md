## Task

Your task is to design a location recognition model by processing Street View images of a city and building a database of SIFT descriptors for location matching. You will:

- Download Street View Panoramas: Use Google Street View Static API to download panoramas of Kyiv. Or you can use streetview lib.

- Extract SIFT Features: Process each panorama to extract SIFT descriptors.
- Store Features in a Vector Database: Aggregate SIFT descriptors for each panorama and store these representations in a vector database.
- Query the Database: Implement a search function that finds the closest panorama match for a given query image using SIFT-based similarity search.
- Test: Play with different photos (not from Street View, better make them yourself).

## My approach

### Part 1 (Data collection)
To obtain Street View panoramas in Kyiv, I used the Google Street View Static API. It's important to note that panorama sources can differ—they may be either outdoor or indoor. I specifically select outdoor panoramas; however, some outdoor photospheres are classified as indoor because the author did not provide additional information.

I retrieve three consecutive images to form a panorama, typically with headings of 0°, 120°, and 240°, and a field of view (FoV) of 120°. To cover a specific district, I start with a square of a given size centered on a specific point, aligned with cardinal directions (north, south, east, and west).

This square is then subdivided into smaller circles with defined radii, and the centers of these circles serve as the coordinates for the initial panorama search. For example:

![example](https://github.com/user-attachments/assets/92ac2dd5-cb59-472a-babf-56bae4518607)

Note that these initial coordinates might not always contain panoramas. Therefore, I search for panoramas within these circles. I encountered some challenges with retrieving images within a specified radius. While the Google API provides a radius parameter for searching panoramas, it can sometimes return panoramas outside the specified radius.

To address this, I use a custom distance function based on the haversine formula. After obtaining accurate panorama coordinates using a metadata request, I verify whether each lies within the specified circle radius. For instance, in the example below, the red tags represent panoramas lying outside their circles:

![example2](https://github.com/user-attachments/assets/1a1a6f15-8d8d-4b0b-8799-479a3bf4bdb7)

__Note:__ all sampled panoramas you can view in __output_map.html__ file.

### Part 2 (Feature Extraction and Similarity Search Workflow)

After collecting the data, I extract features from the images using the SIFT detector. Specifically, I use the RootSIFT variant, which improves codebook clustering by leveraging the Hellinger distance metric. I limit the number of features to $1500$ per image. Here's an example:

![example3](https://github.com/user-attachments/assets/c3de130d-12e9-4b12-b85d-454cac1d5829)

Next, I aggregate the features of each image using VLAD with a codebook size of 512 and store them in ChromaDB for later retrieval. However, the retrieval results are not very accurate, with a mean distance error of approximately $3382$ meters. This is likely due to poor clustering of the codebook, as I used MiniBatchKMeans for clustering.

Here is an example of the worst result:

![worst example](https://github.com/user-attachments/assets/511145a9-8ddb-4218-90fd-c815a2149bcb)

And here is a good example (it's quite difficult to determine if the images are taken 100 meters apart on the same street):

![good_example](https://github.com/user-attachments/assets/836fab8b-5262-4865-9b80-40bb58e97f3d)


### TODO
- Find better codebook to increase retrieval accuracy
- Add more panoramas