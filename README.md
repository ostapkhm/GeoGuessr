<!-- ## Overall goal of a project

Design a location recognition model by processing street view images of a city and building a database of feature embeddings for location matching. 

## Data Scraping

To obtain Street View panoramas in Kyiv, the Google Street View Static API was used. Only outdoor panoramas were selected for image recognition. Three consecutive images to form a panorama were retrieved, typically with headings of 0°, 120°, and 240°, and a field of view (FoV) of 120°. To scrape data from specific district, I start with a square of a given size centered on a specific point, aligned with cardinal directions (north, south, east, and west).

This square is then subdivided into smaller circles with defined radii, and the centers of these circles serve as the coordinates for the initial panorama search. For example:

![Example1](https://github.com/user-attachments/assets/58eff5fb-642c-444d-99aa-3e5b635a8227)

After specifying several districts we have the following map:
[output_map.html](assets/output_map.html). Altough we haven't scraped all Kyiv (it would be at cost:)), the data is quite challanging for location matching. There are lot of location with simmilar foliage or builduings(Troieshchyna is a good example).  

###  Image retrieval using VLAD and RootSIFT
As for the first approach we would use features extracted from the images using the SIFT detector. Specifically, I use the RootSIFT variant, which improves codebook clustering by leveraging the Hellinger distance metric. For example:

![Example2](https://github.com/user-attachments/assets/29c4b8fa-ddd2-4b7b-8c2e-8986e033a62c)


Next, I aggregate the features of each image using VLAD with a specified codebook size and store them in ChromaDB for later retrieval.


### Image retrieval using ViT
As for the second approach I have used vision transformer, mainly pretrained vit_b_16 on imagenet. It was primarely used for feature extraction, resulting in $768$ dimensional vector. Because default panorama size is $1920 \times 400$ I divided it into three parts, used ViT for feature extraction on each part and then concatenated embeddings. The resulting feature embedding has size of $2304$.  

### Comparison
You can see the comparison of two different retrieval methods in ```Comparison.ipynb```. From there we can clearly conclude that ViT are superior for image retrieval on this data.

### How to run
To run ```geoguessr.py``` first create conda environment using ```environment.yml```. Then you can run ```geoguessr.py``` with the following arguments: --data_dir "path_to+store_data" --query_image_path "query_image_path" --verbose.

### Limitations
The crucial limitations is distortions caused by taking omnidirectional images. This defenitely causes SIFT and ViT to extract features from  some images badly.

Also it would be interesting to fine tune ViT feature extraction on panaramic images given their coordinates.


Here’s a refined version of your explanation with improved clarity, corrections, and a more formal tone:  

--- -->

## **Overall Goal of the Project**  

The objective of this project is to develop a location recognition model by processing street view images of a city and constructing a database of feature embeddings for location matching.  

---

## **Data Scraping**  

To obtain Street View panoramas in Kyiv, the **Google Street View Static API** was utilized, selecting only outdoor panoramas suitable for image recognition. Each panorama is composed of three consecutive images captured with headings of **0°**, **120°**, and **240°**, each with a **120° field of view (FoV)**.  

To scrape data from a specific district, the process begins with a **square region of a given size**, centered on a specified point and aligned with the cardinal directions (north, south, east, and west). This square is then subdivided into smaller circles of predefined radii, with the centers of these circles serving as the initial coordinates for panorama retrieval.  

For example:  
![Example1](https://github.com/user-attachments/assets/58eff5fb-642c-444d-99aa-3e5b635a8227)  

After specifying multiple districts, the collected data is visualized in:  
<a href="assets/output_map.html" target="_blank">output_map.html</a>. 

While the entire city of Kyiv was not scraped (due to cost constraints), the dataset presents significant challenges for location matching. There are a lot of locations with similar foliage or buildings (Troieshchyna is a good example).

---

## **Image Retrieval Using VLAD and RootSIFT**  

As a first approach, features are extracted using the SIFT detector, specifically its RootSIFT variant, which enhances codebook clustering by leveraging the Hellinger distance metric. The extracted features are then aggregated using VLAD (Vector of Locally Aggregated Descriptors) with a specified codebook size and stored in ChromaDB for efficient retrieval.  

---

## **Image Retrieval Using Vision Transformer (ViT)**  

As a second approach, a Vision Transformer (ViT) is employed, specifically the pretrained ViT-B/16 model on ImageNet, used primarily for feature extraction. Feature embeddings are also stored in ChromaDB for efficient retrieval.

Since the default panorama size is $1920 \times 400$, each panorama is divided into three segments, with ViT extracting features from each segment individually. The embeddings are then concatenated, resulting in a $2304$-dimensional feature vector.  

---

## **Comparison of Retrieval Methods**  

A comparison of the two retrieval approaches can be found in ```Comparison.ipynb```. The results clearly demonstrate that ViT-based retrieval outperforms VLAD + RootSIFT on this dataset.  

---

## **How to Run the Model**  

To run **geoguessr.py**, follow these steps:  

1. Create a **Conda environment** using `environment.yml`.  
2. Run the script with the following arguments:  
   ```bash
   python geoguessr.py --data_dir "path_to_store_data" --query_image_path "query_image_path" --verbose
   ```  

---

## **Limitations and Future Work**  

- **Omnidirectional Image Distortions**: Both SIFT and ViT struggle with feature extraction due to distortions inherent in panoramic images.

- **Fine-tuning ViT**: Training ViT on panoramic images with associated geographic coordinates could enhance feature extraction and location recognition performance.  
