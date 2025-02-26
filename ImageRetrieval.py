from abc import ABC, abstractmethod

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import chromadb
import os

from FeatureExtractor import RootSIFT, VGGNet


class ImageRetrieval(ABC):
    def __init__(self, db_path, create, verbose=True):
        self.db_path_ = db_path
        self.verbose_ = verbose

        self.feature_extractor_ = None

        os.makedirs(self.db_path_, exist_ok=True)
        chromadb_path = os.path.join(self.db_path_, "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)

        self.chroma_client_ = chromadb.PersistentClient(path=chromadb_path)
        self.setup_database(create)


    def setup_database(self, create):
        """Initializes or resets the database based on the create flag."""
        self.database_ = self.chroma_client_.get_or_create_collection(name="vectors_embedding")

        if create:
            if self.database_exists():
                self.clear_database()
            self.database_ = self.chroma_client_.get_or_create_collection(name="vectors_embedding")
        else:
            if not self.database_exists():
                raise FileNotFoundError("Database does not exist! Run with 'create=True' first.")


    def database_exists(self):
        # Checks if at least one vector exists
        return len(self.database_.peek(1)["ids"]) > 0
    

    def save_to_chromadb(self, embeddings, paths):
        if self.verbose_:
            print("Storing vectors in database...")
            iterable = tqdm(range(len(embeddings)), desc="Saving embeddings")
        else:
            iterable = range(len(embeddings))

        for i in iterable:
            embedding = embeddings[i]
            self.database_.add(
                embeddings=[embedding.tolist()],
                ids=[paths[i]]
            ) 


    def predict(self, image_path):
        return self.get_knn(image_path, 1)[0]
    

    @abstractmethod
    def fit(self, image_dir):
        pass
    
    @abstractmethod
    def get_knn(self, image_path, k):
        pass
    
    @abstractmethod
    def clear_database(self):
        pass



class VLAD(ImageRetrieval):
    def __init__(self, db_path, create, k=None, features_nb=None, verbose=True):
        super().__init__(db_path, create, verbose)

        self.k_ = None
        self.codebook_ = None
        self.codebook_path_ = os.path.join(db_path, "codebook.npy")
        self.feature_extractor_ = RootSIFT(features_nb)

        if create:
            self.k_ = k
        else:
            if not os.path.exists(self.codebook_path_):
                raise FileNotFoundError(f"Codebook file not found at {self.codebook_path_}.")

            self.codebook_ = np.load(self.codebook_path_)

            if self.verbose_:
                print(f"Loaded existing codebook from {self.codebook_path_}")

            self.k_ = self.codebook_.shape[0]


    def clear_database(self):
        """Remove all stored vectors embedding"""
        self.chroma_client_.delete_collection(name="vectors_embedding")
        self.database_ = self.chroma_client_.get_or_create_collection(name="vectors_embedding")
        
        if os.path.exists(self.codebook_path_):
            os.remove(self.codebook_path_)
            print(f"Deleted codebook at {self.codebook_path_}")
    

    def fit(self, image_dir):
        """Train VLAD and store vectors in ChromaDB"""
        
        X_train, paths, _ = self.feature_extractor_.extract_features(image_dir)

        if self.codebook_ is not None:
            if self.verbose_:
                print("Using preloaded codebook.")
            return self

        X = np.vstack(X_train)
        
        if self.verbose_:
            print("Clustering started...")

        kmeans = MiniBatchKMeans(n_clusters=self.k_, n_init=20)
        kmeans.fit(X)

        if self.verbose_:
            print("Clustering ended.")

        self.codebook_ = kmeans.cluster_centers_

        # Save codebook for future use
        np.save(self.codebook_path_, self.codebook_)
        if self.verbose_:
            print(f"Saved codebook to {self.codebook_path_}")

        vlad_vectors = self._extract_vlads(X_train)
        self.save_to_chromadb(vlad_vectors, paths)

        return self


    def get_knn(self, image_path, k):
        X = self.feature_extractor_.extract_features_from_image(image_path)

        vlad = self._vlad(X)

        results = self.database_.query(
            query_embeddings=[vlad.tolist()],
            n_results=k
        )

        return np.array(results['ids'][0])


    def predict(self, image_path):
        return self.get_knn(image_path, 1)[0]


    def _extract_vlads(self, X):
        vlads = []
        iterable = tqdm(X, desc="Extracting VLADs") if self.verbose_ else X

        for x in iterable:
            vlads.append(self._vlad(x))

        return np.vstack(vlads)


    def _vlad(self, x):
        if self.codebook_ is None:
            raise ValueError("Codebook is not initialized. Run fit() first.")

        np.seterr(invalid='ignore', divide='ignore')

        d = x.shape[1]
        visual_word_idx = np.linalg.norm(x - self.codebook_[:, None, :], axis=-1).argmin(axis=0)
        vlad_vector = np.zeros((self.k_, d))

        for codebook_idx in range(self.k_):
            vlad_vector[codebook_idx] = np.sum(x[visual_word_idx == codebook_idx] - self.codebook_[codebook_idx], axis=0)

        # Intra-normalization
        vlad_vector /= np.linalg.norm(vlad_vector, axis=1)[:, None]
        np.nan_to_num(vlad_vector, copy=False)

        # L2-normalization
        vlad_vector /= np.linalg.norm(vlad_vector)

        return vlad_vector.flatten()



class NNet(ImageRetrieval):
    def __init__(self, db_path, create, verbose=True):
        super().__init__(db_path, create, verbose)

        self.feature_extractor_ = VGGNet()

    
    def fit(self, image_dir):
        X_train, paths, _ = self.feature_extractor_.extract_features(image_dir)
        self.save_to_chromadb(X_train, paths)


    def get_knn(self, image_path, k):
        X = self.feature_extractor_.extract_features_from_image(image_path)
        results = self.database_.query(
            query_embeddings=[X.tolist()],
            n_results=k
        )

        return np.array(results['ids'][0])


    def clear_database(self):
        """Remove all stored vectors embedding"""
        self.chroma_client_.delete_collection(name="vectors_embedding")
        self.database_ = self.chroma_client_.get_or_create_collection(name="vectors_embedding")
