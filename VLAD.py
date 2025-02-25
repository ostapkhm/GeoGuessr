import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import chromadb
import os


class VLAD:
    def __init__(self, db_path, verbose=True):
        self.db_path_ = db_path
        self.verbose_ = verbose
        self.k_ = None
        
        self.chroma_client_ = chromadb.PersistentClient(path=db_path)
        self.database_ = self.chroma_client_.get_or_create_collection(name="vlad_vectors")

        self.codebook_path_ = os.path.join(db_path, "codebook.npy")
        self.codebook_ = None


    def database_exists(self):
        # Checks if at least one vector exists
        return len(self.database_.peek(1)["ids"]) > 0  
    

    def create(self, k):
        self.k_ = k
        
        os.makedirs(self.db_path_, exist_ok=True)
        chromadb_path = os.path.join(self.db_path_, "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)


    def load(self):
        if not os.path.exists(self.codebook_path_):
            raise FileNotFoundError(f"Codebook file not found at {self.codebook_path_}.")

        self.codebook_ = np.load(self.codebook_path_)

        if self.verbose_:
            print(f"Loaded existing codebook from {self.codebook_path_}")

        self.k_ = self.codebook_.shape[0]
    

    def fit(self, X_train, paths):
        """Train VLAD and store vectors in ChromaDB"""

        if self.codebook_ is not None:
            if self.verbose_:
                print("Using preloaded codebook.")
            return self

        X = np.vstack(X_train)
        
        print("X's shapes")
        print(X.shape)

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
        self._save_to_chromadb(vlad_vectors, paths)

        return self


    def transform(self, X):
        return self._extract_vlads(X)


    def get_knn(self, x, k):
        vlad = self._vlad(x)

        results = self.database_.query(
            query_embeddings=[vlad.tolist()],
            n_results=k
        )

        return np.array(results['ids'][0])


    def predict(self, x):
        return self.get_knn(x, 1)[0]


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


    def _save_to_chromadb(self, vlad_vectors, paths):
        if self.verbose_:
            print("Storing VLAD vectors in database...")
            iterable = tqdm(range(len(vlad_vectors)), desc="Saving VLADs")
        else:
            iterable = range(len(vlad_vectors))

        for i in iterable:
            vlad_vector = vlad_vectors[i]
            self.database_.add(
                embeddings=[vlad_vector.tolist()],
                ids=[paths[i]]
            )


    def clear_database(self):
        """Remove all stored VLAD vectors and delete the codebook"""
        self.chroma_client_.delete_collection(name="vlad_vectors")
        self.database_ = self.chroma_client_.get_or_create_collection(name="vlad_vectors")
        
        if os.path.exists(self.codebook_path_):
            os.remove(self.codebook_path_)
            print(f"Deleted codebook at {self.codebook_path_}")

        print("VLAD database cleared.")
        