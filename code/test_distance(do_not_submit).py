import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pandas as pd
import os
base_path = "dataset/cars_train/cars_train/"
def test_vicinanza_stili(df, query_index, feature_col='features', n_neighbors=5):
    """
    Mostra l'immagine scelta e le immagini matematicamente più vicine.
    """
    # 1. Preparazione matriciale
    X = np.vstack(df[feature_col].values)
    '''
    # Normalizzazione (fondamentale per CLIP e cosine similarity)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    '''
    # 2. Calcolo similitudine
    query_vec = X[query_index].reshape(1, -1)
    # Usiamo il prodotto scalare su vettori normalizzati (uguale a cosine similarity)
    scores = np.dot(X, query_vec.T).flatten()
    
    # 3. Trova i top indici (escludendo la query stessa)
    indices = np.argsort(scores)[::-1][1:n_neighbors+1]
    
    # 4. Plotting
    fig, axes = plt.subplots(1, n_neighbors + 1, figsize=(20, 5))
    
    # Immagine Query
    axes[0].imshow(Image.open(base_path+df.iloc[query_index]['file_name']))
    axes[0].set_title("QUERY (Original)")
    axes[0].axis('off')
    
    for i, idx in enumerate(indices):
        # 1. Recupera il nome del file e convertilo in stringa standard
        file_name = str(df.iloc[idx]['file_name'])
        
        # 2. Costruisci il percorso completo correttamente
        full_path = os.path.join(base_path, file_name)
        
        try:
            # 3. Apri l'immagine e passala a imshow
            img = Image.open(full_path)
            axes[i+1].imshow(img)
            axes[i+1].set_title(f"Vicino {i+1}\nSim: {scores[idx]:.3f}")
        except FileNotFoundError:
            axes[i+1].set_title(f"Errore:\nFile non trovato")
            print(f"Non trovo il file: {full_path}")
            
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Provalo con un indice a caso
df = pd.read_pickle('data/standard_CLIP_feature.pkl')
test_vicinanza_stili(df, query_index=105)