import requests
import zipfile
import numpy as np

def download(save_path, chunk_size):
    """
    Downloads and unzips the MovieLens 100k dataset from the project's 
    website.

    Parameters
    ----------
    save_path : str
        file path to download data set to.

    chunk_size : int
        Size of the chunk in bytes. Usually 128.

    Returns
    -------
    None

    """
    url = r"http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def create_rating_matrix(path):
    """
    Loads the data provided and creates a user-item rating matrix.

    Parameters
    ----------
    path : str
        File path to the data to read.

    Returns
    -------
    X : ndarray[n_users, n_iters]
        An array of floats. Each element [i,j] represents the rating
        that user i gave item j. 

    """
    df = np.loadtxt(path, delimiter='\t')
    #n_users = int(np.max(df[:, 0]))
    #n_items = int(np.max(df[:, 1]))
    n_users = 943
    n_items = 1682
    X = np.zeros((n_users, n_items))

    for i in range(df.shape[0]):
        user = int(df[i, 0] - 1)
        item = int(df[i, 1] - 1)
        rating = df[i, 2]
        X[user, item] = rating

    return X

# if __name__ == "__main__":
#     download(save_path="../../data/ml-latest-small.zip", chunk_size=128)
#     with zipfile.ZipFile("../../data/ml-latest-small.zip") as zip_ref:
#         zip_ref.extractall("../../data/")
