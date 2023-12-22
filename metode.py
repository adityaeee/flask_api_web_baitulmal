# import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

dataset = np.array([[111, 1, 2, 4, 2, 1, 1, 1],
                        [112, 1, 4, 1, 2, 1, 1, 2],
                        [113, 2, 6, 2, 2, 1, 1, 2],
                        [114, 1, 1, 3, 3, 2, 1, 1],
                        [115, 2, 3, 2, 1, 2, 1, 2],
                        [116, 2, 5, 1, 1, 1, 2, 2],
                        [117, 2, 1, 1, 3, 2, 1, 1],
                        [118, 1, 7, 4, 3, 2, 2, 1],
                        [119, 1, 6, 2, 3, 1, 2, 2],
                        [121, 1, 3, 3, 1, 2, 1, 2]])


def AHCK(dataset):
    # fungsi untuk mengetahui jumlah baris pada array
    # dataset_row = dataset.shape[0]
    
    # fungsi untuk mengetahui jumlah kolom pada array
    # dataset_column = dataset.shape[1]
    
    # fungsi untuk mengambil kolom ke 0 dari dataset
    column_0 = dataset[:, 0]

    # fungsi untuk menghapus kolom ke 0 dari dataset
    data = np.delete(dataset, 0, axis=1)

    # fungsi untuk menjalankan AHC
    agg_cluster = AgglomerativeClustering(
        n_clusters=1, affinity='euclidean', linkage='single')
    agg_labels = agg_cluster.fit_predict(data)

    # menggunakan hasil AHC sebagai inisialisasi centroid awal KMeans
    centers = []
    for i in range(1):
        cluster_points = data[agg_labels == i]
        centers.append(np.min(cluster_points, axis=0))
    centers = np.array(centers)

    # menjalankan KMeans dengan inisialisasi dari AHC
    kmeans = KMeans(n_clusters=1, init=centers)
    kmeans_labels = kmeans.fit_predict(data)
    centroid = kmeans.cluster_centers_

    # menghitung jarak setiap data ke titik centroid
    jarak_data = np.linalg.norm(data - centroid[kmeans.labels_], axis=1)

    # mengurutkan data berdasarkan jarak
    sorted_data = np.argsort(jarak_data)
    
    data = np.insert(data, 0, column_0, axis=1)
    
    # menampilkan urutan data berdasarkan jarak terdekat
    # print("urutan data berdasarkan jarak terdekat ke titik centroid:")
    # for i in sorted_data:
    #     print(
    #         f"Data {data[i][0]}:{data[i]} (Jarak: {jarak_data[i]:.2f})")

    return data

# hasil = AHCK(dataset)
# print(hasil)

