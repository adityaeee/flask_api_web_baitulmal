# import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
# import json
# from metode import AHCK

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, methods="*")

@app.route("/clusterDataGolongan", methods=["POST"])
def cluster():
    try:
        data_awal = request.get_json()
        dataset = np.asarray(data_awal)
        length_dataset = len(dataset)
        column_0 = dataset[:, 0]
        
        # print(length_dataset)
        if length_dataset == 1:
            column_0 = np.asarray(column_0)    
            return jsonify({"res": column_0.tolist()})
        
        else:
        
            data = np.delete(dataset, 0, axis=1)

            agg_cluster = AgglomerativeClustering(
                n_clusters=1)
            agg_labels = agg_cluster.fit_predict(data)

            centers = []
            for i in range(1):
                cluster_points = data[agg_labels == i]
                centers.append(np.min(cluster_points, axis=0))
            centers = np.array(centers)

            kmeans = KMeans(n_clusters=1, init=centers, n_init=1)
            kmeans_labels = kmeans.fit_predict(data)
            centroid = kmeans.cluster_centers_

            jarak_data = np.linalg.norm(data - centroid[kmeans.labels_], axis=1)

            sorted_data = np.argsort(jarak_data)
            
            data = np.insert(data, 0, column_0, axis=1)
            
            result =[]
            for i in sorted_data:
                result.append(data[i][0])
            
            
            result = np.asarray(result)    
            print(result)
            return jsonify({"res": result.tolist()})

    except FileNotFoundError:
        return jsonify({"error": "Model tidak ditemukan"}), 500
    except KeyError:
        return jsonify({"error": "Data input tidak ditemukan"}), 400
    except Exception as e:
        return jsonify({"error": f"Gagal melakukan clustering: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    



# @app.route("/clusterDataGolongan", methods=["POST"])
# def predictSU():
#     try:
#         data = request.get_json()
#         input = data["data"]
#         print("THE DATA:::::", input)
#         model = tf.keras.models.load_model("./modelSU.h5")

#         result = model.predict([input])
#         print("RESULT:::::", result)
#         return jsonify({"res": result.tolist()})

#     except FileNotFoundError:
#         return jsonify({"error": "Model tidak ditemukan"}), 500
#     except KeyError:
#         return jsonify({"error": "Data input tidak ditemukan"}), 400
#     except Exception as e:
#         return jsonify({"error": f"Gagal melakukan prediksi: {str(e)}"}), 500