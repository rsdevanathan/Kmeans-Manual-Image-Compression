import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import time
import configparser
import os

fileDir = os.path.dirname(os.path.realpath('__file__'))
print("Current Directory")
print(fileDir)


start_time = time.time()
def kmediods_initialize_centroids(input_array, k,initialization):
    if initialization == 'bad':
        centroid_indices = range(k)
        centroids = input_array[centroid_indices, :]
    else:
        centroid_indices = np.random.randint(input_array.shape[0], size=(1, k))[0]
        centroids = input_array[centroid_indices, :]
    return centroids


def cluster_assignment(input_array, cluster_centroids, distance_type):
    distance_matrix = cdist(input_array,cluster_centroids,distance_type)
    assigned_clusters = np.argmin(distance_matrix, axis=1)
    return assigned_clusters


def kmediods_centroid_calculation(input_array, assigned_clusters, distance_type):
    cluster_count = np.unique(assigned_clusters).size
    new_centroids = np.zeros((cluster_count, 3))
    for cluster_position,cluster in zip(range(cluster_count),np.unique(assigned_clusters)):
        point_indices, = np.where(np.isclose(assigned_clusters, cluster))
        cluster_points = input_array[point_indices, :]
        cluster_points_sample = cluster_points[np.random.randint(cluster_points.shape[0],
                                                  size=(np.int(cluster_points.shape[0] * 0.1))),:]
        pw_distance_matrix = cdist(cluster_points_sample,cluster_points_sample,distance_type)
        sum_pw_distance_matrix = np.sum(pw_distance_matrix,axis=0)
        min_distance_index = np.argmin(sum_pw_distance_matrix)
        new_centroids[cluster_position] = cluster_points_sample[min_distance_index]
    return new_centroids


def calculate_kmediods(input_array, k, iterations, distance_type,initialization):
    centroids = kmediods_initialize_centroids(input_array, k,initialization)
    previous_assigned_clusters = np.zeros(len(input_array))
    for i in range(0, iterations):
        iter_start_time = time.time()
        assigned_clusters = cluster_assignment(input_array, centroids, distance_type)
        if (assigned_clusters == previous_assigned_clusters).all():
            print("Iteration ", i, " Completed in ", (time.time() - iter_start_time), " Seconds")
            print("Converged")
            break
        previous_assigned_clusters = assigned_clusters
        centroids = kmediods_centroid_calculation(input_array, assigned_clusters, distance_type)
        print("Iteration ", i, " Completed in ",(time.time() - iter_start_time)," Seconds")
    return assigned_clusters,centroids


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('cluster_params.ini')
    iterations = np.int(config['KMediods']['iterations'])
    k = np.int(config['KMediods']['k'])
    distance_type = config['KMediods']['distance']
    input_image_name = config['KMediods']['input']
    initialization = config['KMediods']['initialization']

    print("Running KMediods image compression with ",k," clusters and ",distance_type, "distance and "
          ,initialization," initialization and",iterations,"iterations for image ",input_image_name)

    input_filename = os.path.join(fileDir,"Input/",input_image_name)
    input_image = Image.open(input_filename)
    input_image_array_base = np.array(input_image)
    input_image_array = input_image_array_base.reshape(-1, 3)
    assigned_clusters,centroids = calculate_kmediods(input_image_array, k, iterations, distance_type,initialization)
    new_image_array = centroids[assigned_clusters.astype(int), :].astype(int)

    new_image_array_reshaped = np.reshape(new_image_array, (input_image_array_base.shape[0], input_image_array_base.shape[1],
                           input_image_array_base.shape[2]))

    new_image_array_reshaped = new_image_array_reshaped.astype(np.uint8)
    new_image = Image.fromarray(new_image_array_reshaped)
    output_name = 'KMediods_' +input_image_name.split('.')[0]+"_"+ distance_type + "_" + str(k) + "clusters_"+initialization+"_init"+"_ output"
    image_file_name = os.path.join(fileDir,"Output/",output_name +"_image.bmp")
    class_file_name = os.path.join(fileDir,"Output/",output_name +"_class.txt")
    centroid_file_name = os.path.join(fileDir, "Output/", output_name+"_centroids.txt")
    new_image.save(image_file_name)
    np.savetxt(class_file_name, assigned_clusters, delimiter=',', fmt='%f')
    np.savetxt(centroid_file_name, centroids, delimiter=',', fmt='%f')
    print("Completed KMediods Image Compression")