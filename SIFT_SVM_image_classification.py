# Importing system libraries
import os
import glob
import time
import gc
import multiprocessing as mp

# Importing external libraries
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the extract features file
from extract_features import get_descriptor_and_label

# Defining  configuration variables
classes = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial',
           'DenseResidential', 'Desert', 'Farmland',
           'Forest', 'Industrial', 'Meadow', 'MediumResidential', 'Mountain', 'Park', 'Parking', 'Playground', 'Pond',
           'Port', 'RailwayStation',
           'Resort', 'River', 'School', 'SparseResidential', 'Square', 'Stadium', 'StorageTanks', 'Viaduct']

input_path_folder = os.path.join(os.getcwd(), 'AID')  # folder containing the input dataset
r_s = 42  # random state
test_split = 0.2  # percentage of the test split of the entire dataset
no_of_processes = 6  # used for parallel processing to speed up execution, value is due to me having a hexa-core CPU
max_tasks_per_child = 10  # max number of simultaneous tasks per process
batch_size_for_kmeans = no_of_processes * 256  # 256 for each CPU core, per sklearn documentation
num_cluster = 70  # no of clusters for feature extraction using BoW


def extract_sift_features(input_path):
    """ Extracts image labels and descriptors

        :param input_path: Path to the directory containing the images
        :type input_path: str or :class:`pathlib.Path`

        :return: A tuple containing the image SIFT descriptors and the image labels
        :rtype: tuple
    """
    im_labels = []
    im_descriptors = []

    for target_class in classes:
        folder_path = os.path.join(input_path, target_class)

        image_paths = [f for f in glob.glob(folder_path + "/*.jpg")]
        print(f'Processing the {len(image_paths)} images in the {target_class} class.')

        with mp.Pool(processes=no_of_processes, maxtasksperchild=max_tasks_per_child) as pool:
            results = [pool.apply_async(get_descriptor_and_label, args=(image_path,)) for image_path in image_paths]
            descriptors = [res.get() for res in results]

        labels = [classes.index(target_class)] * len(descriptors)

        im_descriptors += descriptors
        im_labels += labels
        del labels, descriptors, image_paths
        gc.collect()

    return im_descriptors, im_labels


def kmeans_bow(all_descriptors, num_cluster):
    """ Creates the Bag of Visual Words on the features extracted from the images with SIFT

        :param all_descriptors: All descriptors taken from the images, size [n*Dix128]
        :type all_descriptors: list or :class:`numpy.ndarray`
        :param num_cluster: Number of clusters to be created
        :type num_cluster: int

        :return: A tuple containing the clusters of the centers and their inertia
        :rtype: tuple
    """
    bow_dict = []

    mbk = MiniBatchKMeans(n_clusters=num_cluster, batch_size=batch_size_for_kmeans)
    mbk.fit(all_descriptors)

    bow_dict = mbk.cluster_centers_
    bow_inertia = mbk.inertia_

    return bow_dict, bow_inertia


def create_feature_bow(image_descriptors, BoW, num_cluster):
    """ Computes the distances to the cluster centers, and selects the closest center for each gradient

        :param image_descriptors: Image descriptors, size [nxDix128]
        :type image_descriptors: list
        :param BoW: Precomputed clusters
        :type BoW: list
        :param num_cluster: Number of clusters used for BoW
        :type num_cluster: int

        :return: The new features, size [nxnum_cluster]
        :rtype: list
    """
    features = []

    for i in range(len(image_descriptors)):
        descriptor_features = np.array([0] * num_cluster)

        if image_descriptors[i] is not None:
            distance = cdist(image_descriptors[i], BoW)

            argmin = np.argmin(distance, axis=1)

            for j in argmin:
                descriptor_features[j] += 1
        features.append(descriptor_features)

    return features


if __name__ == '__main__':
    # Feature extraction using SIFT image descriptors and retrieving image labels
    start_time = time.time()
    image_descriptors, image_labels = extract_sift_features(input_path=input_path_folder)
    stop_time = time.time()
    image_processing_time = stop_time - start_time
    print("Image processing time: " + str(image_processing_time) + " seconds.")

    # Splitting into the train and test datasets
    X_train, X_test, Y_train, Y_test = train_test_split(image_descriptors, image_labels, test_size=test_split,
                                                        random_state=r_s)
    del image_descriptors, image_labels
    gc.collect()

    # Creating flattened image descriptor array for BoVW creation
    X_train_flattened = []
    for descriptor in X_train:
        if descriptor is not None:
            for des in descriptor:
                X_train_flattened.append(des)

    # Creation of BoVW using only the train image descriptors
    start_time = time.time()
    BoW, inertia = kmeans_bow(X_train_flattened, num_cluster)
    del X_train_flattened
    gc.collect()
    stop_time = time.time()
    bow_processing_time = stop_time - start_time
    print("Time to create BoW: " + str(bow_processing_time) + " seconds.")

    # Creating the feature list based on the created BoW
    start_time = time.time()
    X_train_features = create_feature_bow(X_train, BoW, num_cluster)
    del X_train
    gc.collect()
    X_test_features = create_feature_bow(X_test, BoW, num_cluster)
    del BoW, X_test
    gc.collect()
    stop_time = time.time()
    applying_bow_time = stop_time - start_time
    print("Time to apply BoW on our dataset: " + str(applying_bow_time) + " seconds.")

    # Applying Linear Discriminant Analysis for dimensionality reduction
    lda = LDA()
    lda_x_train = lda.fit_transform(X_train_features, Y_train)
    lda_x_test = lda.transform(X_test_features)

    # Using GridSearchCV for hyperparameter tuning
    parameters = {'C': [1, 10, 20, 30, 40, 50],
                  'gamma': [0.001, 0.01, 0.1, 1, 5, 10, 'scale', 'auto'],
                  'kernel': ('sigmoid', 'rbf')}

    grid_model = GridSearchCV(
        estimator=SVC(random_state=r_s),
        param_grid=parameters,
        cv=10,
        n_jobs=no_of_processes
    )
    grid_model.fit(lda_x_train, Y_train)

    # Predicting the fitted model on the test dataset
    y_pred = grid_model.predict(lda_x_test)

    # Accuracy score
    acc_score = accuracy_score(y_pred, Y_test)
    print(f"The model is {acc_score}% accurate")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_pred, Y_test)
    # print("Confusion matrix")

    # Classification report
    class_report = classification_report(Y_test, y_pred, target_names=classes)
    print("Classification report")
    print(class_report)

    # Plotting the confusion matrix
    plt.figure()
    sns.heatmap(conf_matrix,
                xticklabels=classes,
                yticklabels=classes,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.savefig(os.path.join(os.getcwd(), 'output', 'confusion_matrix.png'))
