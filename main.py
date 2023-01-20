import json
import logging
import math
import os
import random as rd
from os import path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn import metrics
from sklearn import neighbors
from sklearn import svm as Svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.autograph.set_verbosity(2)
from keras.applications.vgg16 import VGG16, preprocess_input as VGG16_pre_process
from keras.utils import image_utils

RES_FOLDER = 'res/'
PROCESSED_DATASET_PATH = RES_FOLDER + 'blood_cell_processed.csv'
IMAGES_DATASET_BASE_PATH = RES_FOLDER + 'blood_cell_cancer_imgs/'
VGG16_WEIGHTS_PATH = RES_FOLDER + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
REPORT_PATH = 'report.json'
FORCE_PROCESSED_DATASET_GEN = False
PLOT = True
BLOCK_PLOTS = True
KNEE_SENSITIVITY = 10
KERNEL_PCA = True
KERNEL_PCA_KERNEL = 'poly'
NORMALIZE_FEATURES = True
AMOUNT_RANDOM_FEATURES_TO_VISUALIZE = 5
BALANCE_TWO_CLASS_DATASET = True
BALANCE_TWO_CLASS_DATASET_RATIO = 0.6
DATASET_TRAIN_RATIO_SPLIT = 0.7
RUN_SVM = True
SVM_KERNEL = 'rbf'
SVM_REGULARIZATION = 1
SVM_GAMMA = 'scale'
RUN_NEAREST_NEIGHBOURS = True
NEAREST_NEIGHBOURS_NEIGHBOURS = 10
NEAREST_NEIGHBOURS_WEIGHTS = 'uniform'
RUN_RANDOM_FOREST = True
RANDOM_FOREST_ESTIMATORS = 200


def getAllFiles(dir_path):
    return [y for y in [path.join(dir_path, x) for x in os.listdir(dir_path)] if path.isfile(y)]


def getAllFolders(dir_path):
    return [y for y in [path.join(dir_path, x) for x in os.listdir(dir_path)] if path.isdir(y)]


def extractFeaturesUsingDeepConvolutionalNetwork(dcnn, img_path):
    dcnn_input_img_size = (224, 224)
    img = np.array(image_utils.load_img(img_path, target_size=dcnn_input_img_size))
    img = np.expand_dims(img, axis=0)  # encapsulate on another np.array. (#samples,x,y,RBG)
    processed_image = VGG16_pre_process(img)
    features = dcnn(processed_image)[0]
    features = features.numpy().flatten()
    return features


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_results_report(rep):
    try:
        with open(REPORT_PATH, 'w') as f:
            json.dump(rep, f, cls=NumpyEncoder, indent=2)
    except Exception as e:
        print(e)


def pretty_repr_pandas_df(df, padding=0):
    with pd.option_context("display.max_rows", 100):
        with pd.option_context("display.max_columns", 100):
            with pd.option_context("display.expand_frame_repr", False):
                cm_str = repr(df).split('\n')
                for l, line in enumerate(cm_str):
                    if l > 0:
                        cm_str[l] = f'{" " * padding}{line}'
                cm_str = '\n'.join(cm_str)
                return f"\"{cm_str}\"  "


def manual_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def mean(arr):
    return sum(arr) / len(arr)


# bit of data understanding
if path.exists(IMAGES_DATASET_BASE_PATH) and PLOT:
    sub_dirs = getAllFolders(IMAGES_DATASET_BASE_PATH)
    labels = [path.basename(sub_dir) for sub_dir in sub_dirs]
    for sub_dir, label in zip(sub_dirs, labels):
        images = getAllFiles(sub_dir)
        rd.shuffle(images)
        img = mpimg.imread(images[0])
        ax = plt.imshow(img)
        plt.yticks([])
        plt.xticks([])
        plt.subplots_adjust(bottom=0, top=0.93, left=0, right=1)
        plt.title(f'Example of {label} blood cell')
        plt.show(block=BLOCK_PLOTS)

# process images into dataset
if FORCE_PROCESSED_DATASET_GEN or not path.exists(PROCESSED_DATASET_PATH):
    if path.exists(VGG16_WEIGHTS_PATH):
        dcnn = VGG16(weights=VGG16_WEIGHTS_PATH)
    else:
        dcnn = VGG16()
    sub_dirs = getAllFolders(IMAGES_DATASET_BASE_PATH)
    labels = [path.basename(sub_dir) for sub_dir in sub_dirs]
    dataset = []
    for sub_dir, label in zip(sub_dirs, labels):
        images = getAllFiles(sub_dir)
        for image in images:
            entry = np.append(extractFeaturesUsingDeepConvolutionalNetwork(dcnn, image), label)
            dataset.append(entry)

    df = pd.DataFrame(dataset, columns=[f'feat_{i}' for i in range(1000)] + ['label'])
    df.to_csv(PROCESSED_DATASET_PATH, encoding='utf-8', index=False)

# data loading
df = pd.read_csv(PROCESSED_DATASET_PATH)
df_features = df[df.columns.difference(['label'])]
df_labels = df['label']

# data preparation + bit understanding

if NORMALIZE_FEATURES:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_features = pd.DataFrame(scaler.fit_transform(df_features))
if KERNEL_PCA:
    kpca_transform = KernelPCA(kernel=KERNEL_PCA_KERNEL).fit_transform(df_features)
    explained_variance = np.var(kpca_transform, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    explained_variance_ratio_sum = np.cumsum(explained_variance_ratio)
else:
    pca = PCA().fit(df_features)
    explained_variance_ratio_sum = np.cumsum(pca.explained_variance_ratio_)
optimum_n_components = KneeLocator(list(range(len(explained_variance_ratio_sum))), explained_variance_ratio_sum,
                                   curve='concave',
                                   direction='increasing', S=KNEE_SENSITIVITY)
print(
    f'Optimum amount of {"Kernel PCA" if KERNEL_PCA else "PCA"} components to contain {round(optimum_n_components.knee_y * 100, 3)}% of '
    f'the variance: {optimum_n_components.knee}')
if PLOT:
    plt.plot(explained_variance_ratio_sum, label='_nolegend_', zorder=1)
    plt.scatter([optimum_n_components.knee],
                [optimum_n_components.knee_y], color='red', label=f'optimum # components ({optimum_n_components.knee})',
                zorder=2)
    plt.title('Cumulative explained variance per # of components')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.tight_layout()
    plt.legend()
    plt.show(block=BLOCK_PLOTS)

sequentializer = {}
label_names = []
labels_2_classes = []
labels_4_classes = []
for label in df_labels:
    if label.lower() == 'benign':
        labels_2_classes.append(0)
    else:
        labels_2_classes.append(1)
    if label not in sequentializer:
        sequentializer[label] = len(sequentializer)
        label_names.append(label)
    labels_4_classes.append(sequentializer[label])

# data understanding

df_sample = df_features.sample(n=AMOUNT_RANDOM_FEATURES_TO_VISUALIZE, axis='columns')
for column in df_sample.columns:
    min_feat = df_features[column].min()
    max_feat = df_features[column].max()
    print(f'Feature `{column}`, minimum value: {min_feat}, maximumum value: {max_feat}')
    if PLOT:
        amount_bins = 10
        bin_width = (math.ceil(max_feat) - math.floor(min_feat)) / amount_bins
        bins = np.arange(math.floor(min_feat), math.ceil(max_feat) + bin_width, bin_width)
        ax = df_sample[column].plot.hist(bins=bins, title=f'Feature `{column}` histogram', rwidth=0.5,
                                         figsize=(12, 4.8))
        ax.set_xlabel(f'Feature `{column}`')
        plt.tight_layout()
        plt.xticks(bins)
        plt.show(block=BLOCK_PLOTS)

for label in set(labels_2_classes):
    occurrence = labels_2_classes.count(label)
    print(f'Occurrence of label {label} in 2 class labels: {occurrence}, '
          f'{round((occurrence / len(labels_2_classes) * 100), 2)}')
print()
for label in set(labels_4_classes):
    occurrence = labels_4_classes.count(label)
    print(f'Occurrence of label {label} in 4 class labels: {occurrence}, '
          f'{round((occurrence / len(labels_4_classes) * 100), 2)}')
print()

# data processing - balancing
indexes_for_2_classes = None
if BALANCE_TWO_CLASS_DATASET:
    indexes_for_2_classes = [[], []]
    for i, y in enumerate(labels_2_classes):
        indexes_for_2_classes[y].append(i)
    rd.shuffle(indexes_for_2_classes[1])  # shuffle
    indexes_for_2_classes[1] = indexes_for_2_classes[1][
                               :int(len(indexes_for_2_classes[0]) * (1 + BALANCE_TWO_CLASS_DATASET_RATIO - .5))]
    indexes_for_2_classes = indexes_for_2_classes[0] + indexes_for_2_classes[1]
    indexes_for_2_classes = set(indexes_for_2_classes)

# dimensionality reduction
if KERNEL_PCA:
    kpca = KernelPCA(kernel=KERNEL_PCA_KERNEL, n_components=optimum_n_components.knee)
    reduced_features_4 = kpca.fit_transform(df_features)
else:
    pca = PCA(n_components=optimum_n_components.knee)
    reduced_features_4 = pca.fit_transform(df_features)

if NORMALIZE_FEATURES:  # normalizing again since PCA messes with scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    reduced_features_4 = scaler.fit_transform(reduced_features_4)

# data understanding
df_reduced = pd.DataFrame(reduced_features_4)
df_sample = df_reduced.sample(n=AMOUNT_RANDOM_FEATURES_TO_VISUALIZE, axis='columns')
for column in df_sample.columns:
    min_feat = df_reduced[column].min()
    max_feat = df_reduced[column].max()
    print(f'New feature `{column}`, minimum value: {min_feat}, maximumum value: {max_feat}')
    if PLOT:
        amount_bins = 10
        bin_width = (math.ceil(max_feat) - math.floor(min_feat)) / amount_bins
        bins = np.arange(math.floor(min_feat), math.ceil(max_feat) + bin_width, bin_width)
        ax = df_sample[column].plot.hist(bins=bins, title=f'New feature `{column}` histogram', rwidth=0.5,
                                         figsize=(12, 4.8))
        ax.set_xlabel(f'New feature `{column}`')
        plt.tight_layout()
        plt.xticks(bins)
        plt.show(block=BLOCK_PLOTS)

reduced_features_2 = []
labels_2_classes_new = []
for i, x in enumerate(reduced_features_4):
    if indexes_for_2_classes is None or i in indexes_for_2_classes:
        reduced_features_2.append(np.copy(x))
        labels_2_classes_new.append(labels_2_classes[i])
labels_2_classes = labels_2_classes_new

# data understanding
print('After balancing...')
for label in set(labels_2_classes):
    occurrence = labels_2_classes.count(label)
    print(f'Occurrence of label {label} in 2 class labels: {occurrence}, '
          f'{round((occurrence / len(labels_2_classes) * 100), 2)}')
print()
for label in set(labels_4_classes):
    occurrence = labels_4_classes.count(label)
    print(f'Occurrence of label {label} in 4 class labels: {occurrence}, '
          f'{round((occurrence / len(labels_4_classes) * 100), 2)}')
print()

# ohe=OneHotEncoder()
# labels_4_classes=ohe.fit_transform([[el] for el in labels_4_classes]).toarray()

# data preparation - train test split
x_2_train, x_2_test, y_2_train, y_2_test = train_test_split(reduced_features_2, labels_2_classes,
                                                            train_size=DATASET_TRAIN_RATIO_SPLIT,
                                                            shuffle=True)
x_4_train, x_4_test, y_4_train, y_4_test = train_test_split(reduced_features_4, labels_4_classes,
                                                            train_size=DATASET_TRAIN_RATIO_SPLIT,
                                                            shuffle=True)

print()

curve_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
report = []
if RUN_SVM:
    svm = Svm.SVC(kernel=SVM_KERNEL, C=SVM_REGULARIZATION, gamma=SVM_GAMMA)
    svm.fit(x_2_train, y_2_train)
    y_2_pred = svm.predict(x_2_test)
    acc = metrics.accuracy_score(y_2_test, y_2_pred)
    pre = metrics.precision_score(y_2_test, y_2_pred)
    rec = metrics.recall_score(y_2_test, y_2_pred)
    f1 = metrics.f1_score(y_2_test, y_2_pred)
    mf1 = manual_f1(pre, rec)
    roc_auc = metrics.roc_auc_score(y_2_test, y_2_pred)
    cm = metrics.confusion_matrix(y_2_test, y_2_pred)
    classification_result = {
        'method': 'svm',
        'type': 'single label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'rec': rec,
            'f1': f1,
            'manual_f1': mf1,
            'roc_auc': roc_auc,
            'cm': {
                'val': [str(el) for el in cm],
                'legend': ['[TP FP]', '[FP TN]']
            },
        }
    }
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder))
    print()
    report.append(classification_result)
    if PLOT:
        false_pos, true_pos, thresholds = metrics.roc_curve(y_2_test, y_2_pred)
        plt.plot(false_pos, true_pos)
        plt.title('ROC Curve - SVM - Single Label')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

        precisions, recalls, thresholds = metrics.precision_recall_curve(y_2_test, y_2_pred)
        plt.plot(recalls, precisions)
        plt.title('PR Curve - SVM - Single Label')
        plt.xlabel('Recall (Coverage)')
        plt.ylabel('Precision (Efficiency)')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

    svm = Svm.SVC(kernel=SVM_KERNEL, C=SVM_REGULARIZATION, gamma=SVM_GAMMA, decision_function_shape='ovo')
    svm.fit(x_4_train, y_4_train)
    y_4_pred = svm.predict(x_4_test)
    acc = metrics.accuracy_score(y_4_test, y_4_pred)
    pre = metrics.precision_score(y_4_test, y_4_pred, average=None)
    rec = metrics.recall_score(y_4_test, y_4_pred, average=None)
    f1 = metrics.f1_score(y_4_test, y_4_pred, average=None)
    cm = metrics.confusion_matrix(y_4_test, y_4_pred)
    classification_result = {
        'method': 'svm',
        'type': 'multi label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'pre_avg': mean(pre),
            'rec': rec,
            'rec_avg': mean(rec),
            'f1': f1,
            'f1_avg': mean(f1),
            'manual_f1_avg': manual_f1(mean(pre), mean(rec)),
            'cm': "$REPLACE_ME",  # rows = actual, columns = pred
        }
    }
    cm_str = pretty_repr_pandas_df(pd.DataFrame(cm, columns=label_names, index=label_names), 11)
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder).replace('"$REPLACE_ME"', cm_str))
    classification_result['metrics']['cm'] = cm
    print()
    report.append(classification_result)
print()
print()

if RUN_NEAREST_NEIGHBOURS:
    knn = neighbors.KNeighborsClassifier(n_neighbors=NEAREST_NEIGHBOURS_NEIGHBOURS, weights=NEAREST_NEIGHBOURS_WEIGHTS)
    knn.fit(x_2_train, y_2_train)
    y_2_pred = knn.predict(x_2_test)
    acc = metrics.accuracy_score(y_2_test, y_2_pred)
    pre = metrics.precision_score(y_2_test, y_2_pred)
    rec = metrics.recall_score(y_2_test, y_2_pred)
    f1 = metrics.f1_score(y_2_test, y_2_pred)
    mf1 = manual_f1(pre, rec)
    roc_auc = metrics.roc_auc_score(y_2_test, y_2_pred)
    cm = metrics.confusion_matrix(y_2_test, y_2_pred)
    classification_result = {
        'method': 'nearest neighbours',
        'type': 'single label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'rec': rec,
            'f1': f1,
            'manual_f1': mf1,
            'roc_auc': roc_auc,
            'cm': {
                'val': [str(el) for el in cm],
                'legend': ['[TP FP]', '[FP TN]']
            },
        }
    }
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder))
    print()
    report.append(classification_result)
    if PLOT:
        false_pos, true_pos, thresholds = metrics.roc_curve(y_2_test, y_2_pred)
        plt.plot(false_pos, true_pos)
        plt.title('ROC Curve - Nearest Neighbours - Single Label')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

        precisions, recalls, thresholds = metrics.precision_recall_curve(y_2_test, y_2_pred)
        plt.plot(recalls, precisions)
        plt.title('PR Curve - Nearest Neighbours - Single Label')
        plt.xlabel('Recall (Coverage)')
        plt.ylabel('Precision (Efficiency)')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

    knn = neighbors.KNeighborsClassifier(n_neighbors=NEAREST_NEIGHBOURS_NEIGHBOURS, weights=NEAREST_NEIGHBOURS_WEIGHTS)
    knn.fit(x_4_train, y_4_train)
    y_4_pred = knn.predict(x_4_test)
    acc = metrics.accuracy_score(y_4_test, y_4_pred)
    pre = metrics.precision_score(y_4_test, y_4_pred, average=None)
    rec = metrics.recall_score(y_4_test, y_4_pred, average=None)
    f1 = metrics.f1_score(y_4_test, y_4_pred, average=None)
    cm = metrics.confusion_matrix(y_4_test, y_4_pred)
    classification_result = {
        'method': 'nearest neighbours',
        'type': 'multi label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'pre_avg': mean(pre),
            'rec': rec,
            'rec_avg': mean(rec),
            'f1': f1,
            'f1_avg': mean(f1),
            'manual_f1_avg': manual_f1(mean(pre), mean(rec)),
            'cm': "$REPLACE_ME",  # rows = actual, columns = pred
        }
    }
    cm_str = pretty_repr_pandas_df(pd.DataFrame(cm, columns=label_names, index=label_names), 11)
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder).replace('"$REPLACE_ME"', cm_str))
    classification_result['metrics']['cm'] = cm
    print()
    report.append(classification_result)
print()
print()

if RUN_RANDOM_FOREST:
    # ohe=OneHotEncoder()
    # y_4_train=ohe.fit_transform([[el] for el in y_4_train]).toarray()
    # y_4_test=ohe.transform([[el] for el in y_4_test]).toarray()

    rf = RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS)
    rf.fit(x_2_train, y_2_train)
    y_2_pred = rf.predict(x_2_test)
    acc = metrics.accuracy_score(y_2_test, y_2_pred)
    pre = metrics.precision_score(y_2_test, y_2_pred)
    rec = metrics.recall_score(y_2_test, y_2_pred)
    f1 = metrics.f1_score(y_2_test, y_2_pred)
    mf1 = manual_f1(pre, rec)
    roc_auc = metrics.roc_auc_score(y_2_test, y_2_pred)
    cm = metrics.confusion_matrix(y_2_test, y_2_pred)
    classification_result = {
        'method': 'random forest',
        'type': 'single label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'rec': rec,
            'f1': f1,
            'manual_f1': mf1,
            'roc_auc': roc_auc,
            'cm': {
                'val': [str(el) for el in cm],
                'legend': ['[TP FP]', '[FP TN]']
            },
        }
    }
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder))
    print()
    report.append(classification_result)
    if PLOT:
        false_pos, true_pos, thresholds = metrics.roc_curve(y_2_test, y_2_pred)
        plt.plot(false_pos, true_pos)
        plt.title('ROC Curve - Random Forest - Single Label')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

        precisions, recalls, thresholds = metrics.precision_recall_curve(y_2_test, y_2_pred)
        plt.plot(recalls, precisions)
        plt.title('PR Curve - Random Forest - Single Label')
        plt.xlabel('Recall (Coverage)')
        plt.ylabel('Precision (Efficiency)')
        plt.xticks(curve_ticks)
        plt.yticks(curve_ticks)
        plt.tight_layout()
        plt.show(block=BLOCK_PLOTS)

    rf = RandomForestClassifier(n_estimators=RANDOM_FOREST_ESTIMATORS)
    rf.fit(x_4_train, y_4_train)
    y_4_pred = rf.predict(x_4_test)
    acc = metrics.accuracy_score(y_4_test, y_4_pred)
    pre = metrics.precision_score(y_4_test, y_4_pred, average=None)
    rec = metrics.recall_score(y_4_test, y_4_pred, average=None)
    f1 = metrics.f1_score(y_4_test, y_4_pred, average=None)
    cm = metrics.confusion_matrix(y_4_test, y_4_pred)
    classification_result = {
        'method': 'random forest',
        'type': 'multi label',
        'metrics': {
            'acc': acc,
            'pre': pre,
            'pre_avg': mean(pre),
            'rec': rec,
            'rec_avg': mean(rec),
            'f1': f1,
            'f1_avg': mean(f1),
            'manual_f1_avg': manual_f1(mean(pre), mean(rec)),
            'cm': "$REPLACE_ME",  # rows = actual, columns = pred
        }
    }
    cm_str = pretty_repr_pandas_df(pd.DataFrame(cm, columns=label_names, index=label_names), 11)
    print(json.dumps(classification_result, indent=2, cls=NumpyEncoder).replace('"$REPLACE_ME"', cm_str))
    classification_result['metrics']['cm'] = cm
    print()
    report.append(classification_result)
print()
print()

save_results_report(report)

print()

# last plot
if PLOT and not BLOCK_PLOTS:
    plt.show()
