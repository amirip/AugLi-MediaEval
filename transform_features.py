import csv
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from os.path import join, splitext
from tqdm import tqdm

FEATURE_PATH='MEDIA-EVAL19/features/DeepSpectrum/5s/'
LABEL_PATH='MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/'

def transform_feature_csv(feature_csv, label_tsv):
    filenames = []
    classes = []
    with open(label_tsv) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        label_index = header.index('TAGS')
        path_index = header.index('PATH')
        for line in tqdm(reader):
            filenames.append(line[path_index])
            classes.append(tuple(line[label_index:]))
    
        label_binarizer = MultiLabelBinarizer()
        label_binarizer.fit(classes)
    classes = label_binarizer.transform(classes)
    label_map = {fn: c for fn, c in zip(filenames, classes)}
    df = pd.read_csv(feature_csv)
    df = df.groupby(['name'])
    names = []
    features = []
    labels = []
    for group in filenames:
        names.append(group)
        labels.append(label_map[group])
        features.append(df.get_group(group).values[:, 2:])
    names = np.array(names)
    labels = np.array(labels)
    features = np.array(features, dtype=np.float32)
    tags = np.array(label_binarizer.classes_, dtype=str)
    print(names.shape, names.dtype, labels.shape, labels.dtype, features.shape, features.dtype, tags.shape)
    np.savez_compressed(f'{splitext(feature_csv)[0]}.npz', names=names, y=labels, X=features, tags=tags)
    
if __name__=='__main__':
    FEATURE_PATH = sys.argv[1]
    LABEL_PATH = sys.argv[2]
    transform_feature_csv(join(FEATURE_PATH, 'train.csv'), join(LABEL_PATH, 'autotagging_moodtheme-train.tsv'))
    transform_feature_csv(join(FEATURE_PATH, 'validation.csv'), join(LABEL_PATH, 'autotagging_moodtheme-validation.tsv'))
    transform_feature_csv(join(FEATURE_PATH, 'test.csv'), join(LABEL_PATH, 'autotagging_moodtheme-test.tsv'))