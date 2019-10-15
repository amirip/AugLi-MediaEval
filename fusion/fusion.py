import numpy as np
import pandas as pd
from sklearn import metrics
from pathlib import Path
from os.path import join

MEDIA_EVAL_PATH = './MEDIA-EVAL19'


def fuse():
    predictions = np.array(list(map(np.load, Path('./').glob('**/predictions.npy'))))
    mean_predictions = np.mean(predictions, axis=0)
    decisions = np.array(list(map(np.load, Path('./').glob('**/decisions.npy'))))
    decision_sum = np.sum(decisions, axis=0)
    decision_sum[decision_sum <= int(decisions.shape[0]/2)] = 0
    decision_sum[decision_sum > int(decisions.shape[0]/2)] = 1 
    return mean_predictions, decision_sum
    


def evaluate(groundtruth_file,
             predictions,
             decisions,
             output_file=None):
    groundtruth = np.load(groundtruth_file)
    for name, data in [('Decision', decisions), ('Prediction', predictions)]:
        if not data.shape == groundtruth.shape:
            raise ValueError('{} file dimensions {} don'
                             't match the groundtruth {}'.format(
                                 name, data.shape, groundtruth.shape))

    results = {
        'ROC-AUC':
        metrics.roc_auc_score(groundtruth, predictions, average='macro'),
        'PR-AUC':
        metrics.average_precision_score(groundtruth,
                                        predictions,
                                        average='macro')
    }

    for average in ['macro', 'micro']:
        results['precision-' + average], results['recall-' + average], results['F-score-' + average], _ = \
            metrics.precision_recall_fscore_support(groundtruth, decisions, average=average)

    for metric, value in results.items():
        print('{}\t{:6f}'.format(metric, value))

    if output_file is not None:
        df = pd.DataFrame(results.values(), results.keys())
        df.to_csv(output_file, sep='\t', header=None, float_format='%.6f')

    return results

if __name__=='__main__':
    predictions, decisions = fuse()
    np.save('all.npy', predictions)
    evaluate(groundtruth_file=join(MEDIA_EVAL_PATH, 'mtg-jamendo-dataset', 'results', 'mediaeval2019', 'groundtruth.npy'),
             predictions=predictions,
             decisions=decisions,
             output_file='evaluation-on-test-DS_5s.tsv')
