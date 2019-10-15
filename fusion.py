import numpy as np
import pandas as pd
import click
from sklearn import metrics
from pathlib import Path
from os.path import join
from os import makedirs

MEDIA_EVAL_PATH = './MEDIA-EVAL19'
FUSION_BASE_PATH = './fusion'


def fuse(glob):
    predictions = np.array(list(map(np.load, glob)))
    mean_predictions = np.mean(predictions, axis=0)
    return mean_predictions
    

def calculate_decisions(groundtruth, predictions, tags, threshold_file=None, decision_file=None, display=False):
    if not predictions.shape == groundtruth.shape:
        raise ValueError('Prediction matrix dimensions {} don''t match the groundtruth {}'.format(
            predictions.shape, groundtruth.shape))

    n_tags = groundtruth.shape[1]
    if not n_tags == len(tags):
        raise ValueError('Number of tags in tag list ({}) doesn''t match the matrices ({})'.format(
            len(tags), n_tags))

    # Optimized macro F-score
    thresholds = {}
    for i, tag in enumerate(tags):
        precision, recall, threshold = metrics.precision_recall_curve(groundtruth[:, i], predictions[:, i])
        f_score = np.nan_to_num((2 * precision * recall) / (precision + recall))
        thresholds[tag] = threshold[np.argmax(f_score)]  # removed float()

    if display:
        for tag, threshold in thresholds.items():
            print('{}\t{:6f}'.format(tag, threshold))

    df = pd.DataFrame(thresholds.values(), thresholds.keys())
    if threshold_file is not None:
        df.to_csv(threshold_file, sep='\t', header=None)

    decisions = predictions > np.array(list(thresholds.values()))
    if decision_file is not None:
        np.save(decision_file, decisions)

    return thresholds, decisions



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

@click.command(help='Fuse model predictions.')
@click.option('-mep',
              '--media-eval-path',
              required=True,
              help='Path to media eval dataset. Should contain the "mtg-jamendo-dataset" folder.',
              type=click.Path(file_okay=False),
              default='MEDIA-EVAL19')
@click.option(
    '-o',
    '--output-dir',
    type=click.Path(writable=True, readable=True),
    help='Directory for storing fused predictions, decisions, thresholds and results.',
    default=None,
)
def main(media_eval_path,
output_dir=None):
    groundtruth_file = join(media_eval_path, 'mtg-jamendo-dataset', 'results', 'mediaeval2019', 'groundtruth.npy')
    tag_file = join(media_eval_path, 'mtg-jamendo-dataset', 'data', 'tags', 'moodtheme_split.txt')
    tags = pd.read_csv(tag_file, delimiter='\t', header=None)[0].to_list()
    dec_out = None
    threshold_out = None
    predictions_out = None
    eval_out = None

    if output_dir is not None:
        makedirs(output_dir, exist_ok=True)
    # all
    print('Fusion of all systems:')
    glob = Path(FUSION_BASE_PATH).glob('**/predictions.npy')
    predictions = fuse(glob)
    if output_dir is not None:
        fused_dir = join(output_dir, 'all')
        makedirs(fused_dir, exist_ok=True)
        dec_out = join(fused_dir, 'decisions.npy')
        threshold_out = join(fused_dir, 'thresholds.txt')
        predictions_out = join(fused_dir, 'predictions.npy')
        eval_out = join(fused_dir, 'eval-on-test.tsv')
        np.save(predictions_out, predictions)
    thresholds, decisions = calculate_decisions(groundtruth=np.load(groundtruth_file), predictions=predictions, tags=tags, threshold_file=threshold_out, decision_file=dec_out)
    evaluate(groundtruth_file=groundtruth_file,
             predictions=predictions,
             decisions=decisions,
             output_file=eval_out)

    # crnn
    print('Fusion of all CRNN systems:')
    glob = Path(FUSION_BASE_PATH).glob('crnn/**/predictions.npy')
    predictions = fuse(glob)
    if output_dir is not None:
        fused_dir = join(output_dir, 'crnn')
        makedirs(fused_dir, exist_ok=True)
        dec_out = join(fused_dir, 'decisions.npy')
        threshold_out = join(fused_dir, 'thresholds.txt')
        predictions_out = join(fused_dir, 'predictions.npy')
        eval_out = join(fused_dir, 'eval-on-test.tsv')
        np.save(predictions_out, predictions)
    thresholds, decisions = calculate_decisions(groundtruth=np.load(groundtruth_file), predictions=predictions, tags=tags, threshold_file=threshold_out, decision_file=dec_out)
    evaluate(groundtruth_file=groundtruth_file,
             predictions=predictions,
             decisions=decisions,
             output_file=eval_out)

    # DS
    print('Fusion of all DS systems:')
    glob = Path(FUSION_BASE_PATH).glob('DeepSpectrum/**/predictions.npy')
    predictions = fuse(glob)
    if output_dir is not None:
        fused_dir = join(output_dir, 'ds')
        makedirs(fused_dir, exist_ok=True)
        dec_out = join(fused_dir, 'decisions.npy')
        threshold_out = join(fused_dir, 'thresholds.txt')
        predictions_out = join(fused_dir, 'predictions.npy')
        eval_out = join(fused_dir, 'eval-on-test.tsv')
        np.save(predictions_out, predictions)
    thresholds, decisions = calculate_decisions(groundtruth=np.load(groundtruth_file), predictions=predictions, tags=tags, threshold_file=threshold_out, decision_file=dec_out)
    evaluate(groundtruth_file=groundtruth_file,
             predictions=predictions,
             decisions=decisions,
             output_file=eval_out)

    # DS 1s
    print('Fusion of all 1s DS systems:')
    glob = Path(FUSION_BASE_PATH).glob('DeepSpectrum/1s/**/predictions.npy')
    predictions = fuse(glob)
    if output_dir is not None:
        fused_dir = join(output_dir, 'ds-1s')
        makedirs(fused_dir, exist_ok=True)
        dec_out = join(fused_dir, 'decisions.npy')
        threshold_out = join(fused_dir, 'thresholds.txt')
        predictions_out = join(fused_dir, 'predictions.npy')
        eval_out = join(fused_dir, 'eval-on-test.tsv')
        np.save(predictions_out, predictions)
    thresholds, decisions = calculate_decisions(groundtruth=np.load(groundtruth_file), predictions=predictions, tags=tags, threshold_file=threshold_out, decision_file=dec_out)
    evaluate(groundtruth_file=groundtruth_file,
             predictions=predictions,
             decisions=decisions,
             output_file=eval_out)

    # DS 5s
    print('Fusion of all 5s DS systems:')
    glob = Path(FUSION_BASE_PATH).glob('DeepSpectrum/5s/**/predictions.npy')
    predictions = fuse(glob)
    if output_dir is not None:
        fused_dir = join(output_dir, 'ds-5s')
        makedirs(fused_dir, exist_ok=True)
        dec_out = join(fused_dir, 'decisions.npy')
        threshold_out = join(fused_dir, 'thresholds.txt')
        predictions_out = join(fused_dir, 'predictions.npy')
        eval_out = join(fused_dir, 'eval-on-test.tsv')
        np.save(predictions_out, predictions)
    thresholds, decisions = calculate_decisions(groundtruth=np.load(groundtruth_file), predictions=predictions, tags=tags, threshold_file=threshold_out, decision_file=dec_out)
    evaluate(groundtruth_file=groundtruth_file,
             predictions=predictions,
             decisions=decisions,
             output_file=eval_out)

if __name__=='__main__':
    main()