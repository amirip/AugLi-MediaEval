import numpy as np
import joblib
import pandas as pd
import keras
import click
import datetime
from os.path import join, abspath
from os import makedirs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multiclass import OneVsRestClassifier
from metrics import SCIKIT_CLASSIFICATION_SCORERS_EXTENDED, ROC_AUC, ClassificationMetricCallback, ClassificationMetric, KERAS_METRIC_MODES, KERAS_METRIC_QUANTITIES
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn import metrics

RANDOM_SEED = 42

FEATURE_PATH = './MEDIA-EVAL19/features/DeepSpectrum/vgg16/fc2/mel/magma/5-5/'
MEDIA_EVAL_PATH = './MEDIA-EVAL19'

OPTIMIZERS = {
    'rmsprop': keras.optimizers.RMSprop,
}


def rnn_classifier_branch(inputs,
                          name='default',
                          dropout=0.2,
                          dropout_final=0.2,
                          rnn_type='gru',
                          rnn_units=256,
                          n_classes=2,
                          rnn_layers=1):
    x = inputs
    for i in range(rnn_layers):
        if rnn_type == 'gru':
            x = keras.layers.GRU(rnn_units,
                                 return_sequences=i < (rnn_layers - 1),
                                 dropout=dropout,
                                 recurrent_dropout=dropout,
                                 name=f'{name}/gru_{i}')(x)
        elif rnn_type == 'lstm':
            x = keras.layers.LSTM(rnn_units,
                                  return_sequences=i < (rnn_layers - 1),
                                  dropout=dropout,
                                  recurrent_dropout=dropout,
                                  name=f'{name}/lstm_{i}')(x)
        elif rnn_type == 'bilstm':
            x = keras.layers.Bidirectional(
                keras.layers.LSTM(rnn_units,
                                  return_sequences=i < (rnn_layers - 1),
                                  dropout=dropout,
                                  recurrent_dropout=dropout,
                                  name=f'{name}/lstm_{i}'))(x)
        x = keras.layers.BatchNormalization(name=f'{name}/batch_norm_{i}')(x)

    x = keras.layers.Dense(rnn_units, name=f'{name}/dense')(x)
    x = keras.layers.BatchNormalization(name=f'{name}/batch_norm_dense')(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dropout(dropout_final, name=f'{name}/dropout_final')(x)

    if n_classes == 2:
        x = keras.layers.Dense(1, activation='sigmoid',
                               name=f'{name}/output')(x)
    else:
        x = keras.layers.Dense(n_classes,
                               activation='sigmoid',
                               name=f'{name}/output')(x)
    return x


def build_model(classes,
                dropout_final=0.2,
                shape=(None, 5, 4096),
                rnn_type='gru',
                rnn_units=256,
                rnn_layers=1,
                rnn_dropout=0.2):
    inputs = keras.Input(shape=shape[1:])
    outputs = rnn_classifier_branch(inputs,
                                    name='rnn',
                                    rnn_type=rnn_type,
                                    rnn_units=rnn_units,
                                    dropout=rnn_dropout,
                                    dropout_final=dropout_final,
                                    n_classes=len(classes),
                                    rnn_layers=rnn_layers)
    model = keras.Model(inputs=inputs, outputs=outputs, name='rnn')

    model.summary()

    return model


def evaluate(groundtruth_file,
             prediction_file,
             decision_file,
             output_file=None):
    groundtruth = np.load(groundtruth_file)
    predictions = np.load(prediction_file)
    decisions = np.load(decision_file)

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

@click.command(help='Train RNN on ds features for mediaeval.')
@click.option('-tr',
              '--train-npz',
              nargs=1,
              required=True,
              help='Path to training npz file.',
              type=click.Path(exists=True, dir_okay=False, readable=True),
              default=join(FEATURE_PATH, 'train.npz'))
@click.option('-v',
              '--val-npz',
              required=True,
              help='Path to validation npz file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              default=join(FEATURE_PATH, 'validation.npz'))
@click.option('-te',
              '--test-npz',
              required=True,
              help='Path to test npz file.',
              nargs=1,
              type=click.Path(exists=True, dir_okay=False, readable=True),
              default=join(FEATURE_PATH, 'test.npz'))
@click.option(
    '-bs',
    '--batch-size',
    type=int,
    help='Define batch size.',
    default=32,
)
@click.option(
    '-e',
    '--epochs',
    type=int,
    help='Define number of training epochs.',
    default=50,
)
@click.option(
    '-lr',
    '--learning-rate',
    type=click.FloatRange(0, 1),
    default=0.001,
    help='Learning rate for optimizer.',
)
@click.option(
    '-dornn',
    '--rnn-dropout',
    type=click.FloatRange(0, 1),
    default=0.3,
    help='RNN Dropout.',
)
@click.option(
    '-dof',
    '--dropout-final',
    type=click.FloatRange(0, 1),
    default=0.3,
    help='Final Denselayer dropout.',
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default=join(FEATURE_PATH, 'experiments'),
)
@click.option(
    '-ru',
    '--rnn-units',
    type=int,
    help='Number of units on recurrent layers.',
    default=256,
)
@click.option(
    '-rl',
    '--rnn-layers',
    type=int,
    help='Number of recurrent layers.',
    default=2,
)
@click.option('-o',
              '--optimizer',
              type=click.Choice(OPTIMIZERS.keys()),
              help='Optimizer used for training.',
              default='adam')
@click.option('-rt',
              '--rnn-type',
              type=click.Choice(('bilstm', 'lstm', 'gru')),
              help='Optimizer used for training.',
              default='gru')
def train(train_npz=None,
          val_npz=None,
          test_npz=None,
          experiment_base_path=None,
          epochs=20,
          optimizer='adam',
          learning_rate=0.001,
          batch_size=32,
          rnn_type='gru',
          rnn_units=256,
          rnn_layers=2,
          dropout_final=0.3,
          rnn_dropout=0.3):
    loaded_train = np.load(train_npz)
    X_train = loaded_train['X']
    y_train = loaded_train['y']
    loaded_eval = np.load(val_npz)
    X_eval = loaded_eval['X']
    y_eval = loaded_eval['y']
    tags = loaded_train['tags']
    print(f'Classes: {tags}')
    print(X_train.shape)
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_base_path = abspath(experiment_base_path)
    experiment_base_path = join(
        experiment_base_path,
        f'{rnn_type}-{rnn_layers}x{rnn_units}-{optimizer}-lr-{learning_rate}-bs-{batch_size}-epochs-{epochs}-dropoutFinal-{dropout_final:.1f}-dropoutRNN-{rnn_dropout:.1f}'
    )
    experiment_base_path = join(experiment_base_path, now)
    makedirs(experiment_base_path, exist_ok=True)

    m = build_model(classes=tags,
                    shape=(None, *X_train.shape[1:]),
                    dropout_final=dropout_final,
                    rnn_type=rnn_type,
                    rnn_units=rnn_units,
                    rnn_layers=rnn_layers,
                    rnn_dropout=rnn_dropout)
    opt = OPTIMIZERS[optimizer](lr=learning_rate)
    m.compile(loss='binary_crossentropy',
              metrics=[keras.metrics.categorical_accuracy],
              optimizer=opt)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=join(
        experiment_base_path, 'log'),
                                             batch_size=batch_size,
                                             histogram_freq=0,
                                             write_graph=True)
    metricsCallback = ClassificationMetricCallback(labels=tags,
                                                   multi_label=True,
                                                   validation_data=(X_eval, y_eval))
    early_stopping = keras.callbacks.EarlyStopping(monitor=KERAS_METRIC_QUANTITIES[ROC_AUC], min_delta=0, patience=100, verbose=1, mode=KERAS_METRIC_MODES[ROC_AUC], baseline=None, restore_best_weights=False)
    mc = keras.callbacks.ModelCheckpoint(
        join(experiment_base_path, 'checkpoints', 'weights.h5'),
        monitor=KERAS_METRIC_QUANTITIES[ROC_AUC],
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=KERAS_METRIC_MODES[ROC_AUC],
        period=1)
    makedirs(join(experiment_base_path, 'checkpoints'), exist_ok=True)
    m.fit(x=X_train,
          y=y_train,
          epochs=epochs,
          validation_data=(X_eval, y_eval),
          callbacks=[metricsCallback, tbCallBack, mc, early_stopping],
          verbose=2,
          shuffle=True)
    m.load_weights(join(experiment_base_path, 'checkpoints', 'weights.h5'))
    loaded_test = np.load(test_npz)
    X_test = loaded_test['X']
    y_test = loaded_test['y']
    scores = m.predict(X_test)
    epoch_rocs = [
        d[KERAS_METRIC_QUANTITIES[ROC_AUC]] for d in metricsCallback._data
    ]
    best_epoch = np.argmax(epoch_rocs)
    cutoffs = metricsCallback._binary_cutoffs[best_epoch]
    predicted_tags = ClassificationMetric._transform_arrays(
        y_true=np.zeros_like(scores),
        y_pred=scores,
        binary=False,
        multi_label=True,
        binary_cutoffs=cutoffs)[1].astype(bool)
    print(scores.shape, predicted_tags.shape)
    np.save(join(experiment_base_path, 'predictions.npy'), scores)
    np.save(join(experiment_base_path, 'decisions.npy'), predicted_tags)
    evaluate(groundtruth_file=join(MEDIA_EVAL_PATH, 'mtg-jamendo-dataset',
                                   'data', 'mediaeval2019', 'groundtruth.npy'),
             prediction_file=join(experiment_base_path, 'predictions.npy'),
             decision_file=join(experiment_base_path, 'decisions.npy'),
             output_file=join(experiment_base_path, 'evaluation-on-test.tsv'))


if __name__ == '__main__':
    train()