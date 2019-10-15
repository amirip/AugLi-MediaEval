import sys

import tensorflow as tf
import csv
import numpy as np
import click
import datetime
import librosa
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from metrics import ClassificationMetricCallback, KERAS_METRIC_QUANTITIES, KERAS_METRIC_MODES, ROC_AUC, ClassificationMetric
from os.path import splitext, join, relpath, dirname, abspath
from os import makedirs
from random import randint
from kapre.time_frequency import Melspectrogram
from kapre.augmentation import AdditiveNoise
from kapre.utils import Normalization2D
import keras
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.utils import multi_gpu_model
from vggish.vggish import VGGish
import vggish.vggish_params


OPTIMIZERS = {
    'rmsprop': keras.optimizers.RMSprop,
}

class AudioDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 csv_file,
                 directory,
                 batch_size=32,
                 window=20,
                 shuffle=True,
                 random_state=42,
                 label_column='TAGS',
                 sr=16000,
                 time_stretch=None,
                 pitch_shift=None,
                 save_dir=None,
                 label_binarizer=None):
        self.files = []
        self.classes = []
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            label_index = header.index(label_column)
            path_index = header.index('PATH')
            for line in reader:
                self.files.append(
                    join(directory, f'{splitext(line[path_index])[0]}.wav'))
                self.classes.append(tuple(line[label_index:]))
        if label_binarizer is None:
            self.label_binarizer = MultiLabelBinarizer()
            self.label_binarizer.fit(self.classes)
        else:
            self.label_binarizer = label_binarizer
        self.directory = directory
        self.window = window
        self.labels = self.label_binarizer.transform(self.classes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.time_stretch = time_stretch
        self.pitch_shift = pitch_shift
        self.save_dir = save_dir
        self.sr = sr
        np.random.seed(self.random_state)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        files_batch = [self.files[k] for k in indexes]
        y = np.asarray([self.labels[k] for k in indexes])

        # Generate data
        x = self.__data_generation(files_batch)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):
        audio_data = []

        for file in files:
            duration = librosa.core.get_duration(filename=file)

            if self.window is not None:
                stretched_window = self.window * (
                    1 + self.time_stretch
                ) if self.time_stretch is not None else self.window
                if self.shuffle:
                    start = randint(0, max(1, int(duration - stretched_window)))

                else:
                    start = duration / 2 if duration / 2 > stretched_window else 0  # take the middle chunk
                y, sr = librosa.core.load(file,
                                          offset=start,
                                          duration=min(stretched_window, duration),
                                          sr=self.sr)
                y = self.__get_random_transform(y, sr)
                end_sample = min(int(self.window * sr), int(duration * sr))
                y = y[:end_sample]
            else:
                y, sr = librosa.core.load(file, sr=self.sr)
                y = self.__get_random_transform(y, sr)

            if self.save_dir:
                rel_path = relpath(file, self.directory)
                save_path = join(self.save_dir, f'{splitext(rel_path)[0]}.wav')
                makedirs(dirname(save_path), exist_ok=True)
                librosa.output.write_wav(
                    join(self.save_dir, f'{splitext(rel_path)[0]}.wav'),
                    audio_data, sr)
            audio_data.append(y)

        audio_data = keras.preprocessing.sequence.pad_sequences(
            audio_data, dtype='float32')
        return audio_data

    def __get_random_transform(self, y, sr):
        if self.time_stretch is not None:
            factor = np.random.uniform(1 - self.time_stretch,
                                       1 + self.time_stretch)
            y = librosa.effects.time_stretch(y, factor)
        if self.pitch_shift is not None:
            steps = np.random.randint(0 - self.pitch_shift,
                                      1 + self.pitch_shift)
            y = librosa.effects.pitch_shift(y, sr, steps)
        return y


def conv2d_block(x,
                 n_filters=128,
                 filter_shape=(4, 4),
                 pool_size=(2, 2),
                 dropout=0.2,
                 n_conv=1):
    for i in range(n_conv):
        x = keras.layers.Conv2D(n_filters, filter_shape, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
       
    x = keras.layers.MaxPooling2D(pool_size=pool_size)(x)
    x = keras.layers.Dropout(dropout)(x)
    return x


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

    x = keras.layers.Dense(1024, name=f'{name}/dense')(x)
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


def build_model_vggish(classes,
                       dropout_final=0.2,
                       shape=(None, 320000),
                       sr=16000,
                       rnn_type='gru',
                       rnn_units=256,
                       focal_alpha=0.95,
                       rnn_layers=1,
                       rnn_dropout=0.2,
                       activation='elu',
                       random_noise=0.2,
                       weights='soundnet'):

    inputs = keras.Input(shape=shape[1:])
    x = keras.layers.Reshape(target_shape=(1, -1))(inputs)
    x = Melspectrogram(n_dft=512,
                       n_hop=256,
                       padding='same',
                       sr=sr,
                       n_mels=64,
                       fmin=125,
                       fmax=7500,
                       power_melgram=1.0,
                       return_decibel_melgram=True,
                       name='trainable_stft')(x)
    if random_noise:
        x = AdditiveNoise(power=random_noise, random_gain=True)(x)
    x = Normalization2D(str_axis='freq')(x)
    x = Lambda(lambda x: K.permute_dimensions(x=x, pattern=(0, 2, 1, 3)),
               name="transpose")(x)

    vggish = VGGish(include_top=False,
                    load_weights=weights,
                    input_shape=x.get_shape().as_list()[1:],
                    pooling=None)
    if weights is not None: # only freeze when using pretrained layers
        for layer in vggish.layers:
            layer.trainable = False
    x = vggish(x)
    x = keras.layers.AveragePooling2D(pool_size=(1, 4))(x)
    x = keras.layers.Reshape(target_shape=(-1, 512))(x)

    outputs = rnn_classifier_branch(x,
                                    name='rnn',
                                    dropout=rnn_dropout,
                                    dropout_final=dropout_final,
                                    rnn_units=rnn_units,
                                    rnn_type=rnn_type,
                                    n_classes=len(classes),
                                    rnn_layers=rnn_layers)

    model = keras.Model(inputs=inputs, outputs=outputs, name='crnn')

    model.summary()

    return model, vggish


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


@click.command(help='Train CRNN for mediaeval.')
@click.option('-mep',
              '--media-eval-path',
              required=True,
              help='Path to media eval dataset. Should contain the "mtg-jamendo-dataset" folder.',
              type=click.Path(file_okay=False),
              default='./MEDIA-EVAL19')
@click.option(
    '-tr',
    '--train-csv',
    nargs=1,
    required=True,
    help='Path to training csv file.',
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=
    './MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv'
)
@click.option(
    '-v',
    '--val-csv',
    required=True,
    help='Path to validation csv file.',
    nargs=1,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=
    './MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv'
)
@click.option(
    '-te',
    '--test-csv',
    required=True,
    help='Path to test csv file.',
    nargs=1,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=
    './MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv'
)
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
    default=100,
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
    '-nw',
    '--n-workers',
    type=int,
    help='Number of training worker processes.',
    default=4,
)
@click.option(
    '-ebp',
    '--experiment-base-path',
    type=click.Path(writable=True, readable=True),
    help='Basepath where logs and checkpoints should be stored.',
    default='./experiments',
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
              default='rmsprop')
@click.option('-rt',
              '--rnn-type',
              type=click.Choice(('bilstm', 'lstm', 'gru')),
              help='Optimizer used for training.',
              default='gru')
@click.option(
    '-ps',
    '--pitch-shift',
    type=int,
    help='Number of semitones to randomly shift the training input up or down (for augmentation purposes).',
    default=None,
)
@click.option(
    '-ts',
    '--time-stretch',
    type=click.FloatRange(0, 3),
    help=
    'Maximum rate of random timestretch transformation (for augmentation purposes).',
    default=None,
)
@click.option(
    '-rn',
    '--random-noise',
    type=click.FloatRange(0, 1),
    help=
    'Add random white noise with maximum power.',
    default=0.1,
)
@click.option(
    '-w',
    '--window',
    type=int,
    help='Window size.',
    default=20,
)
@click.option(
    '-ftp',
    '--fine_tune_portion',
    type=click.FloatRange(0, 1),
    default=0.8,
    help=
    'Portion of the training epochs used for finetuning the lower levels after the classification head has been trained.',
)
def train(
        media_eval_path='./MEDIA-EVAL19',
        train_csv='./MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv',
        val_csv='./MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv',
        test_csv='./MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv',
        experiment_base_path='./experiment',
        epochs=100,
        optimizer='rmsprop',
        learning_rate=0.001,
        n_workers=16,
        batch_size=32,
        rnn_type='gru',
        window=20,
        rnn_units=256,
        rnn_layers=2,
        dropout_final=0.3,
        rnn_dropout=0.3,
        fine_tune_portion=0.8,
        pitch_shift=None,
        time_stretch=None,
        random_noise=0.1):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    fine_tune_epochs = round(fine_tune_portion * epochs)
    experiment_base_path = abspath(experiment_base_path)
    """ experiment_base_path = join(
        experiment_base_path,
        f'noise-{random_noise}-pitchShift-{pitch_shift}-timeStretch-{time_stretch}-spectrogramWindow-{str(window)+"s"}-{rnn_type}-{rnn_layers}x{rnn_units}-{optimizer}-lr-{learning_rate}-bs-{batch_size}-epochs-{epochs}-dropoutFinal-{dropout_final:.1f}-dropoutRNN-{rnn_dropout:.1f}'
    )
    experiment_base_path = join(experiment_base_path, now) """
    makedirs(experiment_base_path, exist_ok=True)

    train_gen = AudioDataGenerator(csv_file=train_csv,
                                   directory=join(media_eval_path, 'wavs'),
                                   window=window,
                                   batch_size=batch_size,
                                   time_stretch=time_stretch,
                                   pitch_shift=pitch_shift,
                                   shuffle=True)

    eval_gen = AudioDataGenerator(csv_file=val_csv,
                                  directory=join(media_eval_path, 'wavs'),
                                  window=window,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  label_binarizer=train_gen.label_binarizer)
    classes = train_gen.label_binarizer.classes_
    print(f'Classes: {classes}')
    x, y = train_gen[0]
    print(x.shape)

    opt = OPTIMIZERS[optimizer](lr=learning_rate)
    
    m, vggish = build_model_vggish(classes=classes,
                        shape=(None, None, *x.shape[2:]),
                        dropout_final=dropout_final,
                        sr=train_gen.sr,
                        rnn_type=rnn_type,
                        rnn_units=rnn_units,
                        rnn_layers=rnn_layers,
                        rnn_dropout=rnn_dropout,
                        random_noise=random_noise,
                        weights='soundnet')
      
    m.compile(loss='binary_crossentropy',
              metrics=[keras.metrics.categorical_accuracy],
              optimizer=opt)
    tbCallBack = keras.callbacks.TensorBoard(log_dir=join(
        experiment_base_path, 'log'),
                                             batch_size=batch_size,
                                             histogram_freq=0,
                                             write_graph=True)
    metricsCallback = ClassificationMetricCallback(
        labels=train_gen.label_binarizer.classes_,
        validation_generator=eval_gen,
        multi_label=True)
    mc = keras.callbacks.ModelCheckpoint(
        join(experiment_base_path, 'checkpoints', 'weights.h5'),
        monitor=KERAS_METRIC_QUANTITIES[ROC_AUC],
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode=KERAS_METRIC_MODES[ROC_AUC],
        period=1)
    makedirs(join(experiment_base_path, 'checkpoints'), exist_ok=True)
    m.fit_generator(train_gen,
                    epochs=epochs - fine_tune_epochs,
                    validation_data=eval_gen,
                    max_queue_size=2 * n_workers,
                    callbacks=[metricsCallback, tbCallBack, mc],
                    verbose=2,
                    use_multiprocessing=True,
                    workers=n_workers)
    
    if fine_tune_epochs > 0:
        print(
            'Finished training RNN classifier head. Proceeding to finetune VGGish feature layers. Loading best weights...'
        )
        m.load_weights(join(experiment_base_path, 'checkpoints', 'weights.h5'))
        opt = keras.optimizers.SGD(lr=learning_rate/10)
        for layer in m.layers:
            m.trainable = True
        for layer in vggish.layers:
            layer.trainable = True
        m.compile(loss='binary_crossentropy',
                metrics=[keras.metrics.categorical_accuracy],
                optimizer=opt)
        m.summary()
        m.fit_generator(train_gen,
                        initial_epoch=epochs - fine_tune_epochs,
                        epochs=epochs,
                        validation_data=eval_gen,
                        max_queue_size=2 * n_workers,
                        callbacks=[metricsCallback, tbCallBack, mc],
                        verbose=2,
                        use_multiprocessing=True,
                        workers=n_workers)

    print('Finished training. Now generating test predictions.')
    m.load_weights(join(experiment_base_path, 'checkpoints', 'weights.h5'))
    test_gen = AudioDataGenerator(csv_file=test_csv,
                                  directory=join(media_eval_path, 'wavs'),
                                  window=window,
                                  batch_size=1,
                                  shuffle=False,
                                  label_binarizer=train_gen.label_binarizer)
    scores = m.predict_generator(test_gen,
                                 max_queue_size=2 * n_workers,
                                 verbose=2,
                                 use_multiprocessing=True,
                                 workers=n_workers)
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
    evaluate(groundtruth_file=join(media_eval_path, 'mtg-jamendo-dataset', 'data',
                              'mediaeval2019', 'groundtruth.npy'),
             prediction_file=join(experiment_base_path, 'predictions.npy'),
             decision_file=join(experiment_base_path, 'decisions.npy'),
             output_file=join(experiment_base_path, 'evaluation-on-test.tsv'))


if __name__ == '__main__':
    train()
