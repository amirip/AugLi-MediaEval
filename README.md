# AugLi-MediaEval
Code for Team AugLi's submission for the 2019 MediaEval Theme Recognition challenge.

> S. Amiriparian, M. Gerczuk, E. Coutinho, A. Baird, S. Ottl, M. Milling, and B. Schuller. Emotion and Themes Recognition in Music Utilising Convolutional and Recurrent Neural Networks. In MediaEval Benchmarking Initiative for Multimedia Evaluation. Sophia Antipolis, France.

## Dependencies
- python >= 3.6
- tensorflow
- keras
- kapre
- scikit-learn
- pandas
- click
- librosa

## Pre-Requisites
Clone the [mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset) and follow the instructions to download the data for the moodtheme subchallenge.


Our CRNN model operates on raw, mono-channel 16kHz wav conversions of the official challenge mp3s. You have to convert the songs (e.g. with ffmpeg or sox) to this format while keeping the original directory structure. Put them in a folder `wav` at the same level as  the `mtg-jamendo-dataset` repository. The parent directory will be denoted as `MEDIAEVAL19` from now on. The resulting directory structure should look like this:
```
 MEDIAEVAL19/
    |
    |__ wav/
    |
    |__ mtg-jamendo-dataset/
```

We also use a pretrained model from a [keras implementation](https://github.com/DTaoo/VGGish) of VGGish. You have to download the weights without the top fully connected layer ("vggish_audioset_weights_without_fc2.h5") to the `vggish` directory.

## Training the CRNN Models
To train the three CRNN models used in our fusion system with the same hyperparameters as in the paper, run the three commands below:
```bash
python -m crnn.py -mep MEDIAEVAL19/ -ebp ./fusion/crnn/lstm -rt lstm -tr MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv -v MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv -te MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv

python -m crnn.py -mep MEDIAEVAL19/ -ebp ./fusion/crnn/bilstm -rt bilstm -tr MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv -v MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv -te MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv

python -m crnn.py -mep MEDIAEVAL19/ -ebp ./fusion/crnn/gru -rt gru -tr MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv -v MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv -te MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv
```

The trained models and their test predictions will be saved to the directorys after `-ebp`.

## Deep Spectrum Systems
The other part of our fusion system makes use of the [Deep Spectrum](https://github.com/DeepSpectrum/DeepSpectrum) toolkit for audio feature extraction with pre-trained CNNs. Follow the instructions in the repository to install the toolkit - we recommend installing our official anaconda package.


### Extracting the Features
Two different feature sets must be extracted with Deep Spectrum. For a window size of 5s:
```bash
deepspectrum features MEDIAEVAL19/audio -t 5 5 -s 0 -e 29 -en VGG16 -el fc2 -fs mel -cm magma -m mel -nl -lf labels/autotagging_moodtheme-train.csv -o MEDIAEVAL19/features/DeepSpectrum/5s/train.csv

deepspectrum features MEDIAEVAL19/audio -t 5 5 -s 0 -e 29 -en VGG16 -el fc2 -fs mel -cm magma -m mel -nl -lf labels/autotagging_moodtheme-validation.csv -o MEDIAEVAL19/features/DeepSpectrum/5s/validation.csv

deepspectrum features MEDIAEVAL19/audio -t 5 5 -s 0 -e 29 -en VGG16 -el fc2 -fs mel -cm magma -m mel -nl -lf labels/autotagging_moodtheme-test.csv -o MEDIAEVAL19/features/DeepSpectrum/5s/test.csv
```

And for 1s windows:
```bash
deepspectrum features MEDIAEVAL19/audio -t 1 1 -s 0 -e 29 -en VGG16 -el fc2 -fs mel  -cm magma -m mel -nl -lf labels/autotagging_moodtheme-train.csv -o MEDIAEVAL19/features/DeepSpectrum/1s/train.csv

deepspectrum features MEDIAEVAL19/audio -t 1 1 -s 0 -e 29 -en VGG16 -el fc2 -fs mel -cm magma -m mel -nl -lf labels/autotagging_moodtheme-validation.csv -o MEDIAEVAL19/features/DeepSpectrum/1s/validation.csv

deepspectrum features MEDIAEVAL19/audio -t 1 1 -s 0 -e 29 -en VGG16 -el fc2 -fs mel -cm magma -m mel -nl -lf labels/autotagging_moodtheme-test.csv -o MEDIAEVAL19/features/DeepSpectrum/1s/test.csv
```

Finally, transform them to `.npz` files:
```bash
python transform_features.py MEDIA-EVAL19/features/DeepSpectrum/5s/ MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/

python transform_features.py MEDIA-EVAL19/features/DeepSpectrum/1s/ MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/
```

### Training the RNN Models
Next, train 3 RNN models for both featuresets:
```bash
python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/1s/lstm -tr MEDIA-EVAL19/features/DeepSpectrum/1s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/1s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/1s/test.npz -rt lstm

python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/1s/bilstm -tr MEDIA-EVAL19/features/DeepSpectrum/1s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/1s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/1s/test.npz -rt bilstm

python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/1s/gru -tr MEDIA-EVAL19/features/DeepSpectrum/1s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/1s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/1s/test.npz -rt gru



python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/5s/lstm -tr MEDIA-EVAL19/features/DeepSpectrum/5s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/5s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/5s/test.npz -rt lstm

python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/5s/bilstm -tr MEDIA-EVAL19/features/DeepSpectrum/5s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/5s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/5s/test.npz -rt bilstm

python train_rnn.py -mep MEDIAEVAL19/ -ebp ./fusion/DeepSpectrum/5s/gru -tr MEDIA-EVAL19/features/DeepSpectrum/5s/train.npz -v MEDIA-EVAL19/features/DeepSpectrum/5s/validation.npz -te MEDIA-EVAL19/features/DeepSpectrum/5s/test.npz -rt gru
```

## Fusion
After training, test set predictions of every model should have been saved to the corresponding experiment paths ("./fusion/..."). For the late fusion described in the paper, prediction scores are averaged and the decisions and metrics are computed with the baseline code. You can also use the included `fusion.py` script:
```bash
python fusion.py -mep MEDIA-EVAL19/ -o fusion-results
```
The results are printed to the commandline and also stored in the folder `fusion-results`.

## References

* Gemmeke, J. et. al.,
  [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html),
  ICASSP 2017

* Hershey, S. et. al.,
  [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html),
  ICASSP 2017

* [keras implementation of VGGish](https://github.com/DTaoo/VGGish)
