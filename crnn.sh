#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=10-0:0:0
#SBATCH --mem=32000
#SBATCH --array=0
#SBATCH --gres=gpu:2
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH -o /home/gerczuk/slurm_out/crnn/crnn.%A.%a.%N.out
#SBATCH -J multiGPU

source /home/gerczuk/.bashrc
# activate deep-spectrum venv
cd /home/gerczuk/projects/lazy-ml/scripts/crnn
conda activate lazy-ml
batchSizes=("32" "64" "128")
batchSize="32"
#batchSize="${batchSizes[$SLURM_ARRAY_TASK_ID]}"
numberOfWorkers="9"
numberOfGPUs="2"
optimizers=("rmsprop" "adam" "adagrad" "adadelta")
optimizers=("rmsprop" "adam" "adagrad" "adadelta")
#optimizer="${optimizers[$SLURM_ARRAY_TASK_ID]}"
optimizer="rmsprop"
lrs=("0.001" "0.001" "0.01" "1.0")
#lr="${lrs[$SLURM_ARRAY_TASK_ID]}"
lr="0.001"
epochs="1000"
experimentBasePath="./experiments-with-test-evaluation"
trainCSV="/nas/student/MauriceGerczuk/MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-train.tsv"
valCSV="/nas/student/MauriceGerczuk/MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-validation.tsv"
testCSV="/nas/student/MauriceGerczuk/MEDIA-EVAL19/mtg-jamendo-dataset/data/splits/split-0/autotagging_moodtheme-test.tsv"
mediaEvalBase="/nas/student/MauriceGerczuk/MEDIA-EVAL19"
dropoutFinal="0.3"
dropoutCNN="0.1"
dropoutRNN="0.3"
rnnTypes=("lstm" "gru" "bilstm")
#rnnType="${rnnTypes[@]:$SLURM_ARRAY_TASK_ID:1}"
rnnType="gru"
rnnLayers="2"
rnnUnits="512"
#losses=("focal" "binary_crossentropy")
#loss="${losses[@]:$SLURM_ARRAY_TASK_ID:1}"
loss="binary_crossentropy"
nonLinearity="elu"
focalAlpha="0.9"
spectrogramWidths=("128" "256" "512")
#spectrogramWidth="-sw 198"
#spectrogramWidth="-sw ${spectrogramWidths[$SLURM_ARRAY_TASK_ID]}"
spectrogramWidth="-w 120"
multiOutput=
vggish='-vggish'
timeStretch=
pitchShift=
randomNoise="--random-noise 0.2"
fineTunePortion="0.95"
trainableFilterbankSwitch=("" "--trainable-filterbank")
trainableFilterbank="${trainableFilterbankSwitch[$SLURM_ARRAY_TASK_ID]}"

command="python crnn.py --n-gpus $numberOfGPUs -ebp $experimentBasePath -ftp $fineTunePortion -nl $nonLinearity -nw $numberOfWorkers -o $optimizer -l $loss -fa $focalAlpha -lr $lr -e $epochs -tr $trainCSV -v $valCSV -te $testCSV -mep $mediaEvalBase -dof $dropoutFinal -docnn $dropoutCNN -dornn $dropoutRNN -bs $batchSize -rt $rnnType -ru $rnnUnits -rl $rnnLayers $spectrogramWidth $multiOutput $vggish $timeStretch $pitchShift $randomNoise $trainableFilterbank"
echo $command
eval ${command}
