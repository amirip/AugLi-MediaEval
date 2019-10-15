#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=10-0:0:0
#SBATCH --mem=32000
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/gerczuk/slurm_out/lazy-ml/rnn-mediaeval.%A.%a.%N.out
#SBATCH -J rnnM/E

source /home/gerczuk/.bashrc
# activate deep-spectrum venv
conda activate lazy-ml
batchSizes=("32" "64" "128")
batchSize="32"
#batchSize="${batchSizes[$SLURM_ARRAY_TASK_ID]}"
optimizers=("rmsprop" "adam" "adagrad" "adadelta")
optimizers=("rmsprop" "adam" "adagrad" "adadelta")
#optimizer="${optimizers[$SLURM_ARRAY_TASK_ID]}"
optimizer="rmsprop"
lrs=("0.001" "0.001" "0.01" "1.0")
#lr="${lrs[$SLURM_ARRAY_TASK_ID]}"
lr="0.001"
epochs="1000"
featureBasePath="/nas/student/MauriceGerczuk/MEDIA-EVAL19/features/DeepSpectrum/vgg16/fc2/mel/magma/1-1"
experimentBasePath="$featureBasePath/experiments"
train="$featureBasePath/train.npz"
val="$featureBasePath/validation.npz"
test="$featureBasePath/test.npz"
dropoutFinal="0.4"
dropoutRNN="0.4"
rnnTypes=("lstm" "gru" "bilstm")
rnnType="${rnnTypes[@]:$SLURM_ARRAY_TASK_ID:1}"
#rnnType="gru"
rnnLayers="2"
rnnUnits="1024"

command="python train_rnn.py -ebp $experimentBasePath -o $optimizer -e $epochs -tr $train -v $val -te $test -dof $dropoutFinal -dornn $dropoutRNN -bs $batchSize -rt $rnnType -ru $rnnUnits -rl $rnnLayers"
echo $command
eval ${command}
