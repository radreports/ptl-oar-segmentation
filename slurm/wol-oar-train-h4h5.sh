#!/bin/bash
#SBATCH --job-name=OAR_GLAND
#SBATCH --mem=84G
#SBATCH -c 12
#SBATCH --gres=gpu:4
#SBATCH -n 1
#SBATCH -t 2-23:59:59
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

echo 'Starting Shell Script'
source /cluster/home/jmarsill/.bashrc
source activate light8

model='WOLNET' # with new windowing
model_name='WOLNET_2022_09_12_160955' # allows us to reload from previous settings...
site='ALL' # 'Oropharynx' #'Oropharynx' # 'ALL' # 'Nasopharynx' # 'ALL' #  #  # 'ALL'   # ''  #  '--site' default site is Oropharynx
split_mode='csv' # 'csv_full' #  #
div_ids='0,1,2,3' # number of gpus
data='RADCURE' # Dataset being used
loss_type="WFTTOPK" #"WDCTOPK" version1 # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
# data='RADCURE' #Dataset being used
# loss_type="TAL" # "WFTTOPK" #"WDCTOPK" version1 # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
optim='RADAM' #'SGD' # 'RADAM' #'ADAM'
dce_version=1
deform=True
volume_type='oars' # oar_version="" # new windowing used as of Aug 22/20
clip_min=-500
clip_max=1000 # clip_min=-300 # clip_max=200
gpus='0,1,2,3' # 2,3' # ,4,5,6,7'
backend='ddp'
epoch=500 # 500 # 100 # number of epochs
fold=3 # for Kfold validation, fold 1 already completed...
workers=3 # number of cpus used (each node has max of 45)
lr=.001 # .00016 # .0004 # learning rate for optimizer
weight_decay=0.000001 # .000001 # decay rate for optimizer
batch=1 # batch size # unet3D can use 2
factor=1 # resample by
tt_split=.9
aug_p=0.9
scheduler=True
scheduler_type='pleateau' # 0.5 at 75 epochs for the training step...
gamma=0.975 # decay lr by this factor...
decay_after=1 # 15# 100 # 250 # decay lr after 4 epochs...
shuffle=True
classes=13 #19 # number of classes (to test on), PAN HNSCC GTV/CTV... (Do we need/want that?)
norm='standard' # 'linear' # 'standard'
overfit=False # False
overfit_by=.15
scale_by=2
window=56 # default is 5
tag="GLANDS"
crop_factor=192 # 448 # 384 # default is 512
crop_as='3D'
external=False
fmaps=56
spacing='3mm' # spacing between slices...
filter=True
data_path="/cluster/projects/radiomics/Temp/joe/RADCURE_VECTOR_UPDATE/" # "/storage/data/ml2022/RADCURE_VECTOR/"
home_path="/cluster/home/jmarsill/ptl-oar-segmentation/" # "/h/jmarsilla/ptl-oar-segmentation/" # server "/home/gpudual"
model_path="/cluster/projects/radiomics/Temp/joe/models-1222" # "/h/jmarsilla/models" #"--model-path"
use_16bit=False # save model to...
path="/cluster/home/jmarsill/ptl-oar-segmentation/train.py" #train_2 if training new model
print_outputs_to=$model'_'$tag'_'$(date "+%b_%d_%Y_%T").txt

echo 'Started python script.'

python $path --model $model --model-name $model_name --external $external \
        --scheduler $scheduler --scheduler-type $scheduler_type --aug-prob $aug_p \
        --use-16bit $use_16bit --data-path $data_path --home-path $home_path \
        --model-path $model_path --volume-type $volume_type \
        --dce-version $dce_version --spacing $spacing --f-maps $fmaps \
        --backend $backend --overfit $overfit --overfit-by $overfit_by \
        --gpus $gpus --n-classes $classes --shuffle-data $shuffle \
        --n-epochs $epoch --data $data --tt-split $tt_split \
        --gamma $gamma --decay-after $decay_after --site $site \
        --split-mode $split_mode --device-ids $div_ids --fold $fold \
        --workers $workers --lr $lr --decay $weight_decay --batch-size $batch \
        --loss $loss_type --optim $optim --norm $norm --crop-factor $crop_factor \
        --scale-factor $scale_by --crop-as $crop_as --clip-min $clip_min\
        --clip-max $clip_max --window $window --filter $filter \
        --resample $factor --tag $tag > $print_outputs_to

echo 'Python script finished.'
