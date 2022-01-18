#!/bin/bash
#SBATCH --job-name=WOLOAR_TEST
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=168G
#SBATCH -c 17
#SBATCH -n 1
#SBATCH -C "gpu32g"
#SBATCH -t 23:59:59
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

# only thing modified here is the loss function implementation... # SBATCH -C "gpu32g" # SBATCH --nodelist=node43

echo 'Starting Shell Script'
source /cluster/home/jmarsill/.bashrc
source activate light7

load_from_mets=False
# submit path to recently run model...
# update to satisfy OAR training ...
weights_path="/cluster/projects/radiomics/Temp/models/WOLNET_2020_08_28_152828/lightning_logs/version_2021387/checkpoints/last.ckpt"
model='WOLNET'
# with new windowing
model_name='WOLNET_2020_08_28_152828' # allows us to reload from previous settings...
pkl='VOLSOAR_2020_05_13_190628_582' # 'VOLSOAR_2020_05_06_185107' # 'VOLUMESGNECK_2020_02_18_100330' #'VOLUMESGNECK_2020_02_08_153149' # 'GCTV_2019_11_28_201830' # 'GCTVF2' # file created from databunch.py
metrics_pkl='METRICSOAR_2020_05_13_190628_582' # 'METRICSOAR_2020_05_06_185107' #  # split_mode='csv'
site='ALL' # 'Oropharynx' #'Oropharynx' # 'ALL' # 'Nasopharynx' # 'ALL' #  #  # 'ALL'   # ''  #  '--site' default site is Oropharynx
split_mode='csv' #'csv_full' #  #
div_ids='0,1,2,3' # number of gpus
data='RADCURE' #Dataset being used
loss_type="WFTTOPK" #"WDCTOPK" version1 # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
optim='RADAM' #'SGD' # 'RADAM' #'ADAM'
dce_version=1
deform=True
volume_type='oars'
oar_version=19
# new windowing used as of Aug 22/20
clip_min=-500
clip_max=1000
# clip_min=-300
# clip_max=200
gpus=1 # '0,1,2,3'
backend='ddp'
epoch=500 #500 # 100 # number of epochs
fold=2 # for Kfold validation
workers=17 # number of cpus used (each node has max of 45)
lr=.0016 # .0004 # learning rate for optimizer
weight_decay=0.000001 # .000001 # decay rate for optimizer
batch=1 # batch size # unet3D can use 2
factor=1 # resample by
tt_split=.9
aug_p=0.9
scheduler=True
scheduler_type='pleateau'

# 0.5 at 75 epochs for the training step...
gamma=0.96 # decay lr by this factor...
decay_after=1 # 15# 100 # 250 # decay lr after 4 epochs...
shuffle=True
classes=19 # number of classes (to test on), PAN HNSCC GTV/CTV... (Do we need/want that?)
norm='standard' # 'linear' # 'standard'
overfit=False # False
overfit_by=.15
scale_by=2
window=64 # default is 5
crop_factor=200 # 448 # 384 # default is 512
crop_as='3D'
external=True
fmaps=56
spacing='3mm' # spacing between slices...
filter=True
dir_path="/cluster/home/jmarsill" # server "/home/gpudual"
model_path="/cluster/projects/radiomics/Temp/models" #"--model-path"
img_path='/cluster/projects/radiomics/Temp/joe/OAR0720/img'
mask_path='/cluster/projects/radiomics/Temp/joe/OAR0720/masks'
# testing automatically set to false...
use_16bit=False
# save model to...
path=/cluster/home/jmarsill/SegmentHN/test.py # train.py #train_2 if training new model from scratch...
print_outputs_to=$model'_'$(date "+%b_%d_%Y_%T").txt

echo 'Started python script.'

# add this to preload trained model...
#--model-name $model_name \

python $path \
        --model-name $model_name \
        --scheduler $scheduler \
        --scheduler-type $scheduler_type \
        --aug-prob $aug_p \
        --use-16bit $use_16bit \
        --img-path $img_path \
        --mask-path $mask_path \
        --dir-path $dir_path \
        --model-path $model_path \
        --volume-type $volume_type \
        --oar-version $oar_version \
        --dce-version $dce_version \
        --weights-path $weights_path \
        --load-from-mets $load_from_mets \
        --spacing $spacing \
        --f-maps $fmaps \
        --metrics-name $metrics_pkl \
        --backend $backend \
        --overfit $overfit \
        --overfit-by $overfit_by \
        --gpus $gpus \
        --n-classes $classes\
        --shuffle-data $shuffle \
        --n-epochs $epoch \
        --pkl-name $pkl \
        --data $data \
        --tt-split $tt_split \
        --gamma $gamma \
        --decay-after $decay_after \
        --site $site \
        --model $model \
        --split-mode $split_mode \
        --device-ids $div_ids \
        --fold $fold \
        --workers $workers \
        --lr $lr \
        --decay $weight_decay \
        --batch-size $batch \
        --loss $loss_type \
        --optim $optim \
        --norm $norm \
        --crop-factor $crop_factor \
        --scale-factor $scale_by \
        --crop-as $crop_as \
        --clip-min $clip_min\
        --clip-max $clip_max \
        --window $window \
        --filter $filter \
        --resample $factor > $print_outputs_to

echo 'Python script finished.'
