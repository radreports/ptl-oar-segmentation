#!/bin/bash
#SBATCH --job-name=WOLGTV1
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=169G
#SBATCH -c 38
#SBATCH -n 1
#SBATCH -C "gpu32g"
#SBATCH -t 2-23:59:59
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics

# only thing modified here is the loss function implementation...

echo 'Starting Shell Script'
source /cluster/home/jmarsill/.bashrc
source activate light8
# Have to modify this for new pipeline
# Arguments to pass to trainer
load_from_mets=False
weights_path="/cluster/projects/radiomics/Temp/models/WOLNET_2021_02_22_200100/lightning_logs/version_2140894/checkpoints/finetune.ckpt" # testing.ckpt # radiomics/Temp/models/WOLNET_2020_09_18_135329/lightning_logs/version_1456310/checkpoints/last.ckpt"
# "/cluster/projects/radiomics/Temp/models/WOLNET_2020_09_18_135329/lightning_logs/version_1491014/checkpoints/last.ckpt"
# /WOLNET_2020_06_25_113232/lightning_logs/version_1090340/checkpoints/_model_version_67.ckpt"
# meta_path='/home/gpudual/bhklab/private/jmarsill/models/WOLNET_2020_05_15_233123/lightning_logs/version_0/checkpoints/meta_tags.csv'
model='WOLNET'
model_name='WOLNET_2021_02_22_200100' # 'WOLNET_2020_09_18_135329' # Old GTV models in here...
pkl='VOLSGTV_2020_06_09_112516_2426' # 'VOLSOAR_2020_05_06_185107' # 'VOLUMESGNECK_2020_02_18_100330' #'VOLUMESGNECK_2020_02_08_153149' # 'GCTV_2019_11_28_201830' # 'GCTVF2' # file created from databunch.py
metrics_pkl='METRICSGTV_2020_06_09_112516_2426' # 'METRICSOAR_2020_05_06_185107' #  # split_mode='csv'
site='ALL' # 'Oropharynx' #'Oropharynx' # 'ALL' # 'Nasopharynx' # 'ALL' #  #  # 'ALL'   # ''  #  '--site' default site is Oropharynx
split_mode='csv_full' #'csv_full' #  #
div_ids='0,1,2,3' # number of gpus
data='RADCURE' # Dataset being used
loss_type="WFTTOPK" # 'FOCALDSC'  'CATEGORICAL' # loss_type='COMBINED' # inital runs without TOPK, if multi consider using it...
optim='RADAM' #'AMSBOUND' #'ADAM'
dce_version=1
volume_type='targets'
reload=True
oar_version=1
clip_min=-500 # -200
clip_max=1000 # 300
gpus='0,1,2,3'
backend='ddp'
epoch=500 #500 # 100 # number of epochs
fold=1 # for Kfold validation
workers=9  # number of cpus used (each node has max of 45)
lr=.0016 # learning rate for optimizer
decay=0.0000005 # .000001 # decay rate for optimizer
batch=2 # batch size # unet3D can use 2
factor=1 # resample by
tt_split=.9
single_fold=True
# 0.5 at 75 epochs for the training step...
# only used if scheduler_type == 'step'
gamma=0.9 # 0.92  #decay lr by this factor...
decay_after=4 # 1 # try 2 #15# 100 # 250 # decay lr after 4 epochs...
scheduler=True
scheduler_type='pleateau'
shuffle=True
classes=1 # 6 # breaking up analysis by disease site... # number of classes (to test on), PAN HNSCC GTV/CTV... (Do we need/want that?)
norm='standard' # 'linear' # 'standard'
overfit=False # False
overfit_by=1
scale_by=2
window=64 # default is 5
crop_factor=176 # 448 # 384 # default is 512
crop_as='3D'
filter_=True
fmaps=56
spacing='3mm' # spacing between slices...
filter=True
dir_path="/cluster/home/jmarsill" # server "/home/gpudual"
model_path="/cluster/projects/radiomics/Temp/models" #"--model-path"
img_path='/cluster/projects/radiomics/Temp/joe/OAR1204/img' # '/cluster/projects/radiomics/Temp/joe/OAR1204/img' # '/cluster/projects/radiomics/Temp/DDNN/img'
mask_path='/cluster/projects/radiomics/Temp/RADCURE_LN/masks' # '/cluster/projects/radiomics/Temp/joe/OAR1204/masks' # '/cluster/projects/radiomics/RADCURE-challenge/data/training/masks' # '/cluster/projects/radiomics/Temp/OARS050120/masks'
scheduler=False
scheduler_type='step'
external=False
use_16bit=False
aug_p=0.9
dataset='GTV'
# save model to...
path=/cluster/home/jmarsill/SegmentHN/train.py #train_2 if training new model
print_outputs_to=$model'_GTV_'$(date "+%b_%d_%Y_%T").txt

echo 'Started python script.'

#
#
python $path \
        --model-name $model_name \
        --weights-path $weights_path \
        --external $external \
        --scheduler $scheduler \
        --scheduler-type $scheduler_type \
        --aug-prob $aug_p \
        --use-16bit $use_16bit \
        --mask-path $mask_path \
        --img-path $img_path \
        --model-path $model_path \
        --dir-path $dir_path \
        --volume-type $volume_type \
        --oar-version $oar_version \
        --dce-version $dce_version \
        --spacing $spacing \
        --f-maps $fmaps \
        --filter $filter_ \
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
        --decay $decay \
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
