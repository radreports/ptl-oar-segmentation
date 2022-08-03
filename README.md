# PyTorch Lightning 3D OAR Segmentation for Automated Radiation Therapy Planning

Original Repo for "[Evaluating clinical acceptability of organ-at-risk segmentation In head & neck cancer using a compendium of open-source 3D convolutional neural networks](https://www.medrxiv.org/content/10.1101/2022.01.15.22269276v1.full.pdf)" (Marsilla et. al, 2022). Pre-print of this paper was accepted to MedRxiv on January 25, 2022.

## Updates for Vector Challenge

[![Video For 2022 CDI Vector Institute Machine Learning Challenge](https://img.youtube.com/vi/nODSJWYwJhY/0.jpg)](https://www.youtube.com/watch?v=nODSJWYwJhY)

Hi Everyone! This is the official repo describing the base boilerplate that will be used for inference of the CDI-VECTOR Machine Learning Challenge. The following evaluation function will be used to rank performance of submissions by participants.

### Evaluation Function

![Evaluation Function](https://keep.google.com/u/0/media/v2/1RP2OPOEyMARmUCqT5SefyPhYWlwhfHbdq4i2bQiS1kpnauS3PmIBTD57Hsa9tf8/1WiXRuoc9gTbTCtMrYDPPIwcbmmo1Y-jS1o9Cg4rknN_3uwZNo0PAYmNcyeoRyQ?accept=image%2Fgif%2Cimage%2Fjpeg%2Cimage%2Fjpg%2Cimage%2Fpng%2Cimage%2Fwebp%2Caudio%2Faac&sz=754)

#### Update: 19/07/2022

self.CalcEvaluationMetric(outs, targets, batch_idx) has been added to SegmentationModule in modules_vector.py. The goal of this function is to record the evaluation metrics for each patient in the test set into a global csv. This will be used to calculate the final mean evaluation metric.  To better understand the equation, please check the added notebook that makes use of data extracted from the original study.

Below are some suggestions, which would be useful to keep the evaluation process efficient.

1. Please create a template of your model and package it **using the boilerplate provided to ensure that we can run your model without issues.**
2. self.CalcEvaluationMetric used in test_step() module requires outputs to be **both softmaxed AND argmaxed prior to the evaluation process**. Outputs should be in the format of **BxCxZxWxH**.
3. For results to be clinically acceptable, model has to be applied to the **entire depth/width/height of the patient**. (If you would like more details on how we did this for the original study, please reach out.)
4. If your template uses modules like nn-UNET as your base, **please ensure you build an api that enables evaluation using pytorch-lightning.**
5. **Please ensure that the order of classes used during training MATCHES THE ORDER GIVEN IN the ROIS list provided in utils.py.**
6. While developing your networks, make sure to explore and enjoy. Happy networking!

#### IMPORTANT NOTE

Your network should be trained end-to-end and be able to export all 34 ROI classes simultaneously, to mimic requirement(s) of clinical environments. Any questions can be sent to the challenge administrators via the participant slack channel!

### Base Tutorials running pre-packaged models coming soon...
### Inference Tutorial Using Colab Pro GPUs coming soon...

## 1. Setup Environment
More instructions [here](https://stackoverflow.com/questions/41274007/anaconda-export-environment-file).

    conda env create -f light.yml

## 2. Running WOLNET ensemble on custom dataset (no-finetuning required).

In order to run the pre-trained WOLNET network on your own dataset, certain aspects of our Lightning Module must be customized. Because we will be in testing mode, the weights of each model during inference will be frozen.

You can use the slurm templates (wol-oar-test-1.sh ... wol-oar-test-5.sh) as base bash scripts for inference. The only things that need to be edited in this file are:

1. Clone repo.
2. Update local path to model weights (Download them [here](https://drive.google.com/drive/folders/15yc--RfBaxuxKTEHaQ3OwbPdCqTOZdgH?usp=sharing)) in wol-oar-test-1.sh (repeat for other .sh test scripts)

3. Update local path to test.py file in wol-oar-test-1.sh (repeat for other .sh test scripts)
4. If you havn't done so already use custom pre-processing scripts (/process) or our [med-imagetools package](https://github.com/bhklab/med-imagetools) to extract raw data from .dicom images
5. Create a custom .csv with two columns - one for image and one for corresponding mask (if you have avaliable; if un-avaliable use image-path again in the second column). Easy to do in python:

        import pandas as pd
        imgs = glob.glob( "PATH_TO_IMG_FOLDER/*.nrrd")
        # if mask folder avaliable
        msks = [i.replace(PATH_TO_IMG_FOLDER, PATH_TO_MSK_FOLDER) for i in imgs]
        # create .csv we need to load same image twice (in same batch) during inference
        img = []
        msk = []
        for i, im in imgs:
            img.append(im)
            img.append(im)
    	    msk.append(msks[i])
    	    msk.append(msks[i])

       data = {'0':img, '1':msk}
       data = pd.DataFrame.from_dict(data)
       data.to_csv('path_to_csv')

 5. Once we have custom .csv we can run inference (change path in . Once predictions for each model have been exported, use process_masks3.py as a template to average the predictions.
 6. **Optional:** to compare results against ground truth data extracted for your own internal datasets, use notebooks provided.
 7. To establish run your own QUANNOTATE server for clinical acceptability testing of countours generated, please [clone our open-source repo](https://github.com/bhklab/quannotate).
 8. All files including model weights, ensemble predictions on each external dataset as well as pre-processed versions of each dataset can be found [here](https://drive.google.com/drive/folders/15yc--RfBaxuxKTEHaQ3OwbPdCqTOZdgH?usp=sharing).

## 3. Slurm Scripts

PyTorch Lightning has been specifically engineered to optimize training across large computing clusters governed by job handlers. In this study we use slurm commands to run the python script (connecting us to the right GPU nodes). These can be found in the /slurm folder. If you don't use slurm, these files can be run by using simple bash commands.

## 4. Figures

![Figure 1. Study Overview:  A general overview of the study design. A total of 582 patients met selection criteria having the same 19 OAR(s) delineated in their RTSTRUCT files. Data was extracted and used to train 11 different open source 3D segmentation networks. The 11 networks were selected as a subset of 60 studies proposing implementing image segmentation networks in a medical context. A subset of 29 studies did not release code along with their publication, and therefore, these models could not be directly assessed. Another 20 studies were removed because of one or more of the following: networks could not be tested with original configuration due to limitation of computational power, 2D networks that could not be converted to 3D convolutional scheme without modifying it’s architectural integrity, close similarity or overlap between other architectures, code released with the study was unmaintained and/or could not be integrated into PytorchLightning. The networks were ranked based on overall performance on a hold out test set of 59 patients across all 19 OARs. The top model was then chosen to be fine-tuned and used in a blinded clinical acceptability assessment conducted by 4 expert radiation oncologists on the open source Quannotate platform. ](https://lh3.googleusercontent.com/pw/AM-JKLW2Lp30kdmSflVdw-3Ea48OKP0iPEsnSyMzUFqucyuejpPrTezcSn0vg1ja1iau7KUWAnWdyL1-naaSe6CxOpykJQ0O4ewUmf7lxVjb64xy8pZIXlSvpLhfA_0M6OsBdcIiCFud0bNbYZ9XcxGVIXHamg=w965-h600-no?authuser=0)

**Figure 1. Study Overview:**  A general overview of the study design. A total of 582 patients met selection criteria having the same 19 OAR(s) delineated in their RTSTRUCT files. Data was extracted and used to train 11 different open source 3D segmentation networks. The 11 networks were selected as a subset of 60 studies proposing implementing image segmentation networks in a medical context. A subset of 29 studies did not release code along with their publication, and therefore, these models could not be directly assessed. Another 20 studies were removed because of one or more of the following: networks could not be tested with original configuration due to limitation of computational power, 2D networks that could not be converted to 3D convolutional scheme without modifying it’s architectural integrity, close similarity or overlap between other architectures, code released with the study was unmaintained and/or could not be integrated into PytorchLightning. The networks were ranked based on overall performance on a hold out test set of 59 patients across all 19 OARs. The top model was then chosen to be fine-tuned and used in a blinded clinical acceptability assessment conducted by 4 expert radiation oncologists on the open source Quannotate platform.

![Supplementary Figure 2. Overview of Clinical Acceptability Protocol, an open source web-based quality assurance tool from our previous study was modified for clinical acceptability testing of radiation therapy contours. For this analysis, 4 expert radiation oncologists were each given the opportunity to assess the acceptability of deep learning or manual ground truth contours in a blinded fashion. 10 ground truth contours paired with 10 deep learning generated contours for the same patient were extracted for each of the 19 OARs. A total of  380 3D contours were assessed by each observer. They were asked to rate acceptability on a 5 point scale taking the complete volume of the entire OAR contour into context. A rating of 4.0 and higher can be considered ‘acceptable’ in that no edits are required by the examining physician for planning purposes. They were also asked to ‘guess’ the observer that generated the contour (“Human”/ “Computer” / “I don’t Know”) before submitting their rating. Mean Acceptability Ratings were then calculated for each OAR, and analysis assessing correlation of acceptability with 6 different segmentation performance metrics was conducted. These metrics include 3D Volumetric Dice Overlap Coefficient, 95% Hausdorff Distance, Applied Path Length, False Negative Length, and False Negative Volume](https://lh3.googleusercontent.com/CxJb6Xruah0V1wOD9O1su-5mt69lHtmhK14IYe_oznW7PEYXtZGQnrr-3MjCORJX4-fHnQEwhUeFEhXq2jmGMdQznV4jZ4bdtVrdiqbhpf9qZzCkYEtAQ1CmawCO-vdl2eAGhMx9t7rWnWuPRCGjlWBQqH4n9e0j1I1hzjJl2YDa47Tgrbt09hdyGjgRqRWN-vxquTox6SqiAB2oiIaybWWCODz0smrhAgXkKYluF_dohK3DcKeQNYj01pP24iWMgb5getrjNpsCxdoTbiki7Fvzj3v_T0KWzv5idykLAXdoSgTNXqAKqy4owG51qX7rtVvsYLT7kczzob3cZujXXjPMCP_Uu4Iz1yTnCJjf0wq8NWCtHk5pvoVBE85LXenvfT_TEKdwUippJOYkA2UAa0ggL3nAZvZIQyZG9K6bPMCmKkdBkBStj-5qqraFYJ6EELbVawN2bhCtIw68jx3LrMrzAP2GPFXO3ULuR39obC7-UiJrX3rAEANSpWvtlXVZ8xaOm33APntD8qGkD5UTL6Ces1JUoHrb-0LuwE3LKbmAzn7GDeUAU0tr4vD8y971gc4nOV7BpQ3G8ZuMgwMot-dtVT88R-EVb0rhaF5QfchRzQSYVao7ERtP-uhG1mu5ZEYifB40ZECfcqEyan_LCDP1tsgy_84UaG_J3KNgPW9T1LLo5g6tff_k1RuBllJUox3668HFq8CeEXQdZ_Iw-FsIQw=w1719-h852-no?authuser=0)

**Supplementary Figure 2:** Overview of Clinical Acceptability Protocol, an open source web-based quality assurance tool from our previous study was modified for clinical acceptability testing of radiation therapy contours. For this analysis, 4 expert radiation oncologists were each given the opportunity to assess the acceptability of deep learning or manual ground truth contours in a blinded fashion. 10 ground truth contours paired with 10 deep learning generated contours for the same patient were extracted for each of the 19 OARs. A total of  380 3D contours were assessed by each observer. They were asked to rate acceptability on a 5 point scale taking the complete volume of the entire OAR contour into context. A rating of 4.0 and higher can be considered ‘acceptable’ in that no edits are required by the examining physician for planning purposes. They were also asked to ‘guess’ the observer that generated the contour (“Human”/ “Computer” / “I don’t Know”) before submitting their rating. Mean Acceptability Ratings were then calculated for each OAR, and analysis assessing correlation of acceptability with 6 different segmentation performance metrics was conducted. These metrics include 3D Volumetric Dice Overlap Coefficient, 95% Hausdorff Distance, Applied Path Length, False Negative Length, and False Negative Volume

![Supplementary Figure 7: Overview of Ensemble & WOLNET External Validation Ensembling by averaging the predictions of similar but distinct CNN models has proven to boost network performance while increasing a model’s generalizability potential. To minimize variance and spurious model predictions, we decided to use five random K-Fold training/validation splits to build a WOLNET segmentation ensemble. A set of 5 publicly available datasets were gathered for external validation of the WOLNET ensemble that span a total of 335 patients. Predictions were saved and performance metrics were extracted for overlapping OARs.](https://lh3.googleusercontent.com/pw/AM-JKLWhk2cZtKq5yOkWd1YN5rXcolozEKUEWGfeizB5-UnfW-6zdU3ZrNGTMBL9DbCq_D7sX5D5GG9kLL2rZY6rixfb8QjVkr_A84-MAlKS7EAmc1H-bEl3-Xng67G_nQK2H2QOpjLlfoYgBkDQ2Dt3wxT0-Q=w981-h533-no?authuser=0)

**Supplementary Figure 7:** Overview of Ensemble & WOLNET External Validation Ensembling by averaging the predictions of similar but distinct CNN models has proven to boost network performance while increasing a model’s generalizability potential. To minimize variance and spurious model predictions, we decided to use five random K-Fold training/validation splits to build a WOLNET segmentation ensemble. A set of 5 publicly available datasets were gathered for external validation of the WOLNET ensemble that span a total of 335 patients. Predictions were saved and performance metrics were extracted for overlapping OARs.

## Citation information

``` @article {Marsilla2022.01.15.22269276,
	author = {Marsilla, Joseph and Won Kim, Jun and Kim, Sejin and Tkachuck, Denis and Rey-McIntyre, Katrina and Patel, Tirth and Tadic, Tony and Liu, Fei-Fei and Bratman, Scott and Hope, Andrew and Haibe-Kains, Benjamin},
	title = {Evaluating clinical acceptability of organ-at-risk segmentation In head \& neck cancer using a compendium of open-source 3D convolutional neural networks},
	elocation-id = {2022.01.15.22269276},
	year = {2022},
	doi = {10.1101/2022.01.15.22269276},
	publisher = {Cold Spring Harbor Laboratory Press},
	abstract = {Deep learning-based auto-segmentation of organs at risk (OAR) holds the potential to improve efficacy and reduce inter-observer variability in radiotherapy planning; yet training robust auto-segmentation models and evaluating their performance is crucial for clinical implementation. Clinically acceptable auto-segmentation systems will transform radiation therapy planning procedures by reducing the amount of time required to generate the plan and therefore shortening the time between diagnosis and treatment. While studies have shown that auto-segmentation models can reach high accuracy, they often fail to reach the level of transparency and reproducibility required to assess the models{\textquoteright} generalizability and clinical acceptability. This dissuades the adoption of auto-segmentation systems in clinical environments. In this study, we leverage the recent advances in deep learning and open science platforms to reimplement and compare the performance of eleven published OAR auto-segmentation models on the largest compendium of head-and-neck cancer imaging datasets to date. To create a benchmark for current and future studies, we made the full data compendium and computer code publicly available to allow the scientific community to scrutinize, improve and build upon. We have developed a new paradigm for performance assessment of auto-segmentation systems by giving weight to metrics more closely correlated with clinical acceptability. To accelerate the rate of clinical acceptability analysis in medically oriented auto-segmentation studies, we extend the open-source quality assurance platform, QUANNOTATE, to enable clinical assessment of auto segmented regions of interest at scale. We further provide examples as to how clinical acceptability assessment could accelerate the adoption of auto-segmentation systems in the clinic by establishing {\textquoteleft}baseline{\textquoteright} clinical acceptability threshold(s) for multiple organs-at-risk in the head and neck region. All centers deploying auto-segmentation systems can employ a similar architecture designed to simultaneously assess performance and clinical acceptability so as to benchmark novel segmentation tools and determine if these tools meet their internal clinical goals.Competing Interest StatementThe authors have declared no competing interest.Clinical Protocols https://www.quannotate.com Funding StatementThis study was funded by Canadian Institutes of Health Research Project-Scheme for the Development and comparison of radiomics models for prognosis and monitoring (Haibe-Kains B, Hope A). Term: 02/2020-01/2023Author DeclarationsI confirm all relevant ethical guidelines have been followed, and any necessary IRB and/or ethics committee approvals have been obtained.YesThe details of the IRB/oversight body that provided approval or exemption for the research described are given below:A UHN institutional review board approved our study and waived the requirement for informed consent (REB 17-5871); we performed all experiments in accordance with relevant guidelines and ethical regulations of Princess Margaret Cancer Center. Only PM data used for network training was involved in the REB. External publicly available datasets were governed by individual REBs by its institution of origin.I confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) outside the research group so cannot be used to identify individuals.YesI understand that all clinical trials and any other prospective interventional studies must be registered with an ICMJE-approved registry, such as ClinicalTrials.gov. I confirm that any such study reported in the manuscript has been registered and the trial registration ID is provided (note: if posting a prospective study registered retrospectively, please provide a statement in the trial ID field explaining why the study was not registered in advance).YesI have followed all appropriate research reporting guidelines and uploaded the relevant EQUATOR Network research reporting checklist(s) and other pertinent material as supplementary files, if applicable.YesAll data and code produced for automated OAR segmentation are available online at github.com/bhklab/ptl-oar-segmentation/. All code produced for radiological imaging preprocessing and DICOM data extraction are available at github.com/bhklab/imgtools. All code produced for QUANNOTATE clinical acceptability testing interface are available online at https://github.com/bhklab/quannotate. https://www.github.com/bhklab/ptl-oar-segmentation https://www.github.com/bhklab/imgtools},
	URL = {https://www.medrxiv.org/content/early/2022/01/25/2022.01.15.22269276},
	eprint = {https://www.medrxiv.org/content/early/2022/01/25/2022.01.15.22269276.full.pdf},
	journal = {medRxiv}
}```
