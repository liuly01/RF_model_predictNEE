We built two separate models for the datasets before and after performing data augmentation and scale optimization, and we required the original data to be identical in the training and validation sets except for the augmented data (RF_aug_model2.m and RF_exaug_model1.m) and subsequently applied to generate global products spanning from 2001 to 2022 (predict_globalNEE.m). The NEE results were then converted to NEP through area calculation (NEE_NEP.m).The code requires Matlab version 2014a or higher.

Using the GEE platform in conjunction with the geemap tool for data pre-processing and site-level extraction and download of code is also available.
