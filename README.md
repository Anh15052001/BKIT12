# BKIT12
 BK_Naver competition
1. Segmentation
- The first you can read README.md file in each folder and following it to prepare run
- After you done, follow this  tutorial
    1.1 Run file training to train: 
        python train.py
    After that, you will have file checkpoint model.h5 and you can use it to predict or eval data.
    
    1.2 To predict, you need model pretrained : model.h5 in folder files or you can download here: https://drive.google.com/file/d/1BFVYCdWympC9TvhjklyxuMQRkWPOzL2_/view?usp=sharing
        python predict.py
    After that, you will have data preprocess for gesture

    1.3 To get result about task1 : segmentation body, you can run file predict with model pretrained and data test
        python eval.py
    
    After that, result will in folder contains mask predict
