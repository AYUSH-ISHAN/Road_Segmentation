# Road_Segmemtation

# Try to get output from the FCN and U-NETS and also try other models also. 



This repo contains the road segemntation from a high qulaity image using the various models, techniques and algorithms.
Changwon University Intern.
Approaches:

1. A fully loaded U-nets network.
2. Pretrained Encoder and Decoder Netwrok Part. COMPLETED link: - https://github.com/JunshengFu/semantic_segmentation <br>
3. If time permits try out various models which are used here: https://github.com/JunHyeok96/Road-Segmentation
4. Also look at the papers metioned in them


# Introduction:

# Model Architecture:
<p align = "center">
<img src ="./architecture.jpg" align = "center"/>
</p>
<br>
In the above shown model, the pretrained VGG-16 networks are used as encoder. The VGG_16 was trained on pretrained on ImageNet for classification. The 
pretrained weights can be found on the link - <a href = "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip">Udacity Self Driving Car</a>

# Files and Folders:

<ol>
  <li><B>Dataset folder : </B>This folder contains the dataset.</li>
  <li><B>Masked Dataset folder : </B>This folder has the dataset in the masked form. The masking was done manually.</li>
  <li><B>Final Showdown folder : </B>This folder has final outputs or predictions by the model.</li>
  <li><B>FCN folder : </B>This folder has all the codes for the model.</li>
</ol>
  

# Results on sample images:

# Results on sample videos:
