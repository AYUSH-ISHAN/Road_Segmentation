# Road_Segmemtation

# Try to get output from the FCN and U-NETS and also try other models also. 



This repo contains the road segemntation from a high qulaity image using the various models, techniques and algorithms.
Changwon University Intern.
Approaches:

1. For video perdictions , do this - >  https://github.com/lb5160482/Road-Semantic-Segmentation/
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

<table>
  <tr>
    <td align = "center"><B>IMAGE</B></td>
    <td align = "center"><B>MASK</B></td>
    <td align = "center"><B>OUTPUT</B></td>
  </tr>
  <tr>
    <td><img src = "./dataset/umm_road_1.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_1.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_1.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_5.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_5.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_5.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_10.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_10.png"/ height = "150", width = "250"></td>
    <td><img src = "./Final_Showdown/umm_road_10.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_20.png"/ height = "150", width = "250"></td>
    <td><img src = "./masked_dataset/umm_road_20.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_20.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_40.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_40.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_40.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_30.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_30.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_30.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/umm_road_1.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/umm_road_1.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/umm_road_1.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/forest_dataset/umm_road_4.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/forest_masked/umm_road_4.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/forest_final/umm_road_4.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/forest_dataset/umm_road_6.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/forest_masked/umm_road_6.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/forest_final/umm_road_6.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/forest_dataset/umm_road_66.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/forest_masked/umm_road_66.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/forest_final/umm_road_66.png" height = "150", width = "250"/></td>
 </tr>
  <tr>
    <td><img src = "./dataset/forest_dataset/umm_road_88.png" height = "150", width = "250"/></td>
    <td><img src = "./masked_dataset/forest_masked/umm_road_88.png" height = "150", width = "250"/></td>
    <td><img src = "./Final_Showdown/forest_final/umm_road_88.png" height = "150", width = "250"/></td>
 </tr>
  
 </table>




# Results on sample videos:

<img src = "./video_prediction.gif">
