# Training ->

Before starting the training process make sure your directory of the model looks similar to this ->

              .
              ├── checkpoints
              └── data
                  ├── data_road
                  │   ├── testing
                  │   │   ├── calib
                  │   │   └── image_2
                  │   └── training
                  │       ├── calib
                  │       ├── gt_image_2
                  │       └── image_2
                  └── vgg
                      └── variables


If the 'calib' directory is not there, then no need to worry. It still works !

To train the model run the following command in your terminal -->

             python video_run.py

You can tune the epochs and batch size in the file itself.

# Predictions: 

First make sure you have entered the directory to your video file correctly in this <a href ="https://github.com/AYUSH-ISHAN/Road_Segmentation/blob/2c481c3d0600debfb93531b9b027805d4c58f4d4/FCN_video_frame/video_test.py#L15">file</a>.<br>
To make the predictions, just run the following command.

            python video_test.py
