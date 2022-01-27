# Training ->

Before starting the training process make sure your directory of the model looks similar to this ->

              .
            ├── checkpoints
            ├── data
            │   ├── data_road
            │   │   ├── testing
            │   │   │   ├── calib
            │   │   │   └── image
            │   │   └── training
            │   │       ├── calib
            │   │       ├── sem_image
            │   │       └── image_2
            │   └── vgg
            │       └── variables
            ├── image_predicions
            ├── __pycache__
            └── video_predictions




If the 'calib' directory is not there, then no need to worry. It still works !

To train the model run the following command in your terminal -->

             python combo_run.py

You can tune the epochs and batch size in the file itself.

# Image Predictions ->

To start the model to predict your image, turn this <a href = "https://github.com/AYUSH-ISHAN/Road_Segmentation/blob/621e1b68aedc96cc703d4127c66e6254d69866f9/FCN_combo/combo_run.py#L155">flag</a> <B>OFF</B>

Then, again rerun the file.

# Video Predictions ->

To predict from videos or real, just adjust some parameters <a href = "https://github.com/AYUSH-ISHAN/Road_Segmentation/blob/621e1b68aedc96cc703d4127c66e6254d69866f9/FCN_combo/combo_test.py#L10">here</a>.

If you want to save the predictions of your video, then turn this <a href = "https://github.com/AYUSH-ISHAN/Road_Segmentation/blob/621e1b68aedc96cc703d4127c66e6254d69866f9/FCN_combo/combo_test.py#L6">flag</a> <B>ON</B>.<br>
For simple preview turn this <B>OFF</B>
