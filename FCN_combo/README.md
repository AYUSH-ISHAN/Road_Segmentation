# Training ->

Before starting the training process make sure your directory of the model looks similar to this ->

              .
            ├── checkpoints
            ├── data
            │   ├── data_road
            │   │   ├── testing
            │   │   │   ├── calib
            │   │   │   └── image_2
            │   │   └── training
            │   │       ├── calib
            │   │       ├── gt_image_2
            │   │       └── image_2
            │   └── vgg
            │       └── variables
            ├── image_predicions
            ├── __pycache__
            └── video_predictions




If the 'calib' directory is not there, then no need to worry. It still works !

To train the model run the following command in your terminal -->

             python video_run.py

You can tune the epochs and batch size in the file itself.
