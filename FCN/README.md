# Training :

Before starting the training process make sure your directory of the model looks similar to this ->

            .
            ├── data
            │   ├── data_road
            │   │   ├── testing
            │   │   │   ├── calib
            │   │   │   └── image
            │   │   └── training
            │   │       ├── calib
            │   │       ├── sem_image
            │   │       └── image
            │   └── vgg
            │       └── variables
            ├── model
            ├── __pycache__
            └── run
          
If the 'calib' directory is not there, then no need to worry. It still works !

o train the model run the following command in your terminal -->

            python run.py

You can tune the epochs and batch size in the file itself.

# Predictions:
For predictions just turn the training to "False" in this <a href="https://github.com/AYUSH-ISHAN/Road_Segmentation/blob/main/FCN/run.py#:~:text=if%20__name__%20%3D%3D%20%27__main__%27%3A-,training_flag,-%3D%20True%20%20%20%23%20True%3A%20train">file</a>.

And then rerun the command -->

           python run.py
