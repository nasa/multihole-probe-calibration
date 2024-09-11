# 5 Hole Probe Calibration

# Files
## Training Code
Run `python 5HoleProbeTrain.py` to process the data and train the model. This will create files in the *checkpoints** folder with the data and model. There should be a model for each probe. 

## Inference
The example [inference](5HoleProbeInference.py) shows how to predict the pitch, yaw, mach, and Cpt from measurement data. 
To run the code call `python 5HoleProbeInference.py` or import the file into your script then call `predict_data`. 

