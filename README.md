# MotionDetection


Getting started on AI

	1) Ensure that .tcl,.bit and .hwh files are in the same directory as predictor2.py
	2) Run predictor.py as a testbench to get a gauge of the model's accuracy based on the extracted 		features saved in a csv file
	3) The model is instantiated using the predictor2's class constructor, and predictions can be made 	when a tuple of 6 attributes are sent to it (acceleration_x,acceleration_y , acceleration_z, gyro_x, 	gyro_y, gyro_z)

datasets

This contains all the datasets used for training. The csv files are the training sets for preliminary weeks, while files in "new_data" folder are for player 1's model while files in "new_data2" are for player 2's model.
CSV files should contain the 93 attributes that are extracted from the raw data. 

The 93 attributes are:"min ax", "max ax", "mean ax", "median ax", "mad ax", "sd ax", "iqr ax", "entropy ax", "rms ax", "range_ax", "avg magf ax", "madf ax", "tdiff ax", "min ay", "max ay", "mean ay", "median ay", "mad ay", "sd ay", "iqr ay", "entropy ay", "rms ay", "range_ay", "avg magf ay", "madf ay",  "tdiff ay","min az", "max az", "mean az", "median az", "mad az", "sd az", "iqr az", "entropy az",  "rms az", "range_az","avg magf az", "madf az",  "tdiff az","min gx", "max gx", "mean gx", "median gx", "mad gx", "sd gx", "iqr gx", "entropy gx",  "rms gx", "range_gx", "avg magf gx", "madf gx",  "tdiff gx", "min gy", "max gy", "mean gy", "median gy", "mad gy", "sd gy", "iqr gy", "entropy gy", "rms gy", "range_gy", "avg magf gy", "madf gy",  "tdiff gy", "min gz", "max gz", "mean gz", "median gz", "mad gz", "sd gz", "iqr gz", "entropy gz", "rms gz",  "range_gz","avg magf gz", "madf gz",  "tdiff gz",  "corr ax ay", "corr ax az", "corr ax gx", "corr ax gy", "corr ax gz", "corr ay az", "corr ay gx", "corr ay gy", "corr ay gz",  "corr az gx", "corr az gy","corr az gz",  "corr gx gy", "corr gx gz", "corr gy gz"


fpga

this contains all the files required to program the FPGA in ultra96 and also the python script used for the prediction of gestures. predictor.py is a testbench script used to calculate the average accuracy of the model in the fpga based on the extracted features of the training set while predictor2.py is the actual script used for prediction in real time. Ensure that .tcl,.bit and .hwh files are in the same directory as predictor2.py

hls 

this contains all the c++ code used for the model's high level synthesis. To synthesize the HLS, simply run the synthesis button on vivado hls and then followed by export to rtl when the synthesis is complete. Each c++ code must be in a separate solution folder in vivado hls.

ipynb

this contains all the google colab notebooks that are used to train the model using pytorch. Model 1 and 2 are based on week11's notebook but with differing datasets. When the training is complete, the weights can be copied over into the hls code. 

vivado

this contains only the block diagram for the model. The generated bitstream(.bit), hardware handover (.hwh) and the block diagram script (.tcl) can be found in the fpga folder. When creating the design for the models in vivado, the design should follow the block diagram. The HLS can be imported by adding the solution folders into the IP repo, and importing them as an IP package. With the 2 DMA setup, the model can predict two players' gestures. Ensure that the MM2S of the DMA is connected to model's S_AXIS and S2MM is connected to model's M_AXIS



