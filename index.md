# Motion Detection using MLP

https://user-images.githubusercontent.com/34931736/177087799-d6a4b794-4406-4e4a-a457-b05bf245a627.mp4


## Aim
To detect primary motions (Shield, Grenade, Reload, Logout and Non-Gestures) using data from Gyrometers and Accelerometers from MPU6050. 

## Overview
The task of Hardware AI is to provide a model to predict the required gestures during the game. With Ultra96’s Zynq processing system, we can make use of the FPGA to program an ML model to predict either a Shield, Reload or Grenade. This section explores the ways that the model was implemented, in terms of training, FPGA implementation and optimizations, and also the challenges faced when implementing the model.

## Ultra96 Synthesis and simulation setup

### Synthesis
There were multiple ways that the synthesis of the AI model can be done on Ultra96, The initial implementation was to synthesise the model using a HDL design on vivado. This however would be more complicated and sub-optimal. This is because optimizations has to be made manually and HDL code logic is harder to implement as compared to higher level languages like C++ or Python. 

Another option was hls4ml. This tool is easier to use as compared to vivado HDL as the tool will implement the  Python scripts onto the FPGA interface. However, the result may not be entirely accurate as the conversion may not be done properly. 

Hence the last option, vivado HLS was the best option for our case. We only need to implement a C++ script for the feedward of the model, and then vivado HLS will synthesise the model into an IP Package that can be used in generating the block diagram and bitstream for the FPGA.  Vivado HLS also allows for code optimization. This allows us to reduce the latency when predicting.

### Simulation Setup
The initial report suggested that Vitis could be used to set up the FPGA. However, when implementing we found that this was not possible as we cannot access the FPGA remotely due to the restrictions by NUS’s firewall.

An alternative to programming the bitstream onto the FPGA would be using Pynq’s library.
This implementation allows us to ‘bypass’ the firewall issue as we can access the Ultra96’s jupyter notebook through SSH tunnelling and subsequently create the necessary python scripts to implement the bitstream onto Ultra96’s FPGA. The scripts can then be used by External Communications to integrate the AI model with the visualizer. 

Python Productivity for Zynq, also known as Pynq, offers us a platform to integrate our generated bitstream from vivado onto the Zynq’s processing system using python scripts. It offers a myriad of libraries, of which, one of them is called Overlay that allows for bitstream implementation on the FPGA. We only need to provide the bitstream, the hardware handover file and the block design file. Pynq will then program the FPGA according to the block diagram that is used to generate the bitstream, and the associating IP in the block diagrams can then be accessed just by calling its name. 


For example, we used a DMA in our implementation and the block diagram names the DMA as axi_dma_0 for player 1 and axi_dma_1 for player 2. This can be seen in the images below.

![image](https://user-images.githubusercontent.com/34931736/177084524-71a4b792-1e0d-4679-aa94-1f6db12528da.png) ![image](https://user-images.githubusercontent.com/34931736/177084511-60c3b5fd-4182-4740-a5ff-f7ba8c34ce7d.png)

This can then be accessed in Pynq as follows:

![image](https://user-images.githubusercontent.com/34931736/177084489-98fa4270-3abd-4ed9-b28f-1a2a724074a7.png)

The overlay can be called using its constructor. The path to the .bit file is set as an argument for the overlay’s constructor. As the overlay will recursively check for the hardware handover and block design file in the folder, the other two files must also be placed in the same directory. The DMA is then accessed by calling the respective DMA’s name as seen in the image above by setting dma_send and dma_rcv to an array of two DMAs.  The two variables, dma_send and dma_rcv, act as the input and output lines for the DMA, where the features for the prediction can be sent through the dma_send line and the prediction can be accessed from the dma_rcv line. More details on  the Predictor will be explained in the section for realisation of the model onto the FPGA board.

## Neural network 

### Initial Neural Network

In the initial report, the proposed model was CNN. However, upon embarking on the design, we realise that using CNN may not be the optimal solution. CNN requires the dataset to have spatial information. However the data that we will be receiving in real time will be the 6 attributes , acceleration_x, acceleration_y, acceleration_z, gyro_x, gyro_y, gyro_z , instantaneously. To represent these data spatially, we would have to keep a 2D image of the data which would require more computational effort. Furthermore, CNN would require convolutional layers before using a dense layer which could use up more resources in the FPGA when the model is being realised, as compared to other models like MLP or DNN where only the dense layers are required. 

The next approach was to implement a Multi-Layer Perceptron, which acts as a dense layer. Comparing this with CNN, we would require certain features to be extracted beforehand. This is because  CNN has the convolutional layers to do the feature extraction for us, hence, with the absence of the convolutional layers, we have to preprocess the incoming data to extract certain key features before putting it into the model. Upon data collection, and visualisation of data, we realised we would require an MLP of 3 hidden layers and one output layer. The model can be seen below.

![image](https://user-images.githubusercontent.com/34931736/177084455-154d79dd-bc34-42ef-bdaa-37c78ba85341.png)


The sizes of the hidden layers are 512, 256, and 64 respectively. The output layer is of size 5 as we would be predicting either a Shield, Grenade, Reload, Logout or No Gestures.  Hence the total number of neurons is 837. For each layer, we would implement a RELU. The reason for RELU is because it can be easily implemented in the FPGA by only reassigning the negative discharge of the perceptron as 0 [14]. The output layer does not have any activation function, as we would want to reduce the computational latency. Since the prediction can be obtained by finding the argmax of the output layer’s result, then no further activation function is required.

## Training and Testing

The data collected for the design was only from our team members. We planned to collect the data from more individuals but due to time constraints, we were not able to do so. For the data collection phase, we collected the data using the Bluepy integrated script provided by Internal Communications. This allows for the data to be collected using bluetooth, which allows for more representative actions as compared to the real-time situation. The raw values of the attributes are scaled down to signed 8 bits, so that the computational power needed by the FPGA can be reduced.  

The data is collected in a one second window, with each action having a specific starting gesture. The individuals would then have to complete the gestures in that window, and the data is then extracted into a text file. From the text files, we can then visualise the data to check for any visible patterns. This would allow for better features extraction as we can see any visible correlations, or minimum or maximum outputs. The visualisations can be found on Figure 29.

From the visualisation, we can see that there are specific peaks, namely that most attributes usually hit a value of 127 or -127. There is also a pattern where certain positive peaks usually come before the negative peaks. Some of the attributes are also negatively correlated. All of which are strong features to determine an action. We conducted a feature extraction using statistical variables. The features are minimum value, maximum value, average value, median value, mean absolute deviation, standard deviation, interquartile range, entropy, root mean square,  range, average magnitude and median absolute deviation of frequencies obtained through FFT,  time difference between positive and negative peaks and correlation between every two distinct attributes. These bring a total of 93 features, where the correlation between every two distinct attributes add up to 15 features, while the other 13 statistical variables on each attribute add up to 78 features. These 93 features then make up the input layer as seen in the previous figure.

Before training, we conducted a correlation check of each feature of each datapoint in the dataset with the output label. We realised that there is a high correlation with most features and an exceptionally strong correlation between the output labels and features extracted from  acceleration_x.  A table on the correlation of the features with the output labels is found on Figure 30.

The dataset is then trained using SGD with batch size 1 and 50 epochs, with 90% of the dataset as training set while the rest as test set. The reason for a very small batch size is because we collected a small size of dataset of about 150 data points per gesture. A small batch size allows the model to be more sensitive to the pattern of the training set. However, this also means that the model will be sensitive to the noise of the training set, in which we reduced this by setting a very low learning rate of 0.000008 and a momentum of 0.9 to allow for a steady convergence of the model’s target concept. We implemented a scheduler for the learning rate to reduce at the next epoch when the learning rate meets a plateau. This prevents overfitting as it ensures that the learning rate is reduced when validation loss does not decrease.  The entire training is done on 10 K-fold to ensure that the model is not overfitting and that it is exposed to the most optimal training and test set. After training, we picked the K-fold with the least validation loss,and extracted the model’s weight from there.

Since the weights in Pytorch are in floats, we would have to convert the weights to integer for easier computation in the FPGA. The weights from Pytorch are scaled by 1000 and then converted into integers by rounding it up to its nearest integer, The weights are then saved and copied over to the HLS code. We trained the dataset for both models on only 3 teammates. For testing, we then tested it on the other two team members. The model showed a fairly good prediction for the other two teammates who are not part of the training, hence showing an absence of overfitting. 

<figcaption align = "center"><b>Visualisation of datasets based on attributes and actions</b></figcaption>

![image](https://user-images.githubusercontent.com/34931736/177084933-bf97e053-a774-4ecc-9a5c-aecc57539e8d.png)

<figcaption align = "center"><b>Correlation between output labels and extracted features of each attribute</b></figcaption>

![image](https://user-images.githubusercontent.com/34931736/177085232-c8c4a8fc-4a58-4505-87d6-6c0ffd40aca0.png)

<figcaption align = "center"><b>Confusion Matrix of Software Training (Output By Target).</b></figcaption>

![image](https://user-images.githubusercontent.com/34931736/177085299-c1d71105-c5d7-4921-887d-18120c41c1d4.png)

## Realisation of model on the FPGA board

### Block Diagram

The block diagram can be seen on page 36. We used 2 DMAs for each model. The DMA allows us to send in input from the python scripts and to read the prediction from the model. The DMA are connected to the Zynq’s processing system using a smart interconnect, which allows for optimal connection and switching of lines.  We used two models because the IMU’s sensitivity is largely different for each player, hence a differently trained model is required for each player. Therefore we came up with two IP packages for the HLS models and used 2 DMA to read from and write to the model when the IMU sends in data.
 
![image](https://user-images.githubusercontent.com/34931736/177085815-0c64a44f-0e0a-4184-8248-9b0258cc891e.png)

### Feedforward implementation
The FPGA implementation only requires a feedforward because the model has already been trained, and no backpropagation is required to reinforce learning. We only require the prediction from the model and computation of validation loss is not possible during live sessions, hence, only feed forward is required. 

The feedforward equation is as follows:

![image](https://user-images.githubusercontent.com/34931736/177085889-f066a930-e414-4779-8522-6d0be380cc7a.png)

As seen in the equation above,W is the weight matrix, x is the input matrix for the layer, and b is the bias matrix. Since there exists 3 hidden layers and one output layer, there will be 4 weight matrices needed to compute Z for each layer and 4 bias matrices.  The computation for the prediction is done by the following steps. 

Feed Forward Algorithm:
1.	Compute Z1 :  Z1 = W1 * x1  + b1,  W1 is of size 512 (hidden layer 1 size) by 18 (input layer size), x1 is the 18 input features and b1 is the bias matrix for hidden layer 1 and is of the size 512 by 1. Therefore the resulting matrix Z1 is 512 by 1.
2.	Compute ReLU for Z1:  ReLU function is  f(x) = max(0, x). Therefore, for each value of z in Z1, if z < 0 assign to 0 else keep the value that it has. 
3.	Compute Zj and ReLU:  Zj = Wj * Zj-1 + bj,  for j = 2 to 4 where 4 is the output layer. Wj is the weight , bj is the bias for the respective layers and Zj-1 is the output of the previous layers. Apply the same ReLU logic for each element of Zj, for j =2 and j =3,  before feeding inputs to the next layer.
4.	At j = 4, when Z4 is computed, pick the argmax of the array. The prediction is the argmax of the array.

### Ultra96 Synthesis

The following would be the steps taken to create a HLS synthesis of the model, followed by setting it up on Ultra96.

•	Initialise Weights:

![image](https://user-images.githubusercontent.com/34931736/177086011-a7042848-f5ba-43bf-9216-55584a1b37e3.png)

•	Define pragmas for interface to allow AXI communications:
 
![image](https://user-images.githubusercontent.com/34931736/177086064-d2080beb-14f8-4c34-8ea3-c75623a82c30.png)

•	Define scale used to convert floating points of weights and biases into integers:

![image](https://user-images.githubusercontent.com/34931736/177086089-f49ba9c6-8135-4d79-b72f-99e409ce0f0e.png)
 
•	Read data through S_AXIS into an input array. **Note: S_AXIS last is not set to 1 because both Masters and Slaves know the input size. Only need to assign data in the AXI line into the array

 ![image](https://user-images.githubusercontent.com/34931736/177086114-f58dd0e9-9070-4687-9e30-338f28b08ed7.png)

•	Implement feed forward by doing matrix multiplication of weights with input using row major order, and adding in bias at the end of every row. The resulting sum would then be scaled down by the scale factor. Note: Row for w1 is the length of input vector, therefore when deriving modulo of word count from input layer size, if the value is lesser than input layer by 1, the word count is at the end of the row in the weight matrix.

![image](https://user-images.githubusercontent.com/34931736/177086149-a966b6ff-93da-46aa-a680-288e544e1f88.png)
 
•	Implement feed forward for respective layers. Note that the input value for each layer is the resultant matrix from previous layer

![image](https://user-images.githubusercontent.com/34931736/177086171-648bdc24-9ed0-41cd-9138-65b9abee6487.png)
 
•	Implement feed forward for the output layer. Note that no ReLU is used for the output layer.

![image](https://user-images.githubusercontent.com/34931736/177086199-8f7971b3-041d-4826-8a4e-7a63a916342f.png)
	 
•	Obtain prediction by finding the argmax of the output array

 ![image](https://user-images.githubusercontent.com/34931736/177086205-58aea552-d56e-409c-83f8-4d7d01dc116a.png)

•	Write output to the M_AXIS line. Note that the last bit is set to 1 to indicate that the last output is written into the M_AXIS line so that DMA need not wait for any further output

 ![image](https://user-images.githubusercontent.com/34931736/177086216-5b97bb03-1907-44e6-8f87-2f00072d068e.png)


•	Synthesise and export the HLS into the solution folder in Vivado HLS

•	Resulting HLS code will appear in the solution folder.


### Importing IP Core into Vivado

The HLS code can be imported into the block diagram as an IP package. This can be done by adding the solution folder that was generated in the previous step into vivado’s IP repo. Ultra96 Zynq’s IP has to be imported inside as well, which can be retrieved from the board repository in Vivado if it cannot be found by default. With the necessary IPs imported, we are only left with adding in the DMA as a communication interface. When the block diagram is ready, generate bitstream and export the bitstream file (.bit), hardware handover (.hwh) and block diagram file (.tcl) to Ultra96. Ensure that the files are in the same directory.

### Ultra96 Setup

With the files above, we can use Pynq’s library to create an overlay and program the FPGA in the ultra96 with bit file. 

![image](https://user-images.githubusercontent.com/34931736/177086278-11c971d6-2adf-4ffe-b9f2-97738550dcdc.png)

The DMA can be accessed by calling the names used in the block diagram. We have a dma_send and dma_recv as different lines for writing and reading from the DMA.  The features are sent through dma_send while the predictions are read through dma_recv. 

We also created input and output buffers to store the input and output data that is to be sent or received from the DMA. When we are sending the input to the DMA, we will write the input into the input buffer, and subsequently read it from the output buffer. This can be seen in the image below. 

![image](https://user-images.githubusercontent.com/34931736/177086294-ffe656b2-1257-4e69-acf3-c461055fe349.png)

We set an interval for the prediction to be retrieved from the model. The interval value is 7. This means that at every 7th incoming data from External Communications, the model will output a value for prediction. This is then stored in another buffer to calculate the accumulation of consecutive predictions. If the number of consecutive predictions meets the threshold for a certain gesture, the model will then send that prediction to External Communications. This can be seen in the image below, for the prediction of Player 1’s Gesture. 

![image](https://user-images.githubusercontent.com/34931736/177086312-fc333228-9d9a-4cc9-8317-1ed925fa7dba.png)


Note that in this function, we set allow_pred to False when prediction is sent to External Communications. This is to block off any predictions for one second. When the timer reaches 77, which is equivalent to 1 second given a data rate of 77Hz,  allow_pred is then set to true to allow for subsequent predictions. 

We also have a testbench, predictor.py, created for the FPGA implementation of the models. This is done by using the features that we used for training, and pushing in each row of features into the predictor’s push to dma function. A confusion matrix was then obtained.

![image](https://user-images.githubusercontent.com/34931736/177086328-dd7f1513-c165-47aa-bfc9-4758e33bb8b5.png)

## Design’s timing, power and area with the given dataset.

### Timing

The timing has been reduced by using optimizations such as DMA, and predicting in intervals. This allows the coprocessor to be used only when needed (in terms of intervals) and does not use up the coprocessor’s computation efforts in remembering the inputs in terms of DMA.  The timing is also optimised by using integer computation instead of floating points. This trades off accuracy at the expense of timing. However, the confusion matrix did not differ much from the software training, hence this optimization did not affect the accuracy as much. The latency for prediction is 0.004468 seconds. However, when taking into account real-time predictions, and latency with respect to data coming from Internal Communications, and External communications sending to Eval Server, the time taken was on average 4 seconds, inclusive of human reaction time. 

### Area

The area of the design could not be optimised as the model was large and most of the BRAM was used  even without any hardware code optimization. With code optimization such as loop unrolling or pipelining, the BRAM utilisation can increase up to more than 50% per model, which in total would exceed the available resources. 

![image](https://user-images.githubusercontent.com/34931736/177086419-19831fdb-016c-4a31-8076-842435e3576e.png)

The total BRAM utilisation for both models is 56% without any hardware code optimization.

### Power

![image](https://user-images.githubusercontent.com/34931736/177086455-f780a506-7e4b-47db-b9ad-0b6b1a8a0675.png)

The power is used up mainly dynamically, of which, most of it is used up by the PS logic. Statically, most of the power is used up by the PL logic. The dynamic power is 1.772 W which is the amount of power used when the design is switching between High and Low in the datapath and clock values.  The total static power is 0.321W which means this is the minimum amount of power the design needs in order to operate, even when no action is done.  The power consumption was optimised by ensuring that the model is only used at every 0.1s or rather at every 7th incoming data. This means that the prediction is made 11 times in 1 second, given a data rate of 77 incoming data per second. Therefore prediction is made at 0.142% of the entire live session. Hence the dynamic power is used at every 0.1s, making the total power 2.093 W at 0.142% of the live session while the remaining time, the power consumption is at 0.321W.






