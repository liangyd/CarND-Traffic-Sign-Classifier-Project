# **Traffic Sign Recognition** 

## Writeup

---

**Traffic Sign Recognition Project**

The goals of this project are the following:

* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the feature on each layer of the network architecture
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report-images/image1.jpg 
[image2]: ./report-images/origin.jpg 
[image3]: ./report-images/gray.jpg 
[image4]: ./report-images/scale.jpg 
[image5]: ./report-images/translation.jpg 
[image6]: ./report-images/rotation.jpg 
[image7]: ./report-images/brightness.jpg 
[image8]: ./report-images/raw_data.jpg
[image9]: ./report-images/augmented_data.jpg
[image10]: ./report-images/CNN.jpg
[image11]: ./report-images/new_image.jpg
[image12]: ./report-images/prediction.jpg
[image13]: ./report-images/origin_visual.jpg
[image14]: ./report-images/conv1.jpg
[image15]: ./report-images/conv2.jpg



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup  that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/liangyd/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows the data distribution in the training dataset, validation dataset and test dataset. The images with respect to each label are not evenly distributed in the datasets. For example, in the training dataset, there are 2010 images for "speed limit(50km/h)" but 180 images for "speed limit(20km/h)" and "dangerous curve to the left". 

![alt text][image1] 


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color does not impact the traffic sign classification. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]  ![alt text][image3]


I decided to generate additional data because the images with different labels are not evenly distributed in the dataset. To add more data to the the data set, I used the techniques such as scaling, translation, rotation and brightness. ConvNet has built-in invariance to these small changes. Adding them will yield more robust learning to potential deformations in the test set. 

Here is an example of an original grayscale image and a scaled image:

![alt text][image3] ![alt text][image4]

Here is an example of an original grayscale image and a translated image:

![alt text][image3] ![alt text][image5]

Here is an example of an original grayscale image and a rotated image:

![alt text][image3] ![alt text][image6]

Here is an example of an original grayscale image and a brighter image:

![alt text][image3] ![alt text][image7]

I combined all the above-mentioned transformation techniques to create new data.

As a last step, I normalized the image data so that the data has mean zero and equal variance. Normalized values make the optimization in the neural network easier.


The augmented dataset has 129000 images, 3000 images for each class. The distribution before and after the augmentation is shown below:

Original Dataset (data size: 34799)

![alt text][image8]

Augmented Dataset (data size: 129000)

![alt text][image9]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Based on the multiscale convolutional network in Yann LeCun's paper [Traffic Sign Recognition with Multi-scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I built the network architecture shown below

![alt text][image10]

The model consists of the following layers: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image   			| 
| Convolution#1	     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU#1				|												|
| Max pooling#1	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution#2  		| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU#2				|           									|
| Max pooling#2	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| Flatten#1				| outputs 400									|
| Convolution#3	     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU#3				|           									|
| Flatten#2				| outputs 400    								|
| Full Connection#0		| inputs: Flatten#1 and Flatten#2, outputs:800	|
| Full Connection#1		| output: 43									|
| Output				|     											|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer which is stochastic gradient descent procedure with adaptive momentum estimation. The loss function is the average cross entropy of the softmax probability and one-hot encoding values. I chose the batch size of 128 and trained the model with 10 epochs. I set the learning rate as 0.001. I also used the dropout technique to reduce overfitting. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* validation set accuracy of 0.963
* test set accuracy of 0.945


At first, I used the LeNet-5 model which had an accuracy of 0.91. I tried to tune the batch size and epoch and modified the image preprocessing. The augmentation dataset in the preprocessing was helpful, but I still could not get an accuracy higher than 0.93.

Thus, I changed the network architecture based on the paper "Traffic Sign Recognition with Multi-scale Convolutional Networks" and built a multiscale convolutional network. Then, I got an accuracy above 0.93. 

I noticed that there was a high accuracy on the training set but low accuracy on the validation set, which means the model is overfitting. Thus, I added a dropout procedure to reduce overfitting and got an accuracy of 0.96.

I found out that the accuracy converged to a constant after 7 or 8 epochs. I did not get a better accuracy with a large epoch value. Thus, I used 10 epochs in the training. 


### Test a Model on New Images

#### 1. Choose German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are  German traffic signs that I found on the web:

![alt text][image11]

Some images may not be easy to classify because they have different brightness and backgrounds.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image12]



| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work    			| Road Work 									|
| Speed Limit(60km/h)	| Speed Limit(60km/h)							|
| No Vehicles      		| No Vehicles					 				|
| General Caution 		| General Caution      							|
| Turn left ahead		| Turn left ahead     							|
| Speed Limit(20km/h)	| Speed Limit(60km/h)      						|
| Yield					| Yield			    							|

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This accuracy is almost the same as the accuracy in the previous test.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five softmax probabilities were

Probability: 99.99985695 % (Stop)

Probability: 0.00008309 % (Keep right)

Probability: 0.00004642 % (No entry)

Probability: 0.00000995 % (Yield)

Probability: 0.00000881 % (Priority road)


For the second image, the model is relatively sure that this is a road work sign (probability of 0.99), and the image does contain a road work sign. The top five softmax probabilities were

Probability: 100.00000000 % (Road work)

Probability: 0.00000000 % (Children crossing)

Probability: 0.00000000 % (Road narrows on the right)

Probability: 0.00000000 % (Slippery road)

Probability: 0.00000000 % (Beware of ice/snow)


For the third image, the model is not quite sure if it is a speed limit(50km/h) or speed limit(60km/h). It gives a higher probability to speed limit(60km/h), which is the correct answer.

Probability: 56.30338788 % (Speed limit (60km/h))

Probability: 43.39033067 % (Speed limit (50km/h))

Probability: 0.23330173 % (Children crossing)

Probability: 0.04509717 % (Speed limit (20km/h))

Probability: 0.01745929 % (Road work)


For the forth image, the model is relatively sure that this is a no vehicle sign (probability of 0.92). The top five softmax probabilities were

Probability: 91.99331403 % (No vehicles)

Probability: 3.60334478 % (Priority road)

Probability: 2.69110594 % (Speed limit (50km/h))

Probability: 1.32278847 % (Speed limit (30km/h))

Probability: 0.17277919 % (Roundabout mandatory)


For the fifth image, the model is relatively sure that this is a general caution sign (probability of 0.99). The top five softmax probabilities were

Probability: 100.00000000 % (General caution)

Probability: 0.00000000 % (Pedestrians)

Probability: 0.00000000 % (Traffic signals)

Probability: 0.00000000 % (Road narrows on the right)

Probability: 0.00000000 % (Right-of-way at the next intersection)

For the sixth image, the model is relatively sure that this is a turn left sign (probability of 0.99). The top five softmax probabilities were

Probability: 100.00000000 % (Turn left ahead)

Probability: 0.00000000 % (No vehicles)

Probability: 0.00000000 % (Keep right)

Probability: 0.00000000 % (Bumpy road)

Probability: 0.00000000 % (No passing)


For the seventh image, the model is relatively sure that this is a speed limit(20km/h) sign (probability of 0.99). The top five softmax probabilities were

Probability: 99.36640263 % (Speed limit (20km/h))

Probability: 0.63353386 % (Speed limit (30km/h))

Probability: 0.00005586 % (Speed limit (50km/h))

Probability: 0.00000264 % (Speed limit (100km/h))

Probability: 0.00000099 % (Speed limit (120km/h))

For the eighth image, the model is relatively sure that this is a yield sign (probability of 0.99). The top five softmax probabilities were

Probability: 100.00000000 % (Yield)

Probability: 0.00000000 % (Speed limit (50km/h))

Probability: 0.00000000 % (No vehicles)

Probability: 0.00000000 % (Priority road)

Probability: 0.00000000 % (Speed limit (30km/h))


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I took the following image ("turn left ahead") as an example.

![alt text][image13]

The output from the first conv layer is :

![alt text][image14]

The output shows that the network uses the left arrow shape and the round circle as classification criteria.


The output from the second conv layer is :

![alt text][image15]

