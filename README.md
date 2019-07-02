# Accident Detection using Deep Learning
<b>A CCTV Camera Accident Detector</b>


<h4>Table of Contents</h4>

[Summary](#Summary)<br /> 
[Data and Processing](#Data)<br />
[The Algorithm](#Model)<br />
[References](#References)<br />

<a name="Summary"/>
<h3>Summary</h3>

The hierarchical recurrent neural network algorithm  model has been deployed to detect accidents in never-before-seen videos.

<br /><br />

<a name="Data"/>
<h3>Data and Processing</h3>

<b>The Data</b>:

We used the CADP dataset for videos containing accidents and the DETRAC dataset which was originally for object detection of vehicles, as our videos not containing accidents. To expand our dataset we also download youtube videos that contain accident. Over 380 videos were collected from the above mentioned sources.


<b>Training Dataset</b>:

For the final dataset, We had 188 videos with car, bus, bike etc accidents recorded in the CCTV camera at the corners of the street. We took the same number of negative cases(without accident) to maintain balanced classes.


<b>Processing</b>:

Each video is broken up into its individual frames to be analyzed separately. Each of these images is a two-dimensional array of pixels where each pixel has information about the red, green, and blue (RGB) color levels. To reduce the dimensionality at the individual image level, we convert the 3-D RGB color arrays to grayscale. Additionally, to make the computations more tractable on a CPU, we resize each image to (144, 256) - in effect reducing the size of each image to a 2-D array of 144x256.

<a name="Model"/>
<h3>The Algorithm:</h3>

A hierarchical recurrent neural network algorithm is used to tackle the complex problem of classifying video footage.

<br />
<b>The Algorithm</b>:

Each video is a set of individual images that are time-dependent sequences. The algorithm - a hierarchical recurrent neural network - is able to treat each video as a time-dependent sequence, but still allow each video to be an independent data point.

The algorithm uses two layers of long short-term memory neural networks. The first neural network (NN) is a recurrent network that analyzes the time-dependent sequence of the images within each video. The second takes the encoding of the first NN and builds a second NN that reflects which videos contain accidents and which do not. The resulting model enables a prediction of whether new dashcam footage has an accident.

Through this method, the HRNN incorporates a time-dependent aspect of the frames within each video to predict how likely it is a new video contains a car accident.

<a name="References"/>
<h3>References</h3>

<ul>
<li> <a href="https://github.com/fchollet/keras/blob/master/examples/mnist_hierarchical_rnn.py">HRNN for the MNIST Dataset for Handwritten Number Recognition</a>
<li> <a href="https://arxiv.org/abs/1506.01057">A Hierarchical Neural Autoencoder for Paragraphs and Documents</a>
<li> <a href="https://ankitshah009.github.io/accident_forecasting_traffic_camera">CADP Dataset</a>
</ul>

<br /><br />

#### Scripts for our minor project [Access to the image dataset](https://docs.google.com/forms/d/e/1FAIpQLSfuMMGafmiZ35alIgYkZeyGkR6gHhBURjxJPSe6aB6CWjN1EA/viewform) is made available under the Open Data Commons Attribution License: https://opendatacommons.org/licenses/by/1.0/.
