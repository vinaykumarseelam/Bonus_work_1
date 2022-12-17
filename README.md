Option:
TensorRT inference, providing using REST APIs, and TensorflowLite inference
the steps used to setup the code were:
Executed the command as instructed in the repository after creating the virtual environment.
Installed the necessary packages because some weren't installed during setup.
To store the models produced, an output folder had to be made.
Python code was modified to allow for the training of a new model on MNIST data.
utilized the model saved to convert to lite mode, made the inference, and then compared the model's accuracy and predictions. the code was executed to generate the model. Here are some specifics on the modifications and outcomes.
Trained a cnn model and saved so that it can be used for api serving and then uploaded that model to the drive and used colab to execute the code present in apiserving.py file and then used the end point http://localhost:8501/v1/models/saved_model:predict to predict the image and th model redicted the image class exactly. More details in below sections.

# Changes made at CNNsimplemodels.py, myTFInference.py, exportTFlite.py
## 1.RT Inference
Trained with Fashion MNIST model.
To train the model using the MNIST data, a new model with a different set of parameters was created.
The class name and other parameters for the MNIST Model were changed as appropriate, and ImageOps was used to transform the RGB to grey scale to create the image array shape (28,28,1). The resulting inference model is tested using a sneaker image, and the prediction performed well with greater accuracy.
The output model has been saved in the output folder and will subsequently be utilized for lite model conversion and prediction use.
![alt](https://github.com/vinaykumarseelam/Bonus_work_1/blob/main/SS_1.png)
![alt](https://github.com/vinaykumarseelam/Bonus_work_1/blob/main/SS_2.png)

## 2.TF Lite 
 For embedded and mobile devices where the model needs to be smaller and more precise, lightweight models are used.
 This time, the converted model forecasted the sneakers as sandals, which is similar to the actual prediction. Export TF lite would take the model stored from the previous step and convert it to a lite model, which is then utilized to make inferences. s.

![alt](https://github.com/vinaykumarseelam/Bonus_work_1/blob/main/SS_3.png)


## 3.Serving with REST APIs
 Serve a TensorFlow model with TensorFlow Serving.
### Steps followed:
 1.Created a new model parameter with the name create simplemodelTest2 in CNNSimpleModels.py after training the classification model with myTFDistributedTrainer.py.
 2.As a result, output/fashion/1 would contain an output folder. This model is combined with our API to generate predictions.
 3.The restfull API returns the JSON format and from there we need to extract the predictions, this repsonse is generated when we call http://localhost:8501/v1/models/saved_model:predict which will return the result in JSON format.
 4.Using this code, we extract the picture from class names[np.argmax(predictions[0])], np.argmax(predictions[0]), class names[test labels[0]], and test labels[0]. the forecasts array
 5.The essential code is located in the file apiserving.py, which has been run in Colab. The model, which is located in the outputs/fashion/1 folder, has been uploaded to my drive and put to use by running the code in Colab to carry out the serving process. 

# MultiModalClassifier
This project repository contains well-known models from Tensorflow and Pytorch for a multi-modal deep learning classifier. These fundamental models serve as a foundation upon which to develop and can serve as a springboard for any fresh concepts or uses. Please consult this repository for ML and DL fundamentals: https://github.com/lkk688/DeepDataMiningLearning.


# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/
