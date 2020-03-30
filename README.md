# Malaria Detection
Neural Network that detects whether a cell is infected or not by Malaria

## :boom: Malaria-infected cell detection. Convolutional Neural Network

This project's idea came from the desire to apply my deep learning knowledge into a real-life problem. This Neural Network goal is to determine whether a cell has been infected by Malaria. The cell's diagnosis is delivered in a pdf file next to the original cell image.

## :computer: Technology stack
Wrote in python 3. Main modules:

**OpenCV** -> Image reading and handling

**TensorFlow** -> Model training and prediction

## :microscope: Model Overview and Flowchart

Our task is accomplished using a 3 layers Convolutional Neural Network (CNN)

Each of the three layers has 64 units and a kernel of size 3x3. The activation process is done using a Rectified Linear Unit (RELU). After each convolutional layer comes a pooling layer of size 2x2. Optimizer; "adam". Loss; "Binary Crossentropy".
The fitting process iterate through 10 epochs, takes a batch size of 32 and has a validation split of 0.2.

Tensorboard was used to compare different models taking the one that best fit. All training run logs will be saved to the following path data/logs.

The training was done using 6.000 colored images of the same dimension 130x130. It was evenly split into 3.000 Parasitized images and 3.000 Uninfected images. After training, model loss and accuracy were 0.14 and 0.95 respectively.

## :chart_with_upwards_trend: Model Stats

### Fitting

Model's name: Malaria-CNN-1585423128
Model: "sequential"
_________________________________________________________________
_Layer (type)                 Output Shape              Param #   
=================================================================_

conv2d (Conv2D)              (None, 128, 128, 64)      1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 62, 62, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 31, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 29, 29, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 12544)             0         
_________________________________________________________________
_dense (Dense)                (None, 1)                 12545     
=================================================================_

Total params: 88,193
Trainable params: 88,193
Non-trainable params: 0

### Training

Train on 4800 samples, validate on 1200 samples
Epoch 1/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.6873 - accuracy: 0.5437 - val_loss: 0.6781 - val_accuracy: 0.5308
Epoch 2/10
4800/4800 [==============================] - 67s 14ms/sample - loss: 0.6752 - accuracy: 0.5794 - val_loss: 0.6690 - val_accuracy: 0.6000
Epoch 3/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.5142 - accuracy: 0.7406 - val_loss: 0.2955 - val_accuracy: 0.8967
Epoch 4/10
4800/4800 [==============================] - 65s 14ms/sample - loss: 0.2591 - accuracy: 0.9127 - val_loss: 0.2958 - val_accuracy: 0.9367
Epoch 5/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.2284 - accuracy: 0.9240 - val_loss: 0.2841 - val_accuracy: 0.9367
Epoch 6/10
4800/4800 [==============================] - 65s 14ms/sample - loss: 0.2408 - accuracy: 0.9146 - val_loss: 0.2588 - val_accuracy: 0.9242
Epoch 7/10
4800/4800 [==============================] - 65s 14ms/sample - loss: 0.2040 - accuracy: 0.9319 - val_loss: 0.2698 - val_accuracy: 0.9200
Epoch 8/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.1943 - accuracy: 0.9323 - val_loss: 0.2508 - val_accuracy: 0.9392
Epoch 9/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.1585 - accuracy: 0.9463 - val_loss: 0.2200 - val_accuracy: 0.9283
Epoch 10/10
4800/4800 [==============================] - 66s 14ms/sample - loss: 0.1415 - accuracy: 0.9471 - val_loss: 0.2224 - val_accuracy: 0.9383

Model performance
6000/6000 [==============================] - 27s 5ms/sample - loss: 0.1421 - accuracy: 0.9588
Loss: 0.14, Accuracy: 0.95

## :wrench: Configuration
Install python and mandatory modules

Use the following commands if you are using the anaconda distribution.

```
conda create -n new_env_name_here
conda activate new_env_name_here
conda install python=3.7
pip install -r requirements.txt
```

**Note:** Environment managers differ from one another. It's strongly recommended to check its documentation.

## :snake: Running the main.py script
Before running the GUI for the first time, you must train the model and save it as a binary file. This training is a one time task, once trained the model is loaded from the binary file. Train your model by navigating to where the rep was downloaded and type `python packages/Model/model.py` in your terminal window. Typically, the training process takes about 3 minutes but it is subjected to your system performance (RAM, cores, etc)

This GUI can run any other model that was trained using the diamonds data set found in the data folder.

### Running GUI using pre-installed model 
Navigate to where the rep was downloaded and type `python main.py` in your terminal. This will run the main.py script which automatically opens the GUI and loads the pre-installed Random Forest model trained in the previous step.

### Running GUI using another model
Any model trained using the diamonds data set can be used to run the GUI. Custom models should be placed in the following path Diamonds_Appraisal/data/model_binary/my_custom_model.pkl

Once there, go to your terminal and run the following line of code: `python main.py my_custom_model.pkl`

![](images/gui1.png) ![](images/gui2.png)

### Predicting diamond's price 
Prices are calculated using all 9 entries at the left part of the window. All entries should be filled. First 6 entries can only take float or integer values, while the last 3 are drop-down lists.

Once all entries are filled, click on the "Calculate price" button to update the "Predicted Price" label.

## :information_source: Data set info

![](images/diamond.jpg)

Comprised by almost 54.000 registries. Data set features are the following:

**price** price in US dollars (min: $326 - max: $18,823)

**carat** weight of the diamond (min: 0.2 - max: 5.01)

**cut** quality of the cut (Fair (lowest), Good, Very Good, Premium, Ideal (highest))

**color** diamond colour, from J (worst) to D (best)

**clarity** a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

**x** length in mm (min: 0 - max: 10.74)

**y** width in mm (min: 0 - max: 58.9)

**z** depth in mm (min: 0 - max: 31.8)

**depth** total depth percentage = z / mean(x, y) = 2 * z / (x + y) (min: 43 - max: 79)

**table** width of top of diamond relative to widest point (min: 43 - max: 95)

## :file_folder: **Folder structure**
```
└── Diamonds_Appraisal
    ├── .gitignore
    ├── requirements.txt
    ├── README.md
    ├── main.py
    ├── notebooks
    │   └── Pipeline.ipynb
    ├── packages
    │   ├── GUI
    │   │   └── GUI.py
    │   └── Model
    │   │   └── model.py
    └── data
        ├── raw
        │   ├── diamonds.csv
        │   ├── diamonds_test.csv
        │   └── diamonds_train.csv
        └── model_binary
            └── RandomForest.pkl
```

## :interrobang: **Custom models**
Check the Pipeline.ipynb notebook or the model.py script to get a broad idea. These two files have all necessary steps to create, test, enhance and save your modules.

## :love_letter: **Contact info**
Any doubt? Advice?  Drop me a line! :smirk: