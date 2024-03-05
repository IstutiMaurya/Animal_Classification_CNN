# Animal_Classification_CNN
Welcome to the Animal Classification Project repository! This project aims to classify animal images into different categories using deep learning techniques. The dataset comprises 90 different animal images, and we'll explore various classification scenarios, including one-vs-rest, binary, and multi-class classification.

## Dataset Preparation
dataset link: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download

I've organize the dataset to facilitate one-vs-rest, binary, and multi-class classification tasks. The data preprocessing steps are detailed in the notebooks and scripts within the repository.

## Model Development
I've develop a custom convolutional neural network (CNN) model specifically tailored for animal classification. The model architecture and training procedures are:

* Model Initialization: A Sequential model (model1) is initialized, representing a linear stack of layers.
* Convolutional Layers:Four convolutional layers are added to the model using the Conv2D layer class. Each convolutional layer applies a specified number of 3x3 filters (32, 64, 64, and 128 respectively), followed by the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
* MaxPooling2D layers are interspersed between convolutional layers to reduce the spatial dimensions of the feature maps and capture the most important features.
* Dropout Layers: Dropout layers are added after the second and third convolutional layers to mitigate overfitting. A dropout rate of 25% is specified, which randomly sets a fraction of input units to zero during training to prevent co-adaptation of feature detectors.
* Flattening: After the final convolutional layer, a Flatten layer is added to convert the 2D feature maps into a 1D feature vector, which can be fed into the densely connected layers.
* Densely Connected Layers: Two Dense layers are added as the fully connected layers of the model. The first Dense layer consists of 64 units with the ReLU activation function, serving as the hidden layer of the artificial neural network (ANN).
* Another Dropout layer is added after the hidden layer to further prevent overfitting.
* The final Dense layer contains a single unit with the sigmoid activation function, which outputs a probability indicating the likelihood of the input image belonging to the positive class (e.g., in binary classification, it could represent a cat if the task is to classify cats vs. non-cats).
*Model Compilation: The model is compiled using binary cross-entropy as the loss function, Adam optimizer, and accuracy as the evaluation metric.

## Training and Evaluation
For training, I have selected random pictures from the drive and predicted the output based on that. The evaluation results and visualizations are presented in the notebooks and scripts.

## Instructions for Use
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/IstutiMaurya/Animal_Classification_CNN.git

Install the required dependencies:

## Copy code
pip install -r requirements.txt
Explore the notebooks in the notebooks/ directory to understand the data preprocessing, model development, and evaluation steps.

Execute the scripts in the models/ directory to train and evaluate the model for various classification tasks.

## Contribution Guidelines
Contributions to this project are welcome! If you have suggestions for improvement, found a bug, or want to add new features, please submit a pull request. Ensure your code adheres to the project's coding standards and documentation guidelines.

## Acknowledgment
I would like to thank SRIP for encouraging me to do this task for their evaluation round for the summer internship.

Thank you for your interest in the Animal Classification Project! If you have any questions or feedback, please feel free to contact us.
