# MoodNet: Image Classification of Happy and Depressed People using CNN
# Project Overview
MoodNet is an image classification project that utilizes Convolutional Neural Networks (CNNs) to distinguish between images of happy and depressed individuals. The project aims to leverage deep learning techniques to automatically detect emotional states based on facial expressions captured in images. Key preprocessing techniques such as image grayscale conversion, normalization, and resizing are applied to enhance model performance.

# Dataset
The dataset consists of images of individuals categorized as happy and depressed.
Each image is labeled with its corresponding emotional state (happy or depressed).
# CNN Architecture
MoodNet utilizes a CNN architecture for image classification tasks.
The architecture comprises multiple convolutional layers followed by pooling layers for feature extraction.
Fully connected layers are employed for classification based on extracted features.
Activation functions such as ReLU are applied to introduce non-linearity in the network.
Dropout layers are incorporated to prevent overfitting.
## Preprocessing Techniques
Image Grayscale Conversion: Convert color images to grayscale to reduce complexity and computational cost.
Normalization: Normalize pixel values to a common scale to facilitate model convergence.
Image Resizing: Resize images to a standard size to ensure uniformity and optimize model performance.
## Model Training
The dataset is split into 80% training, 10% validation and 10% testing sets an for model training and evaluation.
MoodNet is trained using backpropagation and stochastic gradient descent optimization techniques.
Training parameters such as learning rate, batch size, and number of epochs are fine-tuned to optimize model performance.
Model performance metrics such as accuracy, precision, recall, and F1-score are monitored during training.
## Evaluation and Validation
The trained model is evaluated on a separate test set to assess its generalization performance.
Performance metrics are computed to measure the model's accuracy and effectiveness in classifying happy and depressed individuals.
Confusion matrices and ROC curves may be generated to visualize model performance and identify potential areas of improvement.
## Usage
To classify and mood of a person in image using  MoodNet model.
## Future Enhancements
Explore transfer learning techniques to leverage pre-trained CNN models for improved performance.
Experiment with data augmentation strategies to increase dataset diversity and enhance model robustness.
Investigate interpretability techniques to gain insights into model predictions and feature importance.
## Contributors
Suhaib Mukhtar
## Contact
suhaibmukhtar2@gmail.com
https://www.linkedin.com/in/suhaib-mukhtar-63777b217
## Provide contact information for support or inquiries.
