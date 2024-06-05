# Handwritten Digits Classification: A Comparative Study of CNN and Traditional ML Methods

## Authors
- Yue Zhang (zhang.yue23@northeastern.edu)
- Anita George (george.ani@northeastern.edu)

## Introduction
This project focuses on the classification of handwritten digits using the MNIST dataset. We compare the performance of Convolutional Neural Networks (CNNs) with traditional machine learning techniques such as K-Nearest Neighbors (KNN) and Support Vector Machines (SVM). The study aims to evaluate the effectiveness of these different methods in the context of digit recognition.

Our primary CNN benchmark model is based on the LeNet-5 architecture, inspired by the work in "Pruning Distorted Images in MNIST Handwritten Digits"[1]. We also implemented models from TensorFlow tutorials and conducted experiments with traditional ML approaches and hybrid methods combining CNNs with KNN and SVM.

## Dataset
### MNIST
The MNIST dataset is a standard benchmark in image classification, consisting of 70,000 images (60,000 for training and 10,000 for testing). Each image is 28x28 pixels, representing digits from 0 to 9.

### Handwritten Dataset
To further evaluate our models, we created a dataset of 200 handwritten digit images (20 per digit). This dataset helps assess model adaptability beyond the MNIST dataset.

## Models Trained
### CNN Models
1. **CNN 1**: Based on the architecture from "Pruning Distorted Images in MNIST Handwritten Digits"[1].
   - High accuracy through convolutional layers with varying filters and kernel sizes.
   - Dropout regularization, ReLU activation, softmax classification.
   - Trained for 90 epochs with an initial learning rate of 0.0001.
   - 480,554 trainable parameters.
2. **CNN 2**: Adapted from a TensorFlow tutorial, with modifications to the activation functions.
   - Convolutional layers, max-pooling, dropout regularization.
   - ReLU activation, softmax classification, Adam optimizer.
   - Trained for 10 epochs with an initial learning rate of 0.001.
   - 37,610 trainable parameters.
3. **CNN 3**: A classic LeNet-5 architecture.
   - Convolutional layers with average-pooling, Tanh and Sigmoid activation.
   - Sigmoid classification.
   - Trained for 10 epochs with an initial learning rate of 0.01.
   - 61,706 trainable parameters.

### Traditional ML Models
1. **KNN**: Hyperparameter tuning performed with k-values and weight options.
   - Best parameters: K=3, weight='distance'.
2. **SVM**: Implemented with linear and polynomial kernels.
   - Polynomial kernel of degree 4 showed the best performance.
   - MinMaxScaler used for data normalization.

### Hybrid Models
1. **CNN + KNN**: Features extracted from the CNN model used as input for a KNN classifier.
   - Best parameters: K=5, weight='distance'.
2. **CNN + SVM**: Features extracted from the CNN model used as input for an SVM classifier with a polynomial kernel.
   - Polynomial kernel of degree 4.

## Results
### MNIST (In-Distribution) Data
- **CNN Models**:
  - CNN 1: Highest accuracy among all models.
  - CNN 2: Slightly lower accuracy than CNN 1 but still high.
  - CNN 3: Moderate accuracy, lower than CNN 1 and CNN 2.
- **Traditional ML Models**:
  - KNN: Good accuracy but with some random misclassifications.
  - SVM: Notable misclassifications, especially misclassifying many digits as "8".
- **Hybrid Models**:
  - CNN + KNN: Reduced misclassifications compared to standalone KNN.
  - CNN + SVM: Better performance than standalone SVM, but still significant misclassifications.

### Handwritten (Out-of-Distribution) Data
- **Performance Decline**: All models showed a significant performance decline on the handwritten dataset.
- **Misclassification Trends**:
  - CNN 3 and standalone KNN models predominantly misclassified digits as "8" or "0".
  - Hybrid models (CNN + KNN and CNN + SVM) outperformed standalone traditional models but still struggled, especially with digit "8".
- **Best Performance**:
  - CNN 2 achieved the highest accuracy on handwritten data.

## Discussion
- **CNN Superiority**: CNN models, particularly deeper architectures, showed superior performance on MNIST.
- **Traditional ML Limitations**: KNN and SVM, while effective, had limitations in handling the complexity of handwritten digit classification.
- **Hybrid Model Benefits**: Combining CNNs with KNN or SVM improved performance but did not completely resolve misclassification issues.
- **Performance Fluctuations**: Model performance varied significantly between MNIST and handwritten datasets, highlighting challenges in generalizing across different data distributions.

### Key Findings
- **High Accuracy**: CNN 1 achieved the highest training and testing accuracy on MNIST.
- **Performance Drop on OOD Data**: All models experienced a performance drop on the handwritten dataset, with the highest accuracy being 0.704 by CNN 2.
- **Model Biases**: Models showed biases in misclassifying certain digits, particularly "8" and "0".


## References
1. Amarnath R & Vinay Kumar V (2023). Pruning Distorted Images in MNIST Handwritten Digits. [arXiv link](https://arxiv.org/pdf/2307.14343.pdf)
2. LeCun, Y., Cortes, C., Burges, C. (1998). The MNIST database of handwritten digits. [MNIST](http://yann.lecun.com/exdb/mnist/)
3. TensorFlow. Handwritten digit recognition with CNNs. [TensorFlow Tutorial](https://www.tensorflow.org/js/tutorials/training/handwritten_digit_cnn)
4. Tuning the hyper-parameters of an estimator. [Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/grid_search.html)
---

This project was an exciting exploration into the capabilities of various machine learning techniques in the realm of digit recognition. We hope our findings contribute to the broader understanding and development of effective image classification models.
