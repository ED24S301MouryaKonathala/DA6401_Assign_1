import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

def load_data(dataset='fashion_mnist'):
    if dataset == 'fashion_mnist':
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
        x_train_full = x_train_full.reshape(-1,28*28)/255.0
        x_test  = x_test.reshape(-1,28*28)/255.0

        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size = 0.1, random_state = 42)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def plot_samples_per_class(x_train, y_train):
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    plt.figure(figsize=(10,5))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_train[np.random.choice(np.where(y_train == i)[0])].reshape(28, 28), cmap='gray')
        plt.title(class_names[i])
        plt.axis('off')
    plt.show()