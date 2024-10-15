import matplotlib.pyplot as plt

def plot_sample(X, y, y_pred, index):
    plt.imshow(X[index])
    plt.title(f"True: {y[index]} Predicted: {y_pred[index]}")
    plt.show()
