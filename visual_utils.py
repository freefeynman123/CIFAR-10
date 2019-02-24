import matplotlib.pyplot as plt
import seaborn as sns
import random
from typing import Union
import numpy as np


def plot_images(images: Union[list, np.ndarray], labels: Union[list, np.ndarray], label_names: str,
                images_to_show: int = 10):
    """
    Function to plot images from dataset with labels assigned
    """
    num_of_classes = len(set(labels))
    fig, ax = plt.subplots(nrows=num_of_classes, ncols=images_to_show)
    axes = ax.flatten()
    for num_class in range(num_of_classes):
        examples_from_class = images[labels == num_class]
        examples_to_show = random.sample(list(examples_from_class), images_to_show)
        for index, label_name in enumerate(label_names):
            axes[num_class * num_of_classes + index].imshow(examples_to_show[index])
            axes[num_class * num_of_classes + index].set_axis_off()
        axes[num_class * num_of_classes].text(-10, 20, label_names[num_class], ha='right')


def plot_pca_results(reduced_data, hue, label_names, title):
    """
    Function to plot results from PCA dimensionality reduction
    """
    g = sns.scatterplot(x=reduced_data.T[0], y=reduced_data.T[1], hue=hue, legend="full",
                        palette=sns.color_palette("Set1", n_colors=10))
    legend_handles = g.get_legend_handles_labels()[0]
    g.legend(legend_handles, label_names)
    g.set_title(title)


def plot_umap_results(umap_features: Union[list, np.ndarray], neighbors: str, hue, label_names: str):
    """
    Function to plot results from UMAP dimensionality reduction.
    It accounts for different number of neighbors
    """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17, 10))
    axes = ax.flatten()
    for index, (reduced_data, neighbor) in enumerate(zip(umap_features, neighbors)):
        if index == 0:
            legend = 'full'
        else:
            legend = False
        g = sns.scatterplot(x=reduced_data.T[0], y=reduced_data.T[1], hue=hue, legend=legend,
                            palette=sns.color_palette("Set1", n_colors=10), ax=axes[index])
        legend_handles = g.get_legend_handles_labels()[0]
        g.legend(legend_handles, label_names)
        g.set_title("Umap for {i} neighbors".format(i=neighbor))
