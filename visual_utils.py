import matplotlib.pyplot as plt
import seaborn as sns
import random

def plot_images(images, labels, label_names, images_to_show = 10):
    num_of_classes = len(set(labels))
    fig, ax = plt.subplots(nrows=num_of_classes, ncols=images_to_show)
    axes = ax.flatten()
    for num_class in range(num_of_classes):
        examples_from_class = images[labels == num_class]
        examples_to_show = random.sample(list(examples_from_class), images_to_show)
        for index, label_name in enumerate(label_names):
            axes[num_class*num_of_classes + index].imshow(examples_to_show[index])
            axes[num_class*num_of_classes + index].set_axis_off()
        axes[num_class*num_of_classes].text(-10, 20, label_names[num_class], ha='right')

def plot_pca_results(reduced_data, hue, label_names, title):
	g = sns.scatterplot(x=reduced_data.T[0], y=reduced_data.T[1], hue=hue, legend = "full",
	                   palette = sns.color_palette("Set1", n_colors=10))
	legend_handles = g.get_legend_handles_labels()[0]
	g.legend(legend_handles, label_names)
	g.set_title(title)

def plot_umap_results(umap_features, neighbors, hue, label_names):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,10))
    axes=ax.flatten()
    for index, (reduced_data, neighbor) in enumerate(zip(umap_features, neighbors)):
        if index==0:
            legend='full'
        else:
            legend=False
        g = sns.scatterplot(x=reduced_data.T[0], y=reduced_data.T[1], hue=hue, legend = legend,
                           palette = sns.color_palette("Set1", n_colors=10), ax=axes[index])
        legend_handles = g.get_legend_handles_labels()[0]
        g.legend(legend_handles, label_names)
        g.set_title("Umap for {i} neighbors".format(i=neighbor))

