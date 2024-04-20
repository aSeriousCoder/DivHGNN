import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_mat(data, title, dir):
    plt.matshow(data)
    plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig('{}/{}.png'.format(dir, title))
    plt.show()
    plt.close()


def plot_multi_hist(data, label, title, dir):
    # Draw Plot
    data = np.array(data).T
    plt.figure(figsize=(16, 10), dpi=80)
    plt.hist(data, bins=30, rwidth=0.8, label=label, density=True)
    # Decoration
    plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig('{}/{}.png'.format(dir, title))
    plt.show()
    plt.close()


def plot_multi_density_curve(data, label, title, cumulative, dir):
    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    for i in range(len(data)):
        sns.kdeplot(data[i], shade=not cumulative, label=label[i], alpha=.7, cumulative=cumulative)
    # Decoration
    if cumulative:
        title = title + '(cumulative)'
    plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig('{}/{}.png'.format(dir, title))
    plt.show()
    plt.close()


def plot_density_curve(data, title, dir):
    # Draw Plot
    plt.figure(figsize=(16, 10), dpi=80)
    sns.kdeplot(data, shade=True, color="blue", alpha=.7)
    # Decoration
    plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig('{}/{}.png'.format(dir, title))
    plt.show()
    plt.close()


def plot_points_2d(data, title, dir):
    print('>>> PLT with {} points'.format(data.shape[1]))
    x = data[0]
    y = data[1]
    plt.scatter(x, y, alpha=0.2)
    plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig('{}/{}.png'.format(dir, title))
    plt.show()
    plt.close()



