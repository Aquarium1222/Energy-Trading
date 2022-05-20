import matplotlib.pyplot as plt


def plot_loss(x, y, title, save=False):
    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(title)
    if save:
        plt.savefig(title + '.png')
    else:
        plt.show()
    plt.clf()
