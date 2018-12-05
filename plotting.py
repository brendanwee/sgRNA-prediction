import matplotlib.pyplot as plt


def plot_scatter(x,y, xlabel, ylabel, figname):
    plt.scatter(x, y, c="g", alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(figname)
    plt.savefig(figname + ".png", format="png")
    plt.clf()


def plot_lines( y, ylabel, figname, x1, xlabel1, x2, xlabel2):

    plt.plot(y, x1, label=xlabel1)
    plt.plot(y, x2, label=xlabel2)
    plt.xlabel(ylabel)
    plt.ylabel("MSE")
    plt.title(figname)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.savefig(figname + ".png", format="png")
    plt.clf()
