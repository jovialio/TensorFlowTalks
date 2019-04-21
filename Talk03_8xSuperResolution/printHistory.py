import pickle
import matplotlib.pyplot as plt
import numpy as np

def get_running_mean(x, N):
    kernel = np.ones(N) / N
    running_mean = np.convolve(x, kernel, mode = 'same')
    return running_mean

def plot(iterations, avg_losses, other_params, savepath):
    x_limits, y_limits, num_mean = other_params
    plt.plot(iterations, avg_losses, 'g', alpha = 0.6)
    print(len(avg_losses))
    if len(avg_losses) > num_mean:
        plt.plot(iterations, get_running_mean(avg_losses, num_mean), 'b')
    plt.xlabel("Iterations")
    plt.ylabel("Average loss")
    if x_limits:
        plt.xlim(x_limits)
    if y_limits:
        plt.ylim(y_limits)
    plt.grid()
    plt.savefig(savepath+'plot.png')
    plt.show()

objects = []
with (open("trainHistoryDict", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
            
            
loss = objects[0]['loss']

other_params = None, None, 3 #x_limits, y_limits, num_mean
savepath = './'

plot(range(len(loss)), loss, other_params, savepath)

