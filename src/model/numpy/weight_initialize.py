import numpy as np
import matplotlib.pyplot as plt


D = np.random.randn(1000, 5000)
hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh']*len(hidden_layer_sizes)


act = {'relu': lambda x: np.maximum(0, x), 'tanh': lambda x: np.tan(x)}
Hs = {}
for i in range(len(hidden_layer_sizes)):
    X = D if i == 0 else Hs[i-1]
    fan_in = X.shape[1]
    fan_out = hidden_layer_sizes[i]
    W = np.random.randn(fan_in, fan_out) * 0.01

    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    Hs[i] = H # cache result on this layer


# looka at distritutions at each layer


print('input layer had mean %f and std %f' % (np.mean(D), np.std(D)))
layer_means = [np.mean(H) for i, H in Hs.items()]
layer_stds = [np.std(H) for i, H in Hs.items()]
for i, H in Hs.items():
    print("hidden layer %d had mean %f and std %f" % (i+1, layer_means[i], layer_stds[i]))


# plot the means and standard deviations
plt.figure()
plt.subplot(121)
plt.plot(Hs.keys(), layer_means, 'ob-')
plt.title('layer mean')
plt.subplot(122)
plt.plot(Hs.keys(), layer_stds, 'or-')
plt.title('layer std')

# plot the raw distributions
plt.figure()
for i, H in Hs.items():
    plt.subplot(1, len(Hs), i+1)
    plt.hist(H.ravel(), 30, range(-1, 1))


plt.draw()