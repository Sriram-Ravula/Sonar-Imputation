from scipy.special import gamma, kn
import numpy as np


#Get a k-distribution sample at x with parameters b, v
def __get_k(x, b, v):
    one = 2 * b / gamma(v)
    two = (b * x / 2) ** v
    three = kn(v - 1, b * x)

    return one * two * three

#generate a k-distribution with n samples and parameters b, v
def gen_data(n, b, v):
    pdf = np.zeros(n)

    for i in range(n):
        j = i + int(n / 5)

        pdf[i] = __get_k(j / 100, b, v)

    return pdf

#Scale a signal to be in range [-1, 1]
def scale(x):
    mins = np.min(x)
    maxes = np.max(x)

    return (x - mins) / (maxes - mins)

#perform a single pass of a split-window filter on signal x centered at sample m, with window parameters Ws, Wg
def __single_pass(x, m, Ws, Wg):
    n = len(list(x.squeeze()))

    upper = int(m + (Ws + (Wg - 1) / 2))
    lower = int(m - (Ws + (Wg - 1) / 2))

    inner_range = upper - lower + 1

    z = 0

    for j in range(inner_range):
        i = j + lower

        if (abs(i - m) > (Wg - 1) / 2):

            if (i >= n):
                z += x[n - 1]
            elif (i < 0):
                z += x[0]
            else:
                z += x[i]

    z = z / (2 * Ws)

    return z

#Perform clipping for split-window filtering on original signal x, compared to reference z
#centered at sample m, with clipping parameter r
def __clip(x, z, m, r):
    if (x[m] < r * z[m]):
        return x[m]
    else:
        return z[m]

#perform 2-pass split-window filtering on x with clipping parameter r and window parameters Ws, Wg
def two_pass_filtering(x, Ws, Wg, r):
    n = len(list(x.squeeze()))

    z = np.zeros(n)

    for i in range(n):
        z[i] = __single_pass(x, i, Ws, Wg)

    y = np.zeros(n)

    for i in range(n):
        y[i] = __clip(x, z, i, r)

    y_hat = np.zeros(n)

    for i in range(n):
        y_hat[i] = __single_pass(y, i, Ws, Wg)

    x = np.abs(x)
    y_hat = np.abs(y_hat)

    return np.divide(x.squeeze(), y_hat.squeeze())