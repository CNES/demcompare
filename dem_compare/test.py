import numpy as np
import math
def roundUp(x, y):
    return int(math.ceil((x / float(y)))) * y

from dem_compare_lib.a3d_georaster import A3DGeoRaster
dh=A3DGeoRaster('test_baseline/final_dh.tif')
data = dh.r.reshape(dh.r.size)
mu = np.nanmean(data)
sigma = np.nanstd(data)
data = data[np.where((data>mu-3*sigma)*(data<mu+3*sigma))]

borne = np.max([abs(np.nanmin(data)), abs(np.nanmax(data))])
bin_step=1
bins = np.arange(-roundUp(borne, bin_step), roundUp(borne, bin_step)+bin_step, bin_step)

import matplotlib.pyplot as plt
n, bins, patches = plt.hist(data, bins, normed=True, color='b', histtype='step')
plt.show()

from scipy import stats
k2, p = stats.normaltest(data)
alpha = 0.05
print("p = {}".format(p))
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')


from scipy import exp
from scipy.optimize import curve_fit

def gaus(x, a, x_zero, sigma):
    return a * exp(-(x - x_zero) ** 2 / (2 * sigma ** 2))

popt, pcov = curve_fit(gaus, bins[0:bins.shape[0] - 1] + int(bin_step / 2), n,
                                           p0=[1, mu, sigma])
fitted_hist = gaus(np.arange(bins[0], bins[bins.shape[0] - 1], float(bin_step) / 10.), *popt)
k2, p = stats.normaltest(fitted_hist)
alpha = 0.05
print("p = {}".format(p))
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')



import numpy as np
import math
def roundUp(x, y):
    return int(math.ceil((x / float(y)))) * y

from dem_compare_lib.a3d_georaster import A3DGeoRaster
dh=A3DGeoRaster('test_baseline/final_dh.tif')

import matplotlib.pyplot as plt
n, bins, patches = plt.hist(dh.r, normed=True, histtype='step')
plt.show()

from scipy import stats
k2, p = stats.normaltest(dh.r)
alpha = 0.05
print("p = {}".format(p))
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')
