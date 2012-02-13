#!/usr/bin/env python


import sys

import matplotlib
from matplotlib import cm, pyplot
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy
import scipy
import scipy.ndimage


def main():
    reverse = False
    method = 'scatter'
    method = 'fft'
    method = 'density'

    # Load Ubisoft logo.
    print 'Loading Ubisoft logo'
    ubi_logo = pyplot.imread('ubi_logo.png')
    # Convert to grayscale by averaging RGB components.
    ubi_logo_gray = ubi_logo[:, :, 0:3].mean(axis=2)
    if reverse:
        ubi_logo_gray = 1 - ubi_logo_gray
    # Fill background.
    mask = ubi_logo[:, :, 3] <= 0
    ubi_logo_gray[mask] = 1
    # Display Ubisoft logo.
    #pyplot.gray()
    #pyplot.imshow(ubi_logo_gray)
    #pyplot.show()

    if method in ('scatter', 'density'):

        # Assign a sampling weight to each pixel.
        print 'Setting sample weights'
        pixels = []
        for i in xrange(ubi_logo_gray.shape[0]):
            for j in xrange(ubi_logo_gray.shape[1]):
                pixels.append([i, j, 1 - ubi_logo_gray[i, j]])
        pixels = numpy.array(pixels)
        # Ensure weights sum to 1.
        pixels[:, 2] /= pixels[:, 2].sum()

        # Sample pixels based on their weight.
        print 'Sampling pixels'
        n_samples = 10000
        seed = 1827
        rng = numpy.random.RandomState(seed)
        samples = rng.multinomial(n_samples, pixels[:, 2])

        print 'Generating sampled image'
        sampled_logo = ubi_logo_gray.copy()
        sampled_pixels = []
        idx = 0
        for i in xrange(sampled_logo.shape[0]):
            for j in xrange(sampled_logo.shape[1]):
                if samples[idx] > 0:
                    sampled_logo[i, j] = 0
                    sampled_pixels.append([i, j])
                else:
                    sampled_logo[i, j] = 1
                idx += 1
        sampled_pixels = numpy.array(sampled_pixels, dtype=float)

    fig = pyplot.figure(figsize=(6, 6))

    if method == 'density':
        # Compute density estimate.
        print 'Computing density estimates'
        scale_factor = 0.3
        density = numpy.zeros((sampled_logo.shape[0] * scale_factor,
                               sampled_logo.shape[1] * scale_factor))
        point = numpy.zeros(2)
        sigma = 8
        count = 0
        total = density.size
        for i in xrange(density.shape[0]):
            for j in xrange(density.shape[1]):
                point[0] = i / scale_factor
                point[1] = j / scale_factor
                # Squared pairwise difference.
                diff = sampled_pixels - point
                diff *= diff
                # Sum and scale by sigma.
                dist = diff.sum(axis=1) / sigma**2
                # Take exponential.
                rbf = numpy.exp(-dist)
                density[i, j] = rbf.sum()
                count += 1
                sys.stdout.write('\r%2d%%' % int(count / float(total) * 100 + 0.5))
                sys.stdout.flush()

        print
        # Normalize.
        print 'Normalizing'
        density /= density.max()
        density = density

        # Display density
        print 'Plotting density'
        #pyplot.gray()
        #pyplot.imshow(density)
        #pyplot.show()

        ax = fig.gca(projection='3d')
        ax.view_init(elev=50, azim=0)

        # Undocumented feature to hide axes.
        ax._axis3don = False

        add_border = 0.8
        X = numpy.arange(0, density.shape[0] * (1 + add_border))
        Y = numpy.arange(0, density.shape[1] * (1 + add_border))
        X, Y = numpy.meshgrid(X, Y)
        Z = density.T
        Z[Z < 0.02] = 0
        #Z = 1 - Z
        Z *= 0.3
        new_Z = numpy.zeros(X.shape)
        start_0 = density.shape[0] * add_border * 0.5
        start_1 = density.shape[1] * add_border * 0.5
        new_Z[start_0:start_0 + Z.shape[0], start_1:start_1 + Z.shape[1]] = Z
        Z = new_Z
        cmap = cm.Reds
        cmap = cm.gist_heat
        cmap = cm.RdBu
        cmap = cm.jet
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cmap,
                shade=True,
                linewidth=0.2, antialiased=True)
        #ax.set_xlim(0, 62)
        #ax.set_ylim(0, 62)
        ax.set_zlim(-0.01, 1.01)

        #ax.zaxis.set_major_locator(LinearLocator(10))
        #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #fig.colorbar(surf, shrink=0.5, aspect=5)

    if method == 'scatter':
        pyplot.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1])

    if method == 'fft':
        fft = numpy.fft.rfft2(ubi_logo_gray)
        if False:
            fft = numpy.abs(fft)
            fft = numpy.log(fft)
            n = min(fft.shape)
            fft = fft[0:n, 0:n]
            fft = numpy.concatenate((numpy.fliplr(fft), fft), axis=1)
            fft = numpy.concatenate((numpy.flipud(fft), fft), axis=0)
        fft = scipy.ndimage.fourier_gaussian(fft, sigma=1, n=155)
        fft = numpy.abs(fft)
        fft = numpy.log(fft)
        pyplot.imshow(fft)

    #pyplot.show()

    pyplot.savefig('logo.png', dpi=300)

    return 0


if __name__ == '__main__':
    sys.exit(main())
