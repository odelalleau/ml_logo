#!/usr/bin/env python


import sys

import matplotlib
from matplotlib import pyplot
import numpy


def main():
    # Load Ubisoft logo.
    print 'Loading Ubisoft logo'
    ubi_logo = pyplot.imread('/Users/delallea/Downloads/ubi.png')
    # Convert to grayscale by averaging RGB components.
    ubi_logo_gray = ubi_logo[:, :, 0:3].mean(axis=2)
    # Background should be white.
    mask = ubi_logo[:, :, 3] <= 0
    ubi_logo_gray[mask] = 1
    # Display Ubisoft logo.
    #pyplot.gray()
    #pyplot.imshow(ubi_logo_gray)
    #pyplot.show()

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
    n_samples = 3000
    seed = 1827
    rng = numpy.random.RandomState(seed)
    samples = rng.multinomial(n_samples, pixels[:, 2])

    # Plot sampled points.
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

    # Compute density estimate.
    print 'Computing density estimates'
    scale_factor = 0.1
    density = numpy.zeros((sampled_logo.shape[0] * scale_factor,
                           sampled_logo.shape[1] * scale_factor))
    point = numpy.zeros(2)
    sigma = 5
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

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
    X = numpy.arange(0, density.shape[0], 1)
    Y = numpy.arange(0, density.shape[1], 1)
    X, Y = numpy.meshgrid(X, Y)
    Z = density.T
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)
    ax.set_zlim(-0.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    pyplot.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
