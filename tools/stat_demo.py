"""
Tool to visualize various configurations of a β-distribution.
"""
from scipy.stats import beta
import scipy.special
import numpy as np
import matplotlib.pyplot as plt

# Utility to plots, the β-distribution, the β CDF, and β inverse CDF.
def main(args):
    fig, ax = plt.subplots(1, 3)

    a, b = int(args.alpha), int(args.beta)
    x = np.linspace(0, 1, 100)
    rv = beta(a, b)
    ax[0].plot(x, rv.pdf(x),
        'r-', lw=5, alpha=0.6, label='beta pdf')


    # Show that beta CDF and incomplete beta function are the same.
    ax[1].plot(x, rv.cdf(x), 'k:', lw=4, label='beta cdf')
    ax[1].plot(x, scipy.special.betainc(a, b, x), 'y--', lw=1, label='cdf as I_x')

    # Plot I_x vs I_{x}^{-1}
    ax[2].plot(x, scipy.special.betainc(a, b, x), 'g-', lw=2, label='cdf')
    ax[2].plot(x, scipy.special.betaincinv(a, b, x), 'b-', lw=2, label='cdf^-1')

    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Utility that plots the β distribution and its CDF.")
    parser.add_argument("--alpha", default=2, help="Alpha parameter of distribution.")
    parser.add_argument("--beta", default=5, help="Beta parameter of distribution.")
    args = parser.parse_args()
    main(args)
