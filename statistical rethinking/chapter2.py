import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import uniform


def example2_3():
    """Estimate surface of earth which is water, by randomly drawing samples."""
    W = 6  # Water samples
    L = 3  # Land samples
    N = W + L
    n_samples = 500
    p = np.linspace(0, 1, n_samples)  # proportion which is water

    likelihood = binom.pmf(W, N, p)
    likelihood = likelihood / sum(likelihood)

    prior = np.array([1] * n_samples)
    # prior = binom.pmf(W-1, N-1, p)

    prior = prior / sum(prior)

    posterior = prior * likelihood
    posterior = posterior / sum(posterior)

    fig, ax = plt.subplots()
    ax.plot(p, prior, label="Prior")
    ax.plot(p, likelihood, label="Likelihood")
    ax.plot(p, posterior, label="Posterior")
    ax.legend()
    ax.set_title("Distribution of proportions of the earth being covered by water")
    ax.set_xlabel("Proportion p of earth covered in water")
    ax.set_ylabel("Frequency")
    plt.show()


def updating():
    """Shows updating by using the posterior as the next data points prior, see fig 2.6"""
    # sequence = ["W", "W", "L", "W", "L", "W", "L", "W", "W"]
    # sequence = ["W", "W", "W"]
    # sequence = ["W", "W", "W", "L"]
    sequence = ["L", "W", "W", "L", "W", "W", "W"]

    n_samples = 500
    p = np.linspace(0, 1, n_samples)  # proportion which is water
    # prior = np.array([1] * n_samples)
    prior = np.array([1 if i > (n_samples / 2) else 0 for i in range(n_samples)])
    prior = prior / sum(prior)

    likelihood_water = binom.pmf(1, 1, p)
    likelihood_water = likelihood_water / sum(likelihood_water)
    likelihood_land = binom.pmf(0, 1, p)
    likelihood_land = likelihood_land / sum(likelihood_land)

    likelihood_dict = {"W": likelihood_water, "L": likelihood_land}

    distributions = [prior]
    for i, s in enumerate(sequence):
        posterior = distributions[i] * likelihood_dict[s]
        posterior = posterior / sum(posterior)
        distributions.append(posterior)

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(8, 8))

    for i, s in enumerate(sequence):

        row = i % 3
        col = int((i - row) / 3)
        prior = distributions[i]
        posterior = distributions[i + 1]
        ax[col, row].plot(
            p, prior, label="Prior", linestyle="--", linewidth=1, color="red"
        )
        ax[col, row].plot(
            p,
            likelihood_dict[s],
            label="Likelihood",
            linestyle="--",
            linewidth=1,
            color="blue",
        )
        ax[col, row].plot(
            p, posterior, label="Prior", linestyle="--", linewidth=3, color="purple"
        )
        ax[col, row].set_title(f"Sampled {s}")
    fig.tight_layout()
    plt.show()


def globe_MCMC():
    """Shows MCMC sampling implementation. Due to bins not being centered etc. the
    results can be a little skewed, but the idea is clearly shown. See "Overthinking"
    page 45, R code 4.5 and 4.6
    """
    n_samples = 10000

    W = 6
    L = 3
    N = W + L

    # Analytical binomial
    n_grid = 100
    p_grid = np.linspace(0, 1, n_grid)  # proportion which is water
    likelihood = binom.pmf(W, N, p_grid)
    likelihood = likelihood / (sum(likelihood) * (p_grid[1] - p_grid[0]))

    p_samples = [0.5]  # Where to start the sampler.
    for i in range(1, n_samples):
        p = p_samples[i - 1]
        p_delta = norm.rvs(p, 1)
        p_new = p + p_delta
        if p_new < 0:
            p_new = -p_new
        if p_new > 1:
            p_new = 2 - p_new
        # Probability to generate W,L data given p water percentage
        q0 = binom.pmf(W, N, p)
        # Probability to generate W,L data given p_new water percentage
        q1 = binom.pmf(W, N, p_new)
        # Accept if the new percentage if higher. If not, then accept with probability
        # proportional to the ratio of their probabilities
        if uniform.rvs(0, 1) < q1 / q0:
            p_samples.append(p_new)
        else:
            p_samples.append(p)

    fig, ax = plt.subplots(ncols=2)

    ax[0].plot(p_samples)
    ax[0].set_xlabel("Sample number")
    ax[0].set_ylabel("Value of p sampled")

    ax[1].hist(p_samples, density=True, bins=100, label="Histogram of samples")
    ax[1].plot(p_grid, likelihood, label="Binom probability mass function")
    ax[1].set_xlabel("Value of p")
    ax[1].set_ylabel("Relative likelihood of p")
    ax[1].legend()
    plt.show()


def main():
    updating()


if __name__ == "__main__":
    main()
