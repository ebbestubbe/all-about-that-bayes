import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


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
    # Updating
    sequence = ["W", "W", "L", "W", "L", "W", "L", "W", "W"]
    n_samples = 500
    p = np.linspace(0, 1, n_samples)  # proportion which is water
    prior = np.array([1] * n_samples)
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


def main():
    updating()


if __name__ == "__main__":
    main()
