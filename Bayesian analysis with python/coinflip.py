import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def likelihood():
    """Example page 23 - Show likelihood of getting m heads on n tosses
    Remember that we do not actually know theta - the chance of getting heads on a
    coinflip, but this is how it is illustrated if we knew.
    """
    n_params = [1, 2, 10]  # Number of trials
    p_params = [0.25, 0.5, 0.75]  # Probability of success

    x = np.arange(0, max(n_params) + 1)
    fig, ax = plt.subplots(
        len(n_params),
        len(p_params),
        sharex=True,
        sharey=True,
        figsize=(8, 7),
        constrained_layout=True,
    )

    for i in range(len(n_params)):
        for j in range(len(p_params)):
            n = n_params[i]
            p = p_params[j]

            y = stats.binom(n=n, p=p).pmf(x)

            ax[i, j].vlines(x, 0, y, colors="C0", lw=5)
            ax[i, j].set_ylim(0, 1)
            ax[i, j].plot(0, 0, label="N = {:3.2f}\nθ = {:3.2f}".format(n, p), alpha=0)
            ax[i, j].legend()

            ax[2, 1].set_xlabel("y - heads flipped")
            ax[1, 0].set_ylabel("p(y | θ, N) - Chance of getting y heads")
            ax[0, 0].set_xticks(x)
    fig.suptitle("Coinflip likelihood", fontsize=16)


def prior():
    """Notice
    - Different shapes for different parameters - approximating peaking at
    edges, normal distributions, uniform, linear.
    - Restricted between 0 and 1.
    - Conjugate prior of binomial distribution. Meaning that it returns a beta
    distribution for the posterior when combined with a binomial likelihood. This is
    convenient for analytical solutions, but is less important for numerical solutions.
    Further, conjugate priors may give intuition, by more transparently showing how a
    likelihood function updates a prior distribution.
    """
    params = [0.5, 1, 2, 3]
    x = np.linspace(0, 1, 100)
    f, ax = plt.subplots(
        len(params),
        len(params),
        sharex=True,
        sharey=True,
        figsize=(8, 7),
        constrained_layout=True,
    )
    for i in range(4):
        for j in range(4):
            a = params[i]
            b = params[j]
            y = stats.beta(a, b).pdf(x)
            ax[i, j].plot(x, y)
            ax[i, j].plot(0, 0, label="α = {:2.1f}\nβ = {:2.1f}".format(a, b), alpha=0)
            ax[i, j].legend()
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xticks([0, 0.5, 1])
    f.text(0.5, 0.05, "θ", ha="center")
    f.text(0.07, 0.5, "p(θ)", va="center", rotation=0)


def posterior():
    """Given the conjugate prior, the posterior becomes a beta distribution that we can
    plot directly given trials N and number of heads y.
    """
    plt.figure(figsize=(10, 8))

    # n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
    # data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]

    n_trials = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    data = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    theta_real = 1 / 2

    beta_params = [(1, 1), (20, 20), (1, 4)]
    dist = stats.beta
    x = np.linspace(0, 1, 200)

    for idx, N in enumerate(n_trials):
        if idx == 0:
            plt.subplot(4, 3, 2)
            plt.xlabel("θ")
        else:
            plt.subplot(4, 3, idx + 3)
            # plt.xticks([])
        y = data[idx]
        for (a_prior, b_prior) in beta_params:
            p_theta_given_y = dist.pdf(x, a_prior + y, b_prior + N - y)
            plt.fill_between(x, 0, p_theta_given_y, alpha=0.7)

        plt.axvline(theta_real, ymax=0.3, color="k")
        plt.plot(0, 0, label=f"{N:4d} trials\n{y:4d} heads", alpha=0)
        plt.xlim(0, 1)
        plt.ylim(0, 12)
        plt.legend()
        plt.yticks([])


if __name__ == "__main__":
    posterior()

    plt.show()
