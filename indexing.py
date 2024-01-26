import numpy as np
from numba import jit

'''
Utility functions that convert between different moment
indices specifically for radial, spherical and Zernike components.

Also calculate the specific number of polynomials needed to represent the expansion.
'''

"""RADIAL COMPONENTS"""

@jit(nopython=True)
def mu_R(n):
    """
    Calculates the upper bound of components for a given Zernike order `n`.

    Parameters:
        n (int): The maximum Zernike order.

    Returns:
        int: The upper bound of components in R.

    """
    if (n % 2) == 0:
        return int(0.25 * (n ** 2) + n)
    else:
        return int(0.25 * ((n ** 2) + (4 * n) + 2))


@jit(nopython=True)
def nl_to_mu_R(moment):
    """
    Converts the moment tuple (n, l) to the corresponding mu_R index.

    Parameters:
    moment (tuple): A tuple containing two integers, n and l, where n >= l.

    Returns:
    int: The mu_R index corresponding to the input moment.

    Raises:
    ValueError: If n < l.

    """
    n, l = moment
    if n == l:
        return mu_R(n)
    elif l < n:
        if (n % 2) == 0:
            return int(mu_R(n) - (n / 2) + (l / 2))
        else:
            return int(mu_R(n) - (n + 1) / 2 + (l + 1) / 2)


@jit(nopython=True)
def mu_R_to_nl(mu):
    """
    Converts a radial moment index mu to its corresponding (n, l) tuple.

    Parameters:
    mu (int): The radial moment index to convert to (n, l).

    Returns:
    (n, l) (tuple): The tuple representing the (n, l) values corresponding to the input mu.

    """
    n = 0
    while True:
        tot = mu_R(n)
        if mu < tot:
            mu -= mu_R(n - 1)
            break
        elif mu == tot:
            return (n, n)
        n += 1

    if (n % 2) == 0:
        l = 2 * mu - 2
    else:
        l = 2 * mu - 1
    return (n, l)

"""SPHERICAL COMPONENTS"""

@jit(nopython=True)
def mu_Y(l):
    """
    Calculates the upper bound of components for a given Zernike order `n`.

    Parameters:
        n (int): The maximum Zernike order.

    Returns:
        int: The upper bound of components in Y.

    """
    L = l + 1
    return int(0.5 * ((L ** 2) + L))


@jit(nopython=True)
def lm_to_mu_Y(moment):
    """
    Converts the moment tuple (l, m) to the corresponding mu_Y index.

    Parameters:
    moment (tuple): A tuple containing two integers, l and m.

    Returns:
    int: The mu_Y index corresponding to the input moment.
    """
    l, m = moment
    return int(0.5 * ((l ** 2) + l + 2) + m - 1)


@jit(nopython=True)
def mu_Y_to_lm(mu):
    """
    Converts a spherical moment index mu to its corresponding (l, m) tuple.

    Parameters:
    mu (int): The spherical moment index to convert to (l, m).

    Returns:
    (l, m) (tuple): The tuple representing the (l, m) values corresponding to the input mu.
    """
    l = 0
    while True:
        tot = mu_Y(l)
        if mu < tot:
            mu -= mu_Y(l - 1)
            return (l, mu)
        elif mu == tot:
            return (l + 1, 0)
        l += 1

"""ZERNIKE COMPONENTS"""

@jit(nopython=True)
def mu_(n):
    """
    Calculates the Zernike index mu for a given Zernike index n.

    Parameters:
    n (int): The Zernike index.

    Returns:
    int: The Zernike index mu.
    """
    if n < 100:
        sum = 0
        for k in range(0, n + 1):
            if (k % 2) == 0:
                sum += (n + 1 - k) * (k + 2)
            else:
                sum += (n + 1 - k) * (k + 1)
        return int(0.5 * sum) - 1
    else:
        e = np.arange(0, n + 1, 2)
        o = np.arange(1, n + 1, 2)
        return int(0.5 * (np.sum((n + 1 - e) * (e + 2)) + np.sum((n + 1 - o) * (o + 1)))) - 1


@jit(nopython=True)
def nlm_to_mu(moment):
    """
    Converts the moment tuple (n, l, m) to the corresponding mu index for Zernike components.

    Parameters:
    moment (tuple): A tuple containing three integers, n, l and m.

    Returns:
    int: The mu index corresponding to the input moment.
    """
    n, l, m = moment
    m = abs(m)
    if n == l:
        mu = mu_(n)
        return mu - l + m
    elif l < n:
        k = n
        mu = mu_(n)
        while l < k:
            if ((n - k) % 2) == 0:
                mu -= (k + 1)
            k -= 1
        return mu - l + m


@jit(nopython=True)
def mu_to_nlm(mu):
    """
    Converts a Zernike moment index mu to its corresponding unique (n, l, m) tuple.

    Parameters:
    mu (int): The Zernike moment index to convert to (n, l, m).

    Returns:
    (n, l, m) (tuple): The tuple representing the (n, l, m) values corresponding to the input mu.
    """
    n = 0
    while True:
        num_moments = mu_(n)
        if mu < num_moments:
            mu -= mu_(n - 1)
            break
        elif mu == num_moments:
            # mu = 0 # This does nothing?
            return (n, n, n)
        n += 1

    if n % 2 == 0:
        l = 0
    else:
        l = 1

    num_moments = l + 1
    while True:
        if mu < num_moments:
            mu -= num_moments
            break
        elif mu == num_moments:
            return (n, l, l)
        l += 2
        num_moments += (l + 1)
    m = l + mu
    return (n, l, m)

def memory_requirement(n, dim):
    Y_size_p = mu_Y(n) * (dim ** 3) * 8
    R_size_p = mu_R(n) * (dim ** 3) * 4
    total_size_gb = (Y_size_p + R_size_p) / 1e9
    return total_size_gb

def memory_requirement_all(n, dim):
    count = 0
    for N in range(0, n+1):
        for l in range(0, N+1):
            if (N - l) % 2 == 0:
                for m in range(l+1):
                    count += 1
    total_size_gb = (count * (dim ** 3) * 8) / 1e9
    return total_size_gb

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 5

    print(f"Zernike order n={n}, requires {mu_R(n)} radial & {mu_Y(n)} spherical components.")

    dimensions = [64, 96, 128, 156, 196, 220, 256]
    memory_requirements = []
    memory_requirements_all = []

    for dim in dimensions:
        memory_requirements.append(memory_requirement(n, dim))
        memory_requirements_all.append(memory_requirement_all(n, dim))
        print(f"Predicted memory requirement for shape ({dim}, {dim}, {dim}): {memory_requirement(n, dim):.3f} Gb")

    # Plot the results
    plt.plot(dimensions, memory_requirements, marker='o', linestyle='-', color='b', label='Memory Requirement (Zernike3D, spherical harmonics)')
    plt.plot(dimensions, memory_requirements_all, marker='o', linestyle='dashed', color='red', label='Naive computation of all components (Monomial approach)')
    plt.title('Memory Requirements vs. Volume Dimensions')
    plt.xlabel('Volume Dimensions')
    plt.ylabel('Memory Requirement (Gb)')
    #plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.grid(False)
    plt.legend()  # Display the legend
    plt.show()