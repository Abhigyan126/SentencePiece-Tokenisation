from scipy.special import digamma
import math

def manual_digamma(x, terms=10):
    """Computes digamma function ψ(x) using asymptotic expansion."""
    if x <= 0:
        return manual_digamma(1 - x) - math.pi / math.tan(math.pi * x)
    
    result = 0
    while x < 10:
        result -= 1 / x
        x += 1

    # Asymptotic expansion
    result += math.log(x) - 1 / (2 * x)
    coeffs = [1/6, -1/30, 1/42, -1/30, 5/66]  # Bernoulli numbers
    
    x2 = x * x
    power = x2
    for k in range(len(coeffs[:terms])):  # Adjust number of terms
        result -= coeffs[k] / (power * (2 * (k + 1)))
        power *= x2

    return result

# Testing at x = 5
x = 12
print(f"Manual Digamma (terms=10): {manual_digamma(x, terms=10)}")
print(f"SciPy Digamma: {digamma(x)}")
