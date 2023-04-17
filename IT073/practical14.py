import numpy as np
import skfuzzy as fuzz

# Define the input fuzzy sets
x = np.arange(0, 11, 1)
a = fuzz.trimf(x, [0, 2, 4])
b = fuzz.trimf(x, [3, 5, 7])

# Compute the fuzzy union of a and b
a_or_b = fuzz.fuzzy_or(x, a, x, b)

# Compute the fuzzy intersection of a and b
a_and_b = fuzz.fuzzy_and(x, a, x, b)

# Compute the fuzzy complement of a
a_comp = fuzz.fuzzy_not(a)
b_comp = fuzz.fuzzy_not(b)

# Display the results
print("Input fuzzy sets:\n", "a:", a, "\n", "b:", b)
print("\nFuzzy union of a and b:\n", a_or_b)
print("\nFuzzy intersaction of a and b:\n", a_or_b)
print("\nFuzzy complement of a:\n", a_comp)
print("\nFuzzy complement of b:\n", b_comp)

