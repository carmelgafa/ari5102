import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# generate population
population = np.random.normal(loc=50, scale=10, size=10000)

# calculate std error of mean
def compute_sem(sample_sizes, n_repeats=1000):
    sems = []
    for n in sample_sizes:
        sample_means = []
        for _ in range(n_repeats):
            sample = np.random.choice(population, size=n, replace=False)
            sample_means.append(np.mean(sample))
        sem = np.std(sample_means, ddof=1)
        sems.append(sem)
    return sems

# Define different sample sizes to test
sample_sizes = np.arange(10, 5000, 50)
sems = compute_sem(sample_sizes)

# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x=sample_sizes, y=sems)
plt.xlabel('Sample Size')
plt.ylabel('Standard Error of Mean')
plt.title('SEM vs. Sample Size')
plt.grid(True)
plt.show()
