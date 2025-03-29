import random
import matplotlib.pyplot as plt


# random.seed(9999)

SAMPLE_SIZE = 100
NUMBER_OF_RUNS = 1000
ones_percentages = []


for i in range(NUMBER_OF_RUNS):
    selections = random.choices([0,1], [0.6, 0.4], k=SAMPLE_SIZE)
    ones = selections.count(1)
    ones_percentages.append(ones / SAMPLE_SIZE)


plt.hist(ones_percentages, bins=10)

plt.show()

