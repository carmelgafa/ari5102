import random
import matplotlib.pyplot as plt


random.seed(9999)

SAMPLE_SIZE = 40
NUMBER_OF_RUNS = 1000


die_rolls = []
run_averages = []


for i in range(NUMBER_OF_RUNS):
    run_die_rolls = random.choices([1,2,3,4,5,6], [.4,.2,.1,.1,.1,.1], k=SAMPLE_SIZE)
    die_rolls += run_die_rolls

    run_average = sum(run_die_rolls) / SAMPLE_SIZE
    run_averages.append(run_average)


plt.hist(die_rolls, bins=6)
plt.show()

plt.hist(run_averages, bins=10)
plt.show()
