import time
import pandas as pd
import matplotlib.pyplot as plt
from apriori_python import apriori
from apyori import apriori as apriori2
from efficient_apriori import apriori as apriori3
from fpgrowth_py import fpgrowth

data = pd.read_csv("data.csv", names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
print(data.stack().value_counts(normalize=True))

transactions = []
for i in range(data.shape[0]):
    row = data.iloc[i].dropna().tolist()
    transactions.append(row)

print(data.head(5))
print(transactions[0][0])
print(transactions[0], '\n')

data.stack().value_counts(normalize=True)[:20].plot(kind="bar")
plt.show()
data.stack().value_counts()[:20].apply(lambda item: item / data.shape[0]).plot(kind="bar")
plt.show()

# Apriori_python
t = []
start = time.perf_counter()
t1, rules = apriori(transactions, minSup=0.03, minConf=0.4)
time1 = (time.perf_counter() - start)
t.append(time1)

print(rules, '\n\n')

# Apyori
start2 = time.perf_counter()
rules = apriori2(transactions=transactions, min_support=0.03, min_confidence=0.4, min_lift=1.0001)
results = list(rules)
time2 = (time.perf_counter() - start2)
t.append(time2)
print(results, "\n")
for result in results:
    for subset in result[2]:
        print(subset[0], subset[1])
        print("Support: {0}; Confidence: {1}; Lift:{2};".format(result[1], subset[2], subset[3]))
        print()


# Efficient_apriori
start3 = time.perf_counter()
itemsets, rules = apriori3(transactions, min_support=0.03, min_confidence=0.4)
time3 = (time.perf_counter() - start3)
t.append(time3)
for i in range(len(rules)):
    print(rules[i])


# Fpgrowth
start4 = time.perf_counter()
itemsets, rules = fpgrowth(transactions, minSupRatio=0.03, minConf=0.4)
time4 = (time.perf_counter() - start4)
t.append(time4)
for i in range(len(rules)):
    print(rules[i])


# Результаты по времени
print("Время выполнения apriori 1: ", t[0])
print("Время выполнения apriori 2: ", t[1])
print("Время выполнения apriori 3: ", t[2])
print("Время выполнения fpgrowth: ", t[3])
plt.bar(['apriori 1', 'apriori 2', 'apriori 3', 'fpgrowth'], t)
plt.show()
