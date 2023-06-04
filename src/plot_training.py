import matplotlib.pyplot as plt
from pathlib import Path

CSV = Path(r"logs\lightning_logs\version_0\metrics.csv")

with open(CSV, "r") as f:
    lines = f.readlines()

lines = [line.strip().split(",") for line in lines]
lines = lines[1:]

val_loss = [float(line[0]) for line in lines if line[0] != ""]
val_acc = [float(line[1]) for line in lines if line[1] != ""]

fig, ax = plt.subplots()

ax.plot(val_loss, color="red",label="Validation Loss" )
ax.plot(val_acc, color="green", label="Validation Accuracy")
legend = ax.legend(loc='center right', shadow=True, fontsize='x-large')

# Give a title for the sine wave plot
plt.title('Validation Loss and Accuracy')

# Give x axis label for theplot
plt.xlabel('Epoch')

plt.show()