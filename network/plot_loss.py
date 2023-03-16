import matplotlib.pyplot as plt

log_file = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/Output/Wnet_Training_Unet_120_epochs_small/result/logfile.txt"
# Read data from txt file, ignoring the first line (header)
with open(log_file) as f:
    data = f.readlines()[1:]

epochs = []
train_loss = []
#topo_loss = []
val_loss = []
#total_loss = []

# Split data into epochs, training loss, and validation loss
for line in data:
    loss = line.strip().split(",")
    epochs.append(int(loss[0]) + 1)
    train_loss.append(float(loss[1]))
    val_loss.append(float(loss[2]))
    #topo_loss.append(float(loss[3]))
    #total_loss.append(float(loss[4]))

# Plot the data
plt.plot(epochs, train_loss, label="Training Loss")
#plt.plot(epochs, topo_loss, label="Topo Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss per Epoch")
plt.savefig("loss.png")
plt.show()