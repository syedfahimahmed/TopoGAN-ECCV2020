import matplotlib.pyplot as plt

performance_file = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/Output/Topo_Training_Unet_120_epochs_small/result/performance.txt"
performance_file_wnet = "C:/Users/syed_fahim_ahmed/Desktop/Coding_With_Fahim/Unet_MC/TopoSegNetSimple/Output/Wnet_Training_Unet_120_epochs_small/result/performance.txt"

# Read data from txt file, skipping the first line which is assumed to be the header
with open(performance_file) as f:
    data = f.readlines()[1:]

epoch = []
acc = []
prec = []
rec = []
f1 = []
l1_diff = []

# Split data into epoch, accuracy, precision, recall, F1 score, and L1 difference per image
for line in data:
    vals = line.strip().split(",")
    epoch.append(int(vals[0])+1)
    acc.append(float(vals[1]))
    prec.append(float(vals[2]))
    rec.append(float(vals[3]))
    f1.append(float(vals[4]))
    l1_diff.append(float(vals[5]))
    

# Read data from txt file, skipping the first line which is assumed to be the header
with open(performance_file_wnet) as f_wnet:
    data_wnet = f_wnet.readlines()[1:]

epoch_wnet = []
acc_wnet = []
prec_wnet = []
rec_wnet = []
f1_wnet = []
l1_diff_wnet = []

# Split data into epoch, accuracy, precision, recall, F1 score, and L1 difference per image
for line_wnet in data_wnet:
    vals_wnet = line_wnet.strip().split(",")
    epoch_wnet.append(int(vals_wnet[0])+1)
    acc_wnet.append(float(vals_wnet[1]))
    prec_wnet.append(float(vals_wnet[2]))
    rec_wnet.append(float(vals_wnet[3]))
    f1_wnet.append(float(vals_wnet[4]))
    l1_diff_wnet.append(float(vals_wnet[5]))


# Plot accuracy and save the figure
plt.plot(epoch, acc, label="Refinement Training")
plt.plot(epoch_wnet, acc_wnet, label="WNet")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy per Epoch")
plt.savefig("accuracy.png")
plt.show()

# Plot precision and save the figure
plt.plot(epoch, prec, label="Refinement Training")
plt.plot(epoch_wnet, prec_wnet, label="WNet")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision per Epoch")
plt.savefig("precision.png")
plt.show()

# Plot recall and save the figure
plt.plot(epoch, rec, label="Refinement Training")
plt.plot(epoch_wnet, rec_wnet, label="WNet")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.legend()
plt.title("Recall per Epoch")
plt.savefig("recall.png")
plt.show()

# Plot F1 score and save the figure
plt.plot(epoch, f1, label="Refinement Training")
plt.plot(epoch_wnet, f1_wnet, label="WNet")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1 Score per Epoch")
plt.savefig("f1_score.png")
plt.show()

# Plot L1 difference per image and save the figure
plt.plot(epoch, l1_diff, label="Refinement Training")
plt.plot(epoch_wnet, l1_diff_wnet, label="WNet")
plt.xlabel("Epoch")
plt.ylabel("L1 Difference")
plt.legend()
plt.title("L1 Difference per Epoch")
plt.savefig("l1_difference.png")
plt.show()