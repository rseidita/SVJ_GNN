import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(training_losses, validation_losses, output_path, constraints_manager=None):

    if not os.path.exists(f"{output_path}/plots"):
        os.makedirs(f"{output_path}/plots")

    plt.plot(training_losses, label="Total training loss", marker="o", color="blue")
    plt.plot(validation_losses, label="Total validation loss", marker="o", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_path}/plots/loss.pdf")
    plt.close()

    if constraints_manager is not None:
        for name in constraints_manager.get_constraint_names():
            plt.plot(training_losses["constraints_losses"][name], label=f"Training {name}")
            plt.plot( validation_losses["constraints_losses"][name], label=f"Validation {name}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{output_path}/plots/{name}.pdf")
            plt.close()


def plot_rocs(result, fpr, tpr, aucx, output_path):
    
    plt.figure(figsize=(10, 7))

    for signal_name in aucx.keys():
        plt.plot(fpr[signal_name], tpr[signal_name], label=f"AUC = {aucx[signal_name]:.3f}")

# Set the labels and title outside the loop
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC Curve for All Signals, Mean AUC: {result}")
    plt.legend()

# Save the combined plot to a file
    file_path3 = os.path.join(output_path, f'Combined_ROC_.png')
    plt.savefig(file_path3)
