import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
cwd = os.getcwd()
mpgnn = f'{cwd}/MPGNN/lightning_logs/version_0/metrics.csv'
pe_mpgnn = f'{cwd}/PEMPGNN/lightning_logs/version_0/metrics.csv'
def plot(filename, label):

    df = pd.read_csv(filename)
    val_loss = df['val_loss'].dropna()
    plt.plot(val_loss, label=label)



plot(mpgnn, 'MPGNN')
plot(pe_mpgnn, 'PE-MPGNN')
plt.xlabel('Steps')
plt.ylabel('Validation Loss')
plt.title('Validation Losses')
plt.grid()
plt.legend()
plt.savefig('Validation_Loss_plot.png')