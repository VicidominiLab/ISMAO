import CNNx1 as cnnx1
import CNNx1_transfer as tflx1
import numpy as np
import DataAugmentation as dag
import seaborn as sns
import matplotlib.pyplot as plt


## %%CNNx1 Training

cor_path = r'C:\Users\ffersini\Desktop\files\Cor.npy'
lab_path = r'C:\Users\ffersini\Desktop\files\Lab.npy'

cnnx1.CNNx1(cor_path,lab_path)

## %%CNNx1 Testing

test_cor_path = np.load(r'C:\Users\ffersini\Desktop\files\Cor.npy')
labe_cor_path = np.load(r'C:\Users\ffersini\Desktop\files\Lab.npy')

pred_TR_x1, labe = cnnx1.CNNx1_test(test_cor_path, labe_cor_path)

## %%CNNx1 TransferLearning

path_weights = r'C:\Users\ffersini\Desktop\files\modelx1.weights.h5'
path_file = r'C:\Users\ffersini\Desktop\files\modelx1.json'

new_file = r'C:\Users\ffersini\Desktop\files\Cor.npy'
new_label = r'C:\Users\ffersini\Desktop\files\Lab.npy'

test_cor_path = r'C:\Users\ffersini\Desktop\files\Cor.npy'
labe_cor_path = r'C:\Users\ffersini\Desktop\files\Lab.npy'

pred, labe = tflx1.transfer_cnnx1(path_weights, path_file, new_file, new_label, test_cor_path, labe_cor_path)

print(pred)
print(labe)


## %% Correlation Map

#dataset1 = np.abs(pred)
#dataset2 = np.abs(labe)
#correlation_matrix = np.corrcoef(dataset1.T, dataset2.T)
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix[:7, 7:], annot=True, cmap='coolwarm', linewidths=0.5,
#   xticklabels=['Z$_5$', 'Z$_6$', 'Z$_7$', 'Z$_8$', 'Z$_9$', '$Z_{10}$', '$Z_{11}$'],
#      yticklabels=['Z$_5$', 'Z$_6$', 'Z$_7$', 'Z$_8$', 'Z$_9$', '$Z_{10}$', '$Z_{11}$'])
#plt.title('Correlation Map')
#plt.show()

# %% DataAugmntation

#path = r'C:\Users\ffersini\Desktop\files\data-20-02-2024-12-21-47.h5'
#dataAugm = dag.data_augm(path, 2, 3)
