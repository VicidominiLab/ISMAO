import CNNx3 as cnnx3
import CNNx3_transfer as tflx3
import DataAugmentation as dag
import seaborn as sns
import matplotlib.pyplot as plt
import Download_data as dd
import os

## %%CNNx3 Training

url_data = 'https://zenodo.org/records/13789465/files/correlograms_simulated.npy'
    name_data = 'correlograms_simulated.npy'
    if not os.path.isfile(name_data):
        dd.download(url_data, name_data)
    else:
        print('correlograms_simulated.npy already downloaded.')

    cor_path = np.load(name_data)

url_data = 'https://zenodo.org/records/13789465/files/correlograms_labels.npy.npy'
    name_data = 'correlograms_labels.npy.npy'
    if not os.path.isfile(name_data):
        dd.download(url_data, name_data)
    else:
        print('correlograms_labels.npy already downloaded.')

    lab_path = np.load(name_data)

cnnx3.CNNx3(cor_path, lab_path)

##%%CNNx3 Testing

pred_TR_x3, labe = cnnx3.CNNx3_test(cor_path, lab_path)

## %%CNNx3 TransferLearning

path_weights = r'files\modelx3.weights.h5'
path_file = r'files\modelx3.json'

pred, labe = tflx3.transfer_cnnx3(path_weights, path_file, cor_path, lab_path, cor_path, lab_path)

print(pred)
print(labe)

# %% Correlation Map

#dataset1 = np.abs(pred)
#dataset2 = np.abs(labe)
#correlation_matrix = np.corrcoef(dataset1.T, dataset2.T)
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix[:7, 7:], annot=True, cmap='coolwarm', linewidths=0.5,
#            xticklabels=['Z$_5$', 'Z$_6$', 'Z$_7$', 'Z$_8$', 'Z$_9$', '$Z_{10}$', '$Z_{11}$'],
#            yticklabels=['Z$_5$', 'Z$_6$', 'Z$_7$', 'Z$_8$', 'Z$_9$', '$Z_{10}$', '$Z_{11}$'])
##plt.title('Correlation Map')
#plt.show()
# plt.savefig(r'C:\Users\ffersini\OneDrive - Fondazione Istituto Italiano Tecnologia\FFersini\Conferences and Meeting\Paper\Correlation map Simulated data\CNNx3.pdf')

# %% DataAugmntation

#path = r'C:\Users\ffersini\OneDrive - Fondazione Istituto Italiano Tecnologia\FFersini\Data\Beads\GoodData\data-20-02-2024-12-21-47.h5'
#dataAugm = dag.data_augm(path, 2, 3)

