import os
import numpy as np
import scipy.io as sio
import math


class data_get_func():
    def __init__(self, dataname, file_path):
        if dataname == 'Jasper Ridge':
            hsi = sio.loadmat('/data/zzy/HSIU/jasper.mat')
            self.data_hsi = (hsi['Y'] - hsi['Y'].min()) / (hsi['Y'].max() - hsi['Y'].min())
            self.data_aban = hsi['A']
            self.data_endm = hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Urban':
            # mat_hsi = sio.loadmat(os.path.join(file_path, 'Urban_R162.mat'))
            # mat_gt = sio.loadmat(os.path.join(file_path, 'end4_groundTruth.mat'))
            # self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            # self.data_aban = mat_gt['A']
            # self.data_endm = mat_gt['M']
            # self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            # self.num_bands = self.data_hsi.shape[0]
            # self.num_endm = self.data_endm.shape[1]

            mat_hsi = sio.loadmat('/data/zzy/HSIU/Urban6.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            h, w, c = mat_hsi['S_GT'].shape
            self.data_aban = np.asarray(mat_hsi['S_GT'], dtype=np.float32).reshape(c, h * w)
            self.data_endm = mat_hsi['GT'].T
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Apex':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/apex.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_hsi['A']
            self.data_endm = mat_hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Samson':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/samson_dataset.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_hsi['A']
            self.data_endm = mat_hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'houston':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/houston.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            h, w, c = mat_hsi['S_GT'].shape
            self.data_aban = np.asarray(mat_hsi['S_GT'], dtype=np.float32).reshape(c, h * w)
            self.data_endm = mat_hsi['GT'].T
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'moffett':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/moffett.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            h, w, c = mat_hsi['S_GT'].shape
            self.data_aban = np.asarray(mat_hsi['S_GT'], dtype=np.float32).reshape(c, h * w)
            self.data_endm = mat_hsi['GT'].T
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'DC1':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/simulate/30db/data_ex4.mat')
            self.data_hsi = (mat_hsi['r'] - mat_hsi['r'].min()) / (mat_hsi['r'].max() - mat_hsi['r'].min())
            self.data_aban = mat_hsi['alphas']
            self.data_endm = mat_hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        elif dataname == 'Cuprite':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/Cuprite/cuprite_ref.mat')
            end = sio.loadmat('/data/zzy/HSIU/Cuprite/cuprite_end_ref_188_12.mat')
            self.data_hsi = (mat_hsi['x'] - mat_hsi['x'].min()) / (mat_hsi['x'].max() - mat_hsi['x'].min())
            self.data_endm = end['endmembers']
            self.num_rows = 250
            self.num_cols = 191
            self.data_aban = None
            self.num_bands = 188
            self.num_endm = 12

        elif dataname == 'synthetic':
            mat_hsi = sio.loadmat('/data/zzy/HSIU/synthetic/synthetic.mat')
            self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
            self.data_aban = mat_hsi['A']
            self.data_endm = mat_hsi['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]


        elif dataname == 'orchard':
            mat_data = sio.loadmat('/data/zzy/HSIU/orchard/orchard_processed.mat')
            mat_hsi = mat_data['Y']
            if np.isnan(mat_hsi).any():
                mat_hsi = np.nan_to_num(mat_hsi)
            self.data_hsi = (mat_hsi - mat_hsi.min()) / (mat_hsi.max() - mat_hsi.min())
            self.data_aban = mat_data['A']
            self.data_endm = mat_data['M']
            self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
            self.num_bands = self.data_hsi.shape[0]
            self.num_endm = self.data_endm.shape[1]

        # elif dataname == 'DC2':
        #     EE = sio.loadmat('/data/zzy/HSIU/simulate/0db/DC1/EE.mat')
        #     XT = sio.loadmat('/data/zzy/HSIU/simulate/0db/DC1/XT.mat')
        #     Y_clean = sio.loadmat('/data/zzy/HSIU/simulate/0db/DC1/Y_clean.mat')
        #     mat_hsi = sio.loadmat(os.path.join(file_path, 'Urban_R162.mat'))
        #     mat_gt = sio.loadmat(os.path.join(file_path, 'end4_groundTruth.mat'))
        #     self.data_hsi = (mat_hsi['Y'] - mat_hsi['Y'].min()) / (mat_hsi['Y'].max() - mat_hsi['Y'].min())
        #     self.data_aban = mat_gt['A']
        #     self.data_endm = mat_gt['M']
        #     self.num_cols = self.num_rows = int(math.sqrt(self.data_hsi.shape[1]))
        #     self.num_bands = self.data_hsi.shape[0]
        #     self.num_endm = self.data_endm.shape[1]


    def get_hsi_mean(self):
        hsi_mean = np.mean(self.data_hsi, axis=1)
        if len(hsi_mean.shape) == 2:
            hsi_mean = np.mean(hsi_mean, axis=0)
        hsi_mean = np.repeat(hsi_mean, self.num_endm).reshape(-1, self.num_endm).T
        return hsi_mean


