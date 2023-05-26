import numpy as np
from torch.utils.data import Dataset

from typing import Dict, List, Tuple, Union

import multiearth_challenge.datasets.data_filters as df
from multiearth_challenge.datasets import translation_dataset as td
from multiearth_challenge.datasets import base_datasets as bd

DatasetData = Dict[str, Union[np.ndarray, str,
                              List[str], Tuple[float, float], np.datetime64, None]]


class SARToEOPairBase(Dataset):
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1025)
        eo = (eo / 1025.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -18.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -25.0, -7.0) # VH
        sar[0,...] = sar[0,...]+25.0
        sar[1,...] = sar[1,...]+25.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        # sar = (sar / 18.0 * 2.0 - 1.0).astype(np.float32)
        sar = (sar / 25.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example
    
class SARToEOPairBaseV2(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1025)
        eo = (eo / 1025.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -18.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -25.0, -7.0) # VH
        sar[0,...] = sar[0,...]+18.0
        sar[1,...] = sar[1,...]+25.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 18.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example
        
class SARToEOPairBaseV3(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1025)
        eo = (eo / 1025.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example
    
class SARToEOPairBaseV4(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1250.0)
        eo = (eo / 1250.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example

class SARToEOPairBaseV5(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1500.0)
        eo = (eo / 1500.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example
    
class SARToEOPairBaseV6(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 1750.0)
        eo = (eo / 1750.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example
    
class SARToEOPairBaseV7(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                eo_file: str,
                size=None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        visible_files = [eo_file]
        sar_bands =["VV", "VH"]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        single_target_image = False
        merge_sar_bands = True
        merge_visible_bands = True
        error_on_empty = True
        sar_date_window = (-7, 7)
        max_visible_cloud_coverage = 0.0
        
        self.dataset = td.SARToVisibleDatasetPair(
                                                sar_files,
                                                visible_files,
                                                sar_bands,
                                                merge_sar_bands,
                                                sar_date_window,
                                                visible_bands,
                                                merge_visible_bands,
                                                max_visible_cloud_coverage,
                                                single_target_image,
                                                error_on_empty,       
                                            )  
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> Dict[str, bd.DatasetData]:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        sar = data['SAR']['image'].copy() # [VV, VH] x 256 x 256
        eo = data['EO']['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        
        eo = np.clip(eo, 0., 2000.0)
        eo = (eo / 2000.0 * 2.0 - 1.0).astype(np.float32)
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        # 
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            eo = np.flip(eo, axis=2)
            
        
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        eo = eo.transpose(1, 2, 0)
        
        example['image'] = eo.copy()
        example['image2'] = sar.copy()
        # return {'image': eo, 'image2': sar, 'data': data}
        return example

        
class SARBase(Dataset):
    def __init__(self,
                sar_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        sar_bands =["VV", "VH"]
        sar_data_filters = [
            df.DataBandFilter({"Sentinel-1": sar_bands}, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(sar_files[0], sar_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [sar_file for l in range(self._length)],
            'file_path_': [sar_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        sar = data['image'].copy() # [VV, VH] x 256 x 256
        
        ################
        # Normalize
        ################
        
        sar[0,...] = np.clip(sar[0,...], -18.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -25.0, -7.0) # VH
        sar[0,...] = sar[0,...]+25.0
        sar[1,...] = sar[1,...]+25.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        
        sar = (sar / 25.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        example['image'] = sar.copy()
        
        # return {'image': sar, 'data': data}
        return example
    
class SARBaseV2(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        sar_bands =["VV", "VH"]
        sar_data_filters = [
            df.DataBandFilter({"Sentinel-1": sar_bands}, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(sar_files[0], sar_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [sar_file for l in range(self._length)],
            'file_path_': [sar_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        sar = data['image'].copy() # [VV, VH] x 256 x 256
        
        ################
        # Normalize
        ################
        
        sar[0,...] = np.clip(sar[0,...], -18.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -25.0, -7.0) # VH
        sar[0,...] = sar[0,...]+18.0
        sar[1,...] = sar[1,...]+25.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        
        sar = (sar / 18.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        example['image'] = sar.copy()
        
        # return {'image': sar, 'data': data}
        return example
    
class SARBaseV3(Dataset): # SAR Different normalization
    def __init__(self,
                sar_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        sar_files = [sar_file]
        sar_bands =["VV", "VH"]
        sar_data_filters = [
            df.DataBandFilter({"Sentinel-1": sar_bands}, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(sar_files[0], sar_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [sar_file for l in range(self._length)],
            'file_path_': [sar_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        sar = data['image'].copy() # [VV, VH] x 256 x 256
        
        ################
        # Normalize
        ################
        
        sar[0,...] = np.clip(sar[0,...], -30.0, 0.0) # VV
        sar[1,...] = np.clip(sar[1,...], -30.0, 0.0) # VH
        sar[0,...] = sar[0,...]+30.0
        sar[1,...] = sar[1,...]+30.0
        vv_vh = (sar[0,...]+sar[1,...])/2.
        sar = np.vstack((sar, np.expand_dims(vv_vh, axis=0)))
        
        sar = (sar / 30.0 * 2.0 - 1.0).astype(np.float32)
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=1)
        if np.random.uniform() < self.flip_p:
            sar = np.flip(sar, axis=2)
            
        ################
        # Dimension check
        ################
        
        sar = sar.transpose(1, 2, 0)
        example['image'] = sar.copy()
        
        # return {'image': sar, 'data': data}
        return example
        
class EOBase(Dataset):
    def __init__(self,
                eo_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        eo_files = [eo_file]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        eo_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.DataBandFilter(visible_bands, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(eo_files[0], eo_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        eo = data['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        eo = np.clip(eo, 0., 1025)
        eo = (eo / 1025 * 2.0 - 1.0).astype(np.float32)
        
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=2)
        
        ################
        # Dimension check
        ################
        
        eo = eo.transpose(1, 2, 0)
        example['image'] = eo.copy()
        
        # return {'image': eo, 'data': data}
        return example

class EOBaseV2(Dataset):
    def __init__(self,
                eo_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        eo_files = [eo_file]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        eo_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.DataBandFilter(visible_bands, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(eo_files[0], eo_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        eo = data['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        eo = np.clip(eo, 0., 1250.)
        eo = (eo / 1250.0 * 2.0 - 1.0).astype(np.float32)
        
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=2)
        
        ################
        # Dimension check
        ################
        
        eo = eo.transpose(1, 2, 0)
        example['image'] = eo.copy()
        
        # return {'image': eo, 'data': data}
        return example
    
class EOBaseV3(Dataset):
    def __init__(self,
                eo_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        eo_files = [eo_file]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        eo_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.DataBandFilter(visible_bands, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(eo_files[0], eo_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        eo = data['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        eo = np.clip(eo, 0., 1500.)
        eo = (eo / 1500.0 * 2.0 - 1.0).astype(np.float32)
        
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=2)
        
        ################
        # Dimension check
        ################
        
        eo = eo.transpose(1, 2, 0)
        example['image'] = eo.copy()
        
        # return {'image': eo, 'data': data}
        return example
    
class EOBaseV4(Dataset):
    def __init__(self,
                eo_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        eo_files = [eo_file]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        eo_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.DataBandFilter(visible_bands, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(eo_files[0], eo_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        eo = data['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        eo = np.clip(eo, 0., 1750.)
        eo = (eo / 1750.0 * 2.0 - 1.0).astype(np.float32)
        
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=2)
        
        ################
        # Dimension check
        ################
        
        eo = eo.transpose(1, 2, 0)
        example['image'] = eo.copy()
        
        # return {'image': eo, 'data': data}
        return example

class EOBaseV5(Dataset):
    def __init__(self,
                eo_file: str,
                size: None,
                flip_p=0.5):
        
        self.flip_p = flip_p
        eo_files = [eo_file]
        visible_bands = {
            "Sentinel-2": ["B4", "B3", "B2"],
        }
        eo_data_filters = [
            # Cloud filtering must come before the QA bands may be
            # filtered out
            df.DataBandFilter(visible_bands, include=True),
        ]
        merge_sar_bands = True
        self.dataset = bd.NetCDFDataset(eo_files[0], eo_data_filters, merge_sar_bands)
        
        self._length = len(self.dataset)
        self.labels = {
            'relative_filte_path_': [eo_file for l in range(self._length)],
            'file_path_': [eo_file for l in range(self._length)]
        }
        
    def __len__(self) -> int:
        """Get the number of target samples in this dataset.

        Returns
        -------
        int
            The number of target samples in this dataset.
        """
        return self.dataset.__len__()
    
    def __getitem__(self, index: int) -> DatasetData:
        
        example = dict((k, self.labels[k][index]) for k in self.labels)
        
        data = self.dataset.__getitem__(index)
        # <class 'dict'>
        # dict_keys(['image', 'data_source', 'bands', 'lat_lon', 'date'])
        
        eo = data['image'].copy() # [B4, B3, B2] x 256 x 256
        
        ################
        # Normalize
        ################
        eo = np.clip(eo, 0., 2000.)
        eo = (eo / 2000.0 * 2.0 - 1.0).astype(np.float32)
        
        
        ################
        # Flip
        ################
        
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=1)
        if np.random.uniform() < self.flip_p:
            eo = np.flip(eo, axis=2)
        
        ################
        # Dimension check
        ################
        
        eo = eo.transpose(1, 2, 0)
        example['image'] = eo.copy()
        
        # return {'image': eo, 'data': data}
        return example
        
        
    
class SARToEOPairTrain(SARToEOPairBase):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_0per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_0per_train.nc', **kwargs)
        
class SARToEOPairValidation(SARToEOPairBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)
        
class SARToEOPairTrainV2(SARToEOPairBaseV2):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_0per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_0per_train.nc', **kwargs)
        
class SARToEOPairValidationV2(SARToEOPairBaseV2):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)

class SARToEOPairTrainV3(SARToEOPairBaseV3):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_0per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_0per_train.nc', **kwargs)
        
class SARToEOPairValidationV3(SARToEOPairBaseV3):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)
        
class SARToEOPairTrainV4(SARToEOPairBaseV4):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_0per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_0per_train.nc', **kwargs)
        
class SARToEOPairValidationV4(SARToEOPairBaseV4):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/disk1/zlatkd12/multiearth-challenge/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/disk1/zlatkd12/multiearth-challenge/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)

class SARToEOPairTrainV5(SARToEOPairBaseV5):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/unist/multiearch2023/train_test/sent2_1per_train.nc', **kwargs)
        
class SARToEOPairValidationV5(SARToEOPairBaseV5):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/unist/multiearch2023/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)
        
class SARToEOPairTrainV6(SARToEOPairBaseV6):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/unist/multiearch2023/train_test/sent2_1per_train.nc', **kwargs)
        
class SARToEOPairValidationV6(SARToEOPairBaseV6):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/unist/multiearch2023/train_test/sent2_1per_test.nc', flip_p=flip_p, **kwargs)
        
class SARToEOPairTrainV7(SARToEOPairBaseV7):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/lait/dataset/multiearth2023_dataset/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/lait/dataset/multiearth2023_dataset/sent2_1per_train.nc', **kwargs)
        
class SARToEOPairValidationV7(SARToEOPairBaseV7):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(sar_file='/home/lait/dataset/multiearth2023_dataset/sent1_train_nodata_filtered_pair_1per.nc', eo_file='/home/lait/dataset/multiearth2023_dataset/sent2_1per_test.nc', flip_p=flip_p, **kwargs)





class SARTrain(SARBase):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_train.nc', **kwargs)
        
class SARValidation(SARBase):
    def __init__(self, flip_p=0, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_test.nc', flip_p=flip_p, **kwargs)
        
class SARTrainV2(SARBaseV2):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_train.nc', **kwargs)
        
class SARValidationV2(SARBaseV2):
    def __init__(self, flip_p=0, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_test.nc', flip_p=flip_p, **kwargs)
        
class SARTrainV3(SARBaseV3):
    def __init__(self, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_train.nc', **kwargs)
        
class SARValidationV3(SARBaseV3):
    def __init__(self, flip_p=0, **kwargs):
        super().__init__(sar_file='/home/unist/multiearch2023/train_test/sent1_0per_test.nc', flip_p=flip_p, **kwargs)
        
        
class EOTrain(EOBase):
    def __init__(self, **kwargs):
        super().__init__(eo_file='/home/unist/multiearch2023/train_test/sent2_0per_train.nc', **kwargs)
        
class EOValidation(EOBase):
    def __init__(self, flip_p=0, **kwargs):
        super().__init__(eo_file='/home/unist/multiearch2023/train_test/sent2_0per_test.nc', flip_p=flip_p, **kwargs)
        
class EOTrainV2(EOBaseV2):
    def __init__(self, **kwargs):
        super().__init__(eo_file='/home/unist/multiearch2023/train_test/sent2_0per_train.nc', **kwargs)
        
class EOValidationV2(EOBaseV2):
    def __init__(self, flip_p=0, **kwargs):
        super().__init__(eo_file='/home/unist/multiearch2023/train_test/sent2_0per_test.nc', flip_p=flip_p, **kwargs)
        
        
        

class SARTestV1(SARBase):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', flip_p=flip_p, return_data=True, **kwargs)

class SARTestV2(SARBaseV2):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', flip_p=flip_p, return_data=True, **kwargs) 
        
class SARTestV3(SARBaseV3):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', flip_p=flip_p, return_data=True, **kwargs)
        
        
        
class SARToEOPairTestV1(SARToEOPairBase):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', eo_file='', flip_p=flip_p, return_data=True, **kwargs)
        
        
class SARToEOPairTestV2(SARToEOPairBaseV2):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', eo_file='', flip_p=flip_p, return_data=True, **kwargs)

        
class SARToEOPairTestV3(SARToEOPairBaseV3):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', eo_file='', flip_p=flip_p, return_data=True, **kwargs)
        
class SARToEOPairTestV4(SARToEOPairBaseV4):
    def __init__(self, flip_p=-1, **kwargs):
        super().__init__(sar_file='', eo_file='', flip_p=flip_p, return_data=True, **kwargs)
