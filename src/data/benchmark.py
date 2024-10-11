from abc import ABC
from pathlib import Path
from typing import List


class RawData(ABC):
    def __init__(self,
                 folder_path:str = None,
                 file_paths:List = [],
                 name:str = None,
                 ):
        self.folder_path = folder_path
        self.file_paths = file_paths
        self.name = name


class ToolBench(RawData):
    def __init__(self, 
                 folder_path = None, 
                 file_paths = [], 
                 name = None):
        super().__init__(folder_path, file_paths, name)

    def data_loading(self):
        pass


class KaggleData(RawData):
    def __init__(self, 
                 folder_path = None, 
                 file_paths = [], 
                 name = None):
        super().__init__(folder_path, file_paths, name)

    def data_splitting(self):
        pass

    def thought_training_data(self):
        pass