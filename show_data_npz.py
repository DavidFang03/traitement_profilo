import numpy as np

class ThreeD_Data:
    def __init__(self, npz_path, **kwargs):
        '''
        Import data from npz file.
        Attributes :
        - arrS
        - Y
        - height
        - scale
        - theta
        - profiles
        '''
        # ! IMPORT
        
        self.npz_path = npz_path
        data = np.load(npz_path,allow_pickle=True)
        print(data)

        for key in data.files:
            if key in ["profiles", "arrS", "S"]:
                continue
            print(f"Clé: ")
            print(f"{key} : {data[key]}")

npz_name = "olddatanpz/13-03_h1-985_m12.npz"
p = ThreeD_Data(npz_name)
