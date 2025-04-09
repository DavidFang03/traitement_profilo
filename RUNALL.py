import C_VideoData as VideoData
import D_ThreeD as TriD
import numpy as np
import json
import os
import D_ThreeD as ThreeD
# all_params = []
# with open("PARAMS.json", "r") as f:
#     all_params = json.load(f)

# all_npz_paths = []
# for params in all_params:
#     print("running", params["vidpath"])
#     npz_path = VideoData.RUN_VIDEODATA(params)
#     all_npz_paths.append(npz_path)


# np.savetxt("all_npz_paths.txt", all_npz_paths)

for npz_name in os.listdir("./final_datanpz"):
    npz_path = f"./final_datanpz/{npz_name}"
    print("running", npz_path)
    three_d_data = ThreeD.RUN_TRID(npz_path, npz_name[:-4])
