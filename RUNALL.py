import concurrent.futures
import C_VideoData as VideoData
import D_ThreeD as TriD
import numpy as np
import json
import matplotlib

matplotlib.use('Agg')


def process_video(params):
    print("running", params["vidpath"])
    npz_path = VideoData.RUN_VIDEODATA(params)
    return npz_path


all_params = []
with open("ALLPARAMS.json", "r") as f:
    all_params = json.load(f)

all_npz_paths = []

# Utiliser ThreadPoolExecutor pour le parallélisme
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Soumettre toutes les tâches à l'executor
    future_to_npz = {executor.submit(
        process_video, params): params for params in all_params}

    # Récupérer les résultats au fur et à mesure qu'ils sont disponibles
    for future in concurrent.futures.as_completed(future_to_npz):
        params = future_to_npz[future]
        try:
            npz_path = future.result()
            all_npz_paths.append(npz_path)
        except Exception as exc:
            print(f"{params['vidpath']} generated an exception: {exc}")

np.savetxt("all_npz_paths.txt", all_npz_paths)
