from multiprocessing import Process
from D_ThreeD import ThreeD_Data
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

npz_path = "final_datanpz/m2-riv_centre_2803_h5-907_m2.npz"
three_d_data = ThreeD_Data(npz_path)
three_d_data.plot_profile3D()
three_d_data.fit3D()
three_d_data.plot_fit3D()
three_d_data.plot_both()
three_d_data.plot_plan()

three_d_data.end_plot()


def run_anim(name, nb_azim_angles, pbars):

    three_d_data.anim_profile(name, nb_azim_angles, pbars)


if __name__ == '__main__':
    nb_azim_angles = 360
    pbars = {"profile": None, "fit": None, "both": None}
    processes = []
    for name in ["profile", "fit", "both"]:
        pbars[name] = tqdm(
            total=nb_azim_angles, desc=f'Processing {three_d_data.vidpath} : {name}')
        p = Process(target=run_anim, args=(name, nb_azim_angles, pbars))
        processes.append(p)
    # p1 = Process(target=run_anim, args=("profile",))
    # p2 = Process(target=run_anim, args=("profile", nb_azim_angles, pbars))
    # p2 = Process(target=run_anim, args=("fit", nb_azim_angles, pbars))
    # p3 = Process(target=run_anim, args=("both", nb_azim_angles, pbars))

    for process in processes:
        process.start()

    for process in processes:
        process.join()
