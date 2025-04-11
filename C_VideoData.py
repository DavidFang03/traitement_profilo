import matplotlib as mpl
import numpy as np
import cv2
import Cb_tools_vid as tools_vid
import B_ImageData as ImageData
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import A_utilit as utilit
import os
import json
import progress.bar

masses = np.array([0, 0, 89.46, 0, 63.7, 0, 49.59,
                   0, 32.63, 0, 23.81, 0, 16.69, 0, 14.8])
scales = {"2803": 0.33472530155340835, "0404": 0.3669702470186575}

# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20  # Par exemple,
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{xcolor}"
})


class VideoData:
    '''
    From a video path, generate and export a npz file with the profiles of the video.
    '''

    def __init__(self, params) -> None:
        '''
        :params
        - vidpath: path to the video (str)
        - theta : angle (degrees)
        - rotate : rotate the video (bool)
        - t1 : start time (s)
        - t2 : end time (s)
        - mass : mass number (int)
        - height : height (m)
        '''
        mandatory = {
            "vidpath": "",
            "theta_deg": 30,
            "theta": np.radians(30),
            "rotate": False,
            "t1": 0,
            "t2": None,
            "mass": None,
            "height": None,
            "info": "",
            "scale": 1,
        }

        for key, value in mandatory.items():
            setattr(self, key, params.get(key, value))

        # Ajout des autres attributs dynamiques
        for key, value in params.items():
            if key not in mandatory:
                setattr(self, key, value)

        self.params = params
        self.theta = np.radians(self.theta_deg)

        # ! Videos Infos
        self.frame_nb = 0

        self.cap = cv2.VideoCapture(self.vidpath)
        if not self.cap.isOpened():
            raise Exception("cap is not opened - check path")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.min_frame, self.max_frame = tools_vid.get_MinMax_frameNb(self)
        self.res = tools_vid.get_res(self)
        self.X, self.Y = self.res
        self.rangeX = np.arange(self.X)
        self.rangeY = np.arange(self.Y)
        # Middle index of the crater -> Profil perpendiculaire
        self.imid = int(self.Y / 2)

        self.nb_frames_anim = self.max_frame - self.min_frame - 1
        self.bar = progress.bar.Bar(
            f'Processing {self.vidpath}', max=self.nb_frames_anim)

        # ? Boundaries are specific to the frame

        # * Baseline
        self.baselineA = []
        self.baselineB = []
        self.baseline_a_avg = 0
        self.baseline_pcov = []

        # * Position de la nappe
        self.arr_z_imid = []  # Profondeur de la nappe au milieu de l'image
        # y = alpha * x + beta

        # * Profile
        self.points = []
        self.Y_arr = []
        self.X_arr = []

        # X axis of the profile: set by the width of the video.
        self.rangeS = []

        # * Debug
        self.firstFrame = []
        self.lastFrame = []
        self.filterred = False

    def set_up(self) -> None:
        '''
        Here we set up the video
        - We pass the frames that are not treated (before t1)
        '''

        # ! Passer les frames non traitées
        frame = None
        while self.frame_nb < self.min_frame+1:
            frame = self.next_frame()
        self.firstFrame = np.copy(frame)

        # ! check il red filter is needed later.
        firstFrame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.average(firstFrame_gray) > 110:
            self.filterred = True

        self.ymid = int(self.Y/2)

    def next_frame(self) -> np.ndarray:
        """
        Get the next frame of the video.
        - Increment self.frame_nb
        - Read the frame
        - Convert to RGB
        - Rotate the frame if needed
        If needed, rotate, or filter red.
        """
        self.frame_nb += 1
        ret, frame = self.cap.read()
        if not ret:
            raise Exception(
                f"Error reading frame {self.frame_nb}/{self.max_frame} - max_frame too big?")

        # ! rotate
        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def appendData_frame(self, img: ImageData.ImageData) -> None:
        '''
        Process the frame
        - Append the new points
        - Append the baseline to baselineA and baselineB
        - Append the first/last frame to firstFrame/lastFrame
        - Append position of the nappe to arrXMAX and arrYMAX
        '''

        self.baselineA.append(img.baselinea)
        self.baselineB.append(img.baselineb)
        self.baseline_a_avg = np.mean(self.baselineA)
        self.baseline_pcov.append(img.baseline_pcov)

        self.points = np.concatenate((self.points, img.points_list))
        self.Y_arr = np.concatenate((self.Y_arr, img.Y_list))
        self.X_arr = np.concatenate((self.X_arr, img.X_list))

        mask_ymid = (self.Y_arr == self.ymid)
        self.arr_z_imid = self.points[mask_ymid]
        self.arr_x_imid = self.X_arr[mask_ymid]

    def from_frame_to_Image(self, frame: np.ndarray) -> ImageData.ImageData:
        return ImageData.ImageData(frame, rangeX=self.rangeX, rangeY=self.rangeY,
                                   res=self.res, theta=self.theta, mass=self.mass, height=self.height, frame_nb=self.frame_nb, filterred=self.filterred, vidpath=self.vidpath)

    def iterate(self) -> ImageData.ImageData:
        frame = self.next_frame()

        image = self.from_frame_to_Image(frame)

        image.run()
        self.appendData_frame(image)

        return image

    def iterate_anim(self, i: int) -> tuple:
        image = self.iterate()

        # ! image originale
        self.framed_showed.set_array(image.original_frame)
        image.draw()
        # ! Image avec tracés (après éventuels red filter)
        self.framed_drawed.set_array(image.draw_frame)

        # ! Profil perpendiculaire au milieu
        if np.max(self.arr_z_imid) > np.abs(np.min(self.arr_z_imid)):
            coeff = -1
        else:
            coeff = 1

        self.perp_profile_showed.set_data(
            self.arr_x_imid*self.scale, coeff * self.arr_z_imid*self.scale)

        # ! infos
        self.infostext.set_text(
            f"Frame {self.frame_nb}/{self.max_frame} \n $\\theta = {self.theta_deg}$° \n $H = {self.height}$ m \n $m={masses[self.mass]}$ g")

        if self.frame_nb >= self.max_frame:
            print("Funcanimation probably finished")

        return self.animated_lines

    def init_anim(self) -> None:
        """
        Initialise les plots pour l'animation.
        - self.framed_showed : image originale
        - self.framed_drawed : image avec tracés
        - self.profile_showed : profil
        - self.perp_profile_showed : profil perpendiculaire
        - self.infostext : infos : frame number, theta, filename, height, mass

        """
        self.anifig, self.aniaxes = plt.subplots(2, 3, figsize=(18, 9))
        self.framed_showed = self.aniaxes[0, 0].imshow(
            self.firstFrame, animated=True, origin='lower')
        self.framed_drawed = self.aniaxes[0, 1].imshow(
            self.firstFrame, animated=True, origin='lower')

        dx_1mm = 10/self.scale
        self.linescale2 = self.aniaxes[0, 0].axhline(
            10, 0.1, 0.1+1/dx_1mm, color="white")

        self.textscale2 = self.aniaxes[0, 0].annotate("\\small 1 cm", xy=(0.1+0.5/dx_1mm, 0.1), xycoords="axes fraction",
                                                      ha="center", va="top", fontsize=20, color="white")

        self.linescale = self.aniaxes[0, 1].axhline(
            10, 0.1, 0.1+1/dx_1mm, color="white")

        self.textscale = self.aniaxes[0, 1].annotate("\\small 1 cm", xy=(0.1+0.5/dx_1mm, 0.1), xycoords="axes fraction",
                                                     ha="center", va="top", fontsize=20, color="white")

        self.perp_profile_showed, = self.aniaxes[0, 2].plot(
            self.rangeX, np.zeros(self.X), "o", label="Profil perpendiculaire", markersize=3)

        self.aniaxes[0, 1].annotate(
            "$y_0$",
            xy=(0, self.ymid),
            xycoords='data',
            # Ajustez cette valeur pour déplacer le texte vers la gauche
            xytext=(-20, 0),
            textcoords='offset points',
            ha='right',
            va='center',
            fontsize=14,
            color='magenta',
            arrowprops=dict(arrowstyle="->", color='magenta')
        )
        self.aniaxes[0, 1].annotate(
            "$y_0$",
            xy=(self.X, self.ymid),
            xycoords='data',
            # Ajustez cette valeur pour déplacer le texte vers la gauche
            xytext=(20, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            fontsize=14,
            color='magenta',
            arrowprops=dict(arrowstyle="->", color='magenta')
        )

        self.infostext = self.aniaxes[1, 1].text(0.5, 0.5, "Frame 0", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                                                 transform=self.aniaxes[1, 1].transAxes, ha="center")
        self.aniaxes[1, 1].axis('off')
        self.aniaxes[1, 0].axis('off')
        self.aniaxes[1, 2].axis('off')

        self.aniaxes[0, 0].set_xticks([])
        self.aniaxes[0, 0].set_yticks([])
        self.aniaxes[0, 1].set_xticks([])
        self.aniaxes[0, 1].set_yticks([])

        self.aniaxes[0, 0].set_xlabel("$X$")
        self.aniaxes[0, 0].set_ylabel("$Y$", rotation=0, loc="top")
        self.aniaxes[0, 1].set_xlabel("$X$")
        self.aniaxes[0, 1].set_ylabel("$Y$", rotation=0, loc="top")

        self.aniaxes[0, 2].set_title("Profil perpendiculaire")
        self.aniaxes[0, 2].set_ylim(-100*self.scale, 10*self.scale)
        self.aniaxes[0, 2].set_xlim(0, self.X*self.scale)
        self.aniaxes[0, 2].set_xlabel('$X$ (mm)')
        self.aniaxes[0, 2].yaxis.set_label_position("right")
        self.aniaxes[0, 2].yaxis.set_ticks_position('right')

        # self.aniaxes[0, 2].set_ylabel(
        #     r"$Z(X,{\color{red}y_0})$\\(mm)", rotation=0, loc="top", labelpad=30)

        self.aniaxes[0, 2].text(1.05, 1.1,
                                r"$Z(X,$",
                                transform=self.aniaxes[0, 2].transAxes,
                                ha="left", va="top",
                                color="black")

        self.aniaxes[0, 2].text(1.2, 1.1,
                                r"$y_0$",
                                transform=self.aniaxes[0, 2].transAxes,
                                ha="left", va="top",
                                color="magenta")

        self.aniaxes[0, 2].text(1.26, 1.1,
                                r"$)$(mm)",
                                transform=self.aniaxes[0, 2].transAxes,
                                ha="left", va="top",
                                color="black")

        self.animated_lines = [self.framed_showed, self.framed_drawed,
                               self.perp_profile_showed, self.infostext, self.linescale, self.textscale, self.linescale2, self.textscale2]

    def dumb_init(self):
        return self.animated_lines

    def animate(self) -> None:
        self.init_anim()
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        # plt.show()
        # plt.close()
        if self.savemode:
            self.funcanim = animation.FuncAnimation(
                self.anifig, self.iterate_anim, init_func=self.dumb_init, frames=self.nb_frames_anim-10, blit=True, repeat=False, interval=1)
            # self.anifig.tight_layout()
            def progress_callback(i, n): self.bar.next()
            self.funcanim.save(
                f"funcanim/funcanim_{self.date}_{self.vidname}.mp4", fps=3*self.fps, extra_args=['-vcodec', 'libx264'], progress_callback=progress_callback)
        else:
            self.funcanim = animation.FuncAnimation(
                self.anifig, self.iterate_anim, init_func=self.dumb_init, frames=self.nb_frames_anim, interval=1000 / self.fps, blit=True, repeat=False)
            # self.anifig.tight_layout()

    def export_npz(self) -> str:
        """
        Effecue process (retournement)
        et exporte dans self.npz_name
        """
        self.process_before_export()
        self.npz_name = tools_vid.generate_file_name(self)
        npz_path = f"./final_datanpz/{self.npz_name}"
        np.savez(npz_path, vidpath=self.vidpath, vidname=self.vidname, points=self.points,
                 arrX=self.X_arr, arrY=self.Y_arr, Y=self.Y, height=self.height, mass=self.mass, theta_deg=self.theta_deg, t1=self.t1, t2=self.t2, info=self.info, timestamp=self.timestamp, date=self.date)
        return npz_path

    def process_before_export(self) -> None:
        """
        Retourne si mauvais signe
        """
        if np.max(self.points) > np.abs(np.min(self.points)):
            self.points = -np.array(self.points)
        return

    def add_to_history(self):
        self.params["npz_name"] = self.npz_name
        utilit.add_to_history(self.params, "history_vid.json")

    def __str__(self):
        return f"{self.X}x{self.Y} Video"


def get_videos_firstinfos(vidpath: str) -> float:
    cap = cv2.VideoCapture(vidpath)
    if not cap.isOpened():
        raise Exception("cap is not opened - check path")

    fps = cap.get(cv2.CAP_PROP_FPS)
    nbframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = nbframes / fps
    return duration


def init_params() -> dict:
    """
    Initialize the parameters for the video processing : vidpath, mass, height, angle, t1, t2, rotate, info
    Either uses the last parameters used or asks the user for new parameters.
    """
    with open('lastparams.json', 'r') as openfile:
        json_object = json.load(openfile)

    print(f"lastparams are {json_object}")
    same = input("> Run with the same params ? (y/n)")
    if same == "n":
        params = ask_user_for_params(lastparams=json_object)
    else:
        params = json_object

    vidpath = params["vidpath"]
    if not os.path.isfile(vidpath):
        raise Exception(f"{vidpath} not found")

    json_object = json.dumps(params, indent=4)
    with open("lastparams.json", "w") as outfile:
        outfile.write(json_object)
    print(f"ok, opening {json_object}")

    return params


def ask_user_for_params(lastparams: dict) -> dict:
    """
    Ask the user for the parameters of the video processing : vidpath, mass, height, angle, t1, t2, rotate, info
    Suggests to use the last params.
    """
    # ! date
    date = input(f"DATE ? ({lastparams["date"]})") or lastparams["date"]
    folder = f"./vids/{date}"

    # ! vidname
    vidname = input(
        f'NOM DU FICHIER ? (seulement le nom) - ({lastparams["vidpath"]})') or lastparams["vidname"]
    vidpath = f"{folder}/{vidname}.mp4"
    if not os.path.isfile(vidpath):
        raise Exception(f"{vidpath} not found")
    else:
        print(f"ok, opening {vidpath}")

    duration = get_videos_firstinfos(vidpath)

    mass = int(input(f'MASSE ? ({lastparams["mass"]})') or lastparams["mass"])
    height = float(
        input(f'HAUTEUR ? ({lastparams["height"]})') or lastparams["height"])
    theta_deg = float(
        input(f'THETA (deg) ? ({lastparams["theta_deg"]})') or lastparams["theta_deg"])
    t1 = float(
        input(f'T1 ? (max:{duration:.2f} s) ({lastparams["t1"]})') or lastparams["t1"])
    t2 = float(
        input(f'T2 ? (max:{duration:.2f} s) ({lastparams["t2"]})') or lastparams["t2"])
    rotate = input(
        f'ROTATE ? ({lastparams["rotate"]})') or lastparams["rotate"]
    rotate = rotate == "y" or rotate == "True" or rotate == "true" or rotate == True
    info = input("INFO ? vide sinon")

    params = {"date": date, "vidname": vidname, "vidpath": vidpath, "rotate": rotate, "mass": mass,
              "height": height, "theta_deg": theta_deg,
              "t1": t1, "t2": t2, "info": info, }
    return params


def go(params: dict) -> None:
    # ! Historique
    params["timestamp"] = utilit.get_timestamp()

    p = VideoData(params)

    p.set_up()

    p.animate()

    print(p)

    if "savemode" in params and not params["savemode"]:
        print("heyyy")
        plt.show()

    # p.export_npz()
    # print("Exported to npz")
    # p.add_to_history()


def RUN_VIDEODATA(params: dict) -> None:
    # ! Historique
    params["timestamp"] = utilit.get_timestamp()
    params["savemode"] = True
    p = VideoData(params)

    p.set_up()
    p.animate()

    # print(p)

    npz_path = p.export_npz()
    print(f"Exported to {npz_path}")
    # p.add_to_history()


def main(scale, savemode=False) -> None:

    # params = init_params()
    params = {
        "date": "2803",
        "vidname": "m2-riv_centre",
        "vidpath": "./vids/2803/m2-riv_centre.mp4",
        "rotate": True,
        "mass": 2,
        "height": 5.907,
        "theta_deg": 42.85,
        "t1": 0.0,
        "t2": 0,
        "info": ""}

    params["savemode"] = savemode
    if scale:
        params["scale"] = scales[params["date"]]
    go(params)

    again = False
    while again:
        useragain = input("Run again ? (y/n)")
        if useragain == "y":
            print("OK, running again")
            go(params)
            again = True
        else:
            print("Bye")
            again = False


if __name__ == "__main__":
    # main(scale=True, savemode=True)
    params = {
        "date": "2803",
        "vidname": "m2",
        "vidpath": "./vids/2803/m2.mp4",
        "rotate": True,
        "mass": 2,
        "height": 2.008,
        "theta_deg": 42.75,
        "t1": 0.0,
        "t2": 0,
        "info": ""
    }
    RUN_VIDEODATA(params)
