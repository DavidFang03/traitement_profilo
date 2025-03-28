import numpy as np
import cv2
import tools_vid
import ImageData
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import utilit
import os
import json

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
            "scale": None,
            "info": "",
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

        # ? Boundaries are specific to the frame

        # * Baseline
        self.baselineA = []
        self.baselineB = []
        self.baseline_a_avg = 0
        self.baseline_pcov = []

        # * Position de la nappe
        self.arrXMAX = []
        self.arrYMAX = []
        self.arr_z_imid = []  # Profondeur de la nappe au milieu de l'image
        # y = alpha * x + beta
        self.arrS = []  # Position de la nappe

        # * Profile
        self.listOf_profiles = []  # List of 2D profiles (3D)
        # X axis of the profile: set by the width of the video.
        self.rangeS = []

        # * Debug
        self.firstFrame = []
        self.lastFrame = []


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

    def next_frame(self) -> np.ndarray:
        """
        Get the next frame of the video.
        - Increment self.frame_nb
        - Read the frame
        - Convert to RGB
        - Rotate the frame if needed
        """
        self.frame_nb += 1
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error reading frame - max_frame too big?")
        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def appendData_frame(self, img: ImageData.ImageData) -> None:
        '''
        Process the frame
        - Append the profile to listOf_profiles
        - Append the baseline to baselineA and baselineB
        - Append the first/last frame to firstFrame/lastFrame
        - Append position of the nappe to arrXMAX and arrYMAX
        '''
        # ! Append the profile to listOf_profiles
        self.listOf_profiles.append(img.profile)
        self.baselineA.append(img.baselinea)
        self.baselineB.append(img.baselineb)
        self.baseline_a_avg = np.mean(self.baselineA)
        self.baseline_pcov.append(img.baseline_pcov)
        self.arrXMAX.append(img.XMAX)
        self.arrYMAX.append(img.YMAX)
        self.arr_z_imid.append(img.profile[self.imid])
        # self.arrS.append(np.cos(np.arctan(self.baseline_a_avg))*img.XMAX)
        nparrXMAX = np.array(self.arrXMAX)
        self.arrS = np.cos(np.arctan(self.baseline_a_avg)) * nparrXMAX

    def from_frame_to_Image(self, frame: np.ndarray) -> ImageData.ImageData:
        return ImageData.ImageData(frame, rangeX=self.rangeX, rangeY=self.rangeY,
                                   res=self.res, theta=self.theta, mass=self.mass, height=self.height, frame_nb=self.frame_nb)

    def iterate(self) -> ImageData.ImageData:
        frame = self.next_frame()

        image = self.from_frame_to_Image(frame)

        image.run()
        self.appendData_frame(image)
        return image

    def iterate_anim(self, i: int) -> tuple:
        image = self.iterate()

        self.framed_showed.set_array(image.frame)
        image.draw()
        self.framed_drawed.set_array(image.frame)

        self.profile_showed.set_data(self.rangeY, image.profile)
        self.profile_fit_showed.set_data(self.rangeY[image.i1:image.i2], utilit.hyperbolic(
            image.rangeY[image.i1:image.i2], *image.popt))

        if np.max(self.arr_z_imid) > np.abs(np.min(self.arr_z_imid)):
            self.perp_profile_showed.set_data(
                self.arrS, -np.array(self.arr_z_imid))
        else:
            self.perp_profile_showed.set_data(self.arrS, self.arr_z_imid)

        self.infostext.set_text(f"Frame {self.frame_nb}/{self.max_frame} \n Theta: {self.theta_deg}° \n Filename: {self.vidpath} \n Height: {self.height} m \n Mass: {self.mass}")

        if self.frame_nb>=self.max_frame:
            print("Funcanimation probably finished")

        return self.framed_showed, self.framed_drawed, self.profile_showed, self.profile_fit_showed, self.perp_profile_showed, self.infostext


    def init_anim(self) -> None:
        """
        Initialise les plots pour l'animation.
        - self.framed_showed : image originale
        - self.framed_drawed : image avec tracés
        - self.profile_showed : profil
        - self.profile_fit_showed : fit hyperbolique du profil
        - self.perp_profile_showed : profil perpendiculaire
        - self.infostext : infos : frame number, theta, filename, height, mass     

        """
        self.anifig, self.aniaxes = plt.subplots(2, 3)
        self.framed_showed = self.aniaxes[0, 0].imshow(
            self.firstFrame, animated=True)
        self.framed_drawed = self.aniaxes[0, 1].imshow(
            self.firstFrame, animated=True)

        self.profile_showed, = self.aniaxes[0, 2].plot(
            self.rangeY, 10+np.zeros(self.Y), label="Profil")
        self.profile_fit_showed, = self.aniaxes[0, 2].plot(self.rangeY, -100+np.zeros(self.Y),
                                                           label="Hyperbolic fit")

        self.perp_profile_showed, = self.aniaxes[1, 2].plot(
            self.rangeX, np.zeros(self.X), label="Profil perpendiculaire")
        
        self.infostext = self.aniaxes[1,1].text(0.5, 0.5, "Frame 0", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=self.aniaxes[1,1].transAxes, ha="center")

        self.aniaxes[1, 2].set_title("Profil perpendiculaire")
        self.aniaxes[1, 2].set_ylim(-100, 10)

    def dumb_init(self):
        return self.framed_showed, self.framed_drawed, self.profile_showed, self.profile_fit_showed, self.perp_profile_showed, self.infostext

    def animate(self) -> None:
        # init
        self.init_anim()

        self.nb_frames_anim = self.max_frame - self.min_frame - 1
        # self.iterate_anim(0) # est en fait automatiquement appelé par FuncAnimation lorsque aucune init_func n'est donnée
        self.funcanim = animation.FuncAnimation(
            self.anifig, self.iterate_anim, init_func=self.dumb_init, frames=self.nb_frames_anim, interval=1000 / self.fps, blit=True, repeat=False)
        self.anifig.tight_layout()
        # self.animate.save("lastanimation.mp4", fps=self.fps,
        #                   extra_args=['-vcodec', 'libx264'])


    def export_npz(self) -> None:
        """
        Effecue process (retournement)
        et exporte dans self.npz_name
        """
        self.process_before_export()
        self.npz_name = tools_vid.generate_file_name(self)
        np.savez(f"./datanpz/{self.npz_name}", vidpath=self.vidpath, profiles=np.array(self.listOf_profiles),
                 arrS=self.arrS, Y=self.Y, height=self.height, mass=self.mass, scale=self.scale, theta_deg=self.theta_deg, t1=self.t1, t2=self.t2, info=self.info, timestamp=self.timestamp, date=self.date)

    def process_before_export(self) -> None:
        """
        Retourne si mauvais signe
        """
        if np.max(self.listOf_profiles) > np.abs(np.min(self.listOf_profiles)):
            self.listOf_profiles = -np.array(self.listOf_profiles)
        return
        self.baseline_a_avg = np.mean(self.baselineA)
        # self.baselineB = np.array(self.baselineB)

    def add_to_history(self):
        self.params["npz_name"] = self.npz_name
        utilit.add_to_history(self.params,"history_vid.json")

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


    # ! Historique
    params["timestamp"] = utilit.get_timestamp()
    return params


def ask_user_for_params(lastparams:dict) -> dict:
    """
    Ask the user for the parameters of the video processing : vidpath, mass, height, angle, t1, t2, rotate, info
    Suggests to use the last params.
    """
    # ! date
    date = input(f"DATE ? ({lastparams["date"]})") or lastparams["date"]
    folder = f"./vids/{date}"

    # ! vidname
    vidname = input(f'NOM DU FICHIER ? (seulement le nom) - ({lastparams["vidpath"]})') or lastparams["vidname"]
    vidpath = f"{folder}/{vidname}.mp4"
    if not os.path.isfile(vidpath):
        raise Exception(f"{vidpath} not found")
    else:
        print(f"ok, opening {vidpath}")
    

    duration = get_videos_firstinfos(vidpath)

    mass = int(input(f'MASSE ? ({lastparams["mass"]})') or lastparams["mass"])
    height = float(input(f'HAUTEUR ? ({lastparams["height"]})') or lastparams["height"])
    theta_deg = float(input(f'THETA (deg) ? ({lastparams["theta_deg"]})') or lastparams["theta_deg"])
    t1 = float(input(f'T1 ? (max:{duration:.2f} s) ({lastparams["t1"]})') or lastparams["t1"])
    t2 = float(input(f'T2 ? (max:{duration:.2f} s) ({lastparams["t2"]})') or lastparams["t2"])
    rotate = input(f'ROTATE ? ({lastparams["rotate"]})') or lastparams["rotate"]
    rotate = rotate == "y" or rotate == "True" or rotate == "true"
    info = input("INFO ? vide sinon")

    params = {"date":date, "vidname":vidname, "vidpath": vidpath, "rotate": rotate, "mass": mass,
              "height": height, "theta_deg": theta_deg,
              "t1": t1, "t2": t2, "info": info,}
    return params


def go(params : dict) -> None:
    p = VideoData(params)

    p.set_up()
    p.animate()

    print(p)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()

    p.export_npz()
    print("Exported to npz")
    p.add_to_history()


def main() -> None:
    params = init_params()
    go(params)

    again = True
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
    main()
