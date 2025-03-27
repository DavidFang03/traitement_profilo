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

    def __init__(self, **kwargs) -> None:
        '''
        :params
        - vid_path: path to the video (str)
        - theta : angle (degrees)
        - rotate : rotate the video (bool)
        - t1 : start time (s)
        - t2 : end time (s)
        - mass : mass number (int)
        - height : height (m)
        '''
        defaults = {
            "vid_path": "",
            "theta": 30,
            "rotate": False,
            "t1": 0,
            "t2": None,
            "mass": None,
            "height": None,
            "scale": None,
            "info": "",
        }

        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))

        # Ajout des autres attributs dynamiques
        for key, value in kwargs.items():
            if key not in defaults:
                setattr(self, key, value)

        # ! Videos Infos
        self.frame_nb = 0

        self.cap = cv2.VideoCapture(self.vid_path)
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
        self.rangeX = []
        self.rangeY = []
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
            self.frame_nb += 1
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Error reading frame - min_frame too big?")
        self.firstFrame = np.copy(frame)

    def run_film(self) -> None:
        '''
        The video is read frame by frame here.
        Are appended
        - Each profile > listOf_profiles
        - Each baseline > baselineA and baselineB
        - first/last frame > firstFrame/lastFrame
        '''
        while self.cap.isOpened() and self.frame_nb < self.max_frame:
            self.iterate()

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
                                   res=self.res, theta=self.theta, mass=self.mass, height=self.height)

    def iterate(self, i: int) -> ImageData.ImageData:
        self.frame_nb += 1
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Error reading frame - max_frame too big?")

        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image = self.from_frame_to_Image(frame)

        image.run()
        self.appendData_frame(image)
        return image

    def iterate_anim(self, i: int) -> tuple[plt.AxesImage, plt.AxesImage, plt.Line2D, plt.Line2D, plt.Line2D]:
        image = self.iterate(i)
        self.framed_showed.set_array(image.frame)
        image.draw()
        self.framed_drawed.set_array(image.frame)

        # image.set_profile_on_lines(
        #     self.profile_showed, self.profile_fit_showed)

        self.profile_showed.set_data(self.rangeY, image.profile)
        self.profile_fit_showed.set_data(self.rangeY[image.i1:image.i2], utilit.hyperbolic(
            image.rangeY[image.i1:image.i2], *image.popt))

        if np.max(self.arr_z_imid) > np.abs(np.min(self.arr_z_imid)):
            self.perp_profile_showed.set_data(
                self.arrS, -np.array(self.arr_z_imid))
        else:
            self.perp_profile_showed.set_data(self.arrS, self.arr_z_imid)

        self.aniaxes[1, 1].set_title(
            f"Frame {self.frame_nb/self.nb_frames_anim}")

        return self.framed_showed, self.framed_drawed, self.profile_showed, self.profile_fit_showed, self.perp_profile_showed

    def animate(self) -> None:
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

        self.aniaxes[1, 2].set_title("Profil perpendiculaire")
        self.aniaxes[1, 2].set_ylim(-100, 10)
        # init for blitting

        self.nb_frames_anim = self.max_frame - self.min_frame

        self.iterate_anim(0)
        self.animate = animation.FuncAnimation(
            self.anifig, self.iterate_anim, frames=self.nb_frames_anim, interval=1000 / self.fps, blit=True, repeat=False)
        self.anifig.tight_layout()
        # self.animate.save("lastanimation.mp4", fps=self.fps,
        #                   extra_args=['-vcodec', 'libx264'])

    # def init_funcanim(self):
    #     self.profile_showed.set_data([],[])
    #     self.profile_fit_showed.set_data([],[])
    #     return self.framed_showed, self.framed_drawed, self.profile_showed, self.profile_fit_showed

    def export_npz(self) -> None:
        self.process_before_export()
        npz_name = tools_vid.generate_file_name(self)
        np.savez(f"./datanpz/{npz_name}", profiles=np.array(self.listOf_profiles),
                 arrS=self.arrS, Y=self.Y, height=self.height, scale=self.scale, theta=self.theta, t1=self.t1, t2=self.t2)

    def process_before_export(self) -> None:
        if np.max(self.listOf_profiles) > np.abs(np.min(self.listOf_profiles)):
            self.listOf_profiles = -np.array(self.listOf_profiles)
        return
        self.baseline_a_avg = np.mean(self.baselineA)
        # self.baselineB = np.array(self.baselineB)

    def __str__(self):
        return f"{self.X}x{self.Y} Video"


def get_videos_firstinfos(vid_path: str) -> float:
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise Exception("cap is not opened - check path")

    fps = cap.get(cv2.CAP_PROP_FPS)
    nbframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = nbframes / fps
    return duration


def main() -> None:
    params = init_params()
    filepath = params["filepath"]
    mass = params["mass"]
    height = params["height"]
    theta = params["theta"]
    t1 = params["t1"]
    t2 = params["t2"]
    rotate = params["rotate"]
    info = params["info"]

    go(filepath, t1, t2,
       rotate, mass, height, theta, info)

    again = True
    while again:
        print("run again ? (y/n)")
        useragain = input()
        if useragain == "y":
            print("ok, running again")
            go(filepath, t1, t2,
               rotate, mass, height, theta, info)
            again = True
        else:
            print("bye")
            again = False


if __name__ == "__main__":
    main()


def init_params() -> dict:
    with open('lastparams.json', 'r') as openfile:
        json_object = json.load(openfile)

    print(f"lastparams are {json_object} run with the same ? (y/n)")
    same = input()
    if same == "n":
        params = ask_user_for_params()
    else:
        params = json_object

    filepath = params["filepath"]
    if not os.path.isfile(filepath):
        raise Exception(f"{filepath} not found")

    json_object = json.dumps(params, indent=4)
    with open("lastparams.json", "w") as outfile:
        outfile.write(json_object)
    print(f"ok, opening {json_object}")
    return params


def ask_user_for_params() -> dict:
    folder = "./vids/1403"
    print('filename')
    filename = input()
    filepath = f"{folder}/{filename}.mp4"
    if not os.path.isfile(filepath):
        raise Exception(f"{filepath} not found")
    duration = get_videos_firstinfos(filepath)
    print('Numéro de masse')
    mass = int(input())
    print("Hauteur (m)")
    height = float(input())
    print("Angle (°)")
    theta = np.radians(float(input()))
    print("t1 (s)")
    t1 = float(input())
    print(f"t2 (max:{duration:.2f}) (s) ?")
    t2 = float(input())
    print("rotate ? (y/n)")
    rotate = input()
    if rotate == "y":
        rotate = True
    else:
        rotate = False
    print("info ? vide sinon")
    info = input()

    params = {"filepath": filepath, "rotate": rotate, "mass": mass,
              "height": height, "theta": theta, "info": info,
              "t1": t1, "t2": t2}
    return params


def go(filepath, t1, t2,
       rotate, mass, height, theta, info):
    p = VideoData(filepath, t1=t1, t2=t2,
                  rotate=rotate, mass=mass, height=height, theta=theta, info=info)

    p.set_up()
    p.animate()

    print(p)

    plt.show()
    p.export_npz()
