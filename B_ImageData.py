import Bb_tools_img as tools_img
import cv2
import numpy as np
import matplotlib.pyplot as plt
import A_utilit as utilit
import D_ThreeD as ThreeD

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 16  # Par exemple, 14 points
scales = {"2803": 0.33472530155340835, "0404": 0.3669702470186575}


def frame_to_profile(image):
    """
    Convertit une image donnée en un profil en traitant l'image pour détecter et analyser la ligne de profil.
    Remarques:
    - La fonction convertit l'image d'entrée en niveaux de gris.
    - Elle recadre l'image pour ignorer les régions noires et se concentrer sur la zone d'intérêt.
    - Elle calcule la ligne de base en ajustant une ligne aux données du profil.
    - La fonction renvoie l'image traitée ainsi que les données du profil et de la ligne de base.
    """
    image.Xline = tools_img.XLINE(image, method="weight center")
    image.Xbaseline = tools_img.XBASELINE(image)

    profile_between_y1_and_y2 = utilit.z(
        image.Xline[image.y1:image.y2], image.Xbaseline[image.y1:image.y2], image.theta)

    image.profile[image.y1:image.y2] = profile_between_y1_and_y2  # copy ?

    image.points_list = image.profile
    image.X_list = image.Xline
    image.Y_list = image.rangeY

    return


class ImageData:
    def __init__(self, frame: np.ndarray, **kwargs) -> None:
        '''
        Parameters:
        - mandatory
            - frame (RGB frame)
        - Recommended
            - rangeX
            - rangeY
            - res
        - Misc
            - theta
            - mass
            - height
            - hyperbolic_threshold
            - filterred
        '''
        self.vidpath = kwargs.get("vidpath", None)
        self.theta = kwargs.get("theta", 0)
        self.mass = kwargs.get("mass", None)
        self.height = kwargs.get("height", None)

        self.frame_nb = kwargs.get("frame_nb", 0)
        self.filterred = kwargs.get("filterred", False)
        self.date = kwargs.get("date", "0404")

        self.original_frame = frame
        self.frame = frame  # frame with eventual red filter
        self.gray_frame = []

        self.scale = scales[self.date]

        # Resolution of the image (width, height)
        self.res = kwargs.get("res", (np.shape(frame)[0:2])[::-1])
        # Horizontal and vertical length (orientation vers la droite et le BAS)
        self.X, self.Y = self.res

        # ! boundariesNappe :  limites de la nappe laser (seule région où on peut obtenir des données)
        self.y1 = 0
        self.y2 = 0
        #! boundariesBaseline :  limites de la région centrale du cratère (partie à ignorer pour la baseline)
        self.ya = 0
        self.yb = 0
        #! boundariesHyperbolic # Pour le fit on veut seulement la partie hyperbolique du profil
        self.i1 = 0
        self.i2 = 0
        # Pourcentage de la profondeur du profil.
        self.hyperbolic_threshold = kwargs.get("hyperbolic_threshold", 0.15)

        # X axis of the profile: set by the width of the image.
        self.rangeX = kwargs.get("rangeX", np.arange(self.X))
        # Y axis of the profile: set by the height of the image.
        self.rangeY = kwargs.get("rangeY", np.arange(self.Y))

        self.baselinea = 0
        self.baselineb = 0
        self.baseline_pcov = []

        self.Xline = []  # Là où est détecté la nappe

        self.popt = []  # Parameters of the hyperbolic fit
        self.pcov = []  # Covariance of the hyperbolic fit

    def __str__(self):
        return f"{self.X}x{self.Y} Image"

    def set_up(self) -> None:
        '''
        set up
        - gray_frame
        - profile init
        - boundaries
        Ready for profile calculation (but not for fit yet)
        '''
        if self.filterred:
            self.frame = tools_img.redfilter(self.frame)
            self.redfiltered_frame = self.frame
        # * gray_frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)

        # * Init profile with zeros
        self.profile = np.zeros(self.Y)

        # * Boundaries
        self.y1, self.y2, self.ya, self.yb = tools_img.boundaries(
            self.gray_frame)

        return

    def get_profile(self) -> None:
        frame_to_profile(self)
        return

    def run(self) -> None:
        '''
        Run the full process, once frame is defined
        '''
        self.set_up()
        self.get_profile()
        return

    def draw(self) -> None:
        '''
        Edit the image to draw the following:
        - Original image
        - Argmax
        - Baseline
        '''
        # Si on veut tracer sur l'image, on ne veut que des valeurs entières
        Xbaseline_int = np.astype(self.Xbaseline, np.int16)
        Xline_int = np.astype(self.Xline, np.int16)
        self.draw_frame = cv2.cvtColor(self.filtered_frame, cv2.COLOR_GRAY2RGB)
        # ! draw argmax
        self.draw_frame[self.rangeY, Xline_int, :] = [255, 0, 0]
        self.draw_frame[self.rangeY, Xline_int-1, :] = [255, 0, 0]
        self.draw_frame[self.rangeY, Xline_int+1, :] = [255, 0, 0]
        # ! draw baseline
        try:
            self.draw_frame[self.rangeY, Xbaseline_int, :] = [0, 0, 255]
            self.draw_frame[self.rangeY, Xbaseline_int-1, :] = [0, 0, 255]
            self.draw_frame[self.rangeY, Xbaseline_int+1, :] = [0, 0, 255]
        except IndexError:
            print(self.vidpath,
                  "Error: Index out of bounds while drawing baseline. Rotate ?")
        # ! draw areas where baseline fit is done
        self.draw_frame[self.y1, :, :] = [195, 100, 25]
        self.draw_frame[self.ya, :, :] = [195, 100, 25]
        self.draw_frame[self.yb, :, :] = [195, 100, 25]
        self.draw_frame[self.y2-1, :, :] = [195, 100, 25]

    def plot(self) -> None:
        '''
        Plot. Starts to draw, then proceed to plot 
        - Frame (edited)
        and on the other side:
        - Profile
        - Hyperbolic fit
        '''

        Xbaseline_int = np.astype(self.Xbaseline, np.int16)
        Xline_int = np.astype(self.Xline, np.int16)
        self.drawpartial1_frame = self.gray_frame.copy()
        self.drawpartial1_frame[self.ya:self.yb, :] = 0

        # self.drawpartial2_frame = self.gray_frame[self.y1:self.y2, :]

        self.drawline_frame = np.zeros(np.shape(self.frame))
        self.drawbsline_frame = cv2.cvtColor(
            self.drawpartial1_frame, cv2.COLOR_GRAY2RGB)

        # ! draw argmax
        self.drawline_frame[self.rangeY, Xline_int, :] = [0, 255, 0]
        # ! draw baseline
        try:
            self.drawbsline_frame[self.rangeY, Xbaseline_int, :] = [0, 0, 255]
        except IndexError:
            print(self.vidpath,
                  "Error: Index out of bounds while drawing baseline. Rotate ?")

        self.fig, self.ax = plt.subplots(2, 3)

        self.ax[0, 0].imshow(self.original_frame, origin="lower")
        self.ax[0, 0].set_title("Image originale")

        self.ax[0, 1].imshow(self.redfiltered_frame, origin="lower")

        self.ax[0, 2].imshow(self.gray_frame, cmap='gray', origin="lower")

        self.ax[1, 0].imshow(self.drawpartial1_frame,
                             cmap='gray', origin="lower")

        self.ax[1, 1].imshow(self.drawbsline_frame,
                             cmap='gray', origin="lower")

        self.ax[1, 2].imshow(self.drawline_frame, cmap='gray', origin="lower")

        dx_1mm = 10/self.scale
        for i in range(0, 2):
            for j in range(0, 3):
                self.ax[i, j].axhline(
                    10, 0.1, 0.1+1/dx_1mm, color="white")

                self.ax[i, j].annotate("\\small 1 cm", xy=(0.1+0.5/dx_1mm, 0.1), xycoords="axes fraction",
                                       ha="center", va="top", fontsize=20, color="white")

                self.ax[i, j].axhline(
                    10, 0.1, 0.1+1/dx_1mm, color="white")

        for i in range(self.ax.shape[0]):  # Parcours des lignes
            for j in range(self.ax.shape[1]):  # Parcours des colonnes
                self.ax[i, j].set_xlabel('$X$')
                self.ax[i, j].set_ylabel('$Y$', rotation=0, ha='right')
                self.ax[i, j].set_xticks([])
                self.ax[i, j].set_yticks([])
                self.ax[i, j].set_aspect('equal')

        self.fig3dshow = plt.figure()
        step = 12
        self.ax3dshow = self.fig3dshow.add_subplot(projection='3d')

        self.ax3dshow.scatter(
            self.X_list[::step], self.Y_list[::step], self.points_list[::step], marker="o")
        self.ax3dshow.set_xlabel('$X$')
        self.ax3dshow.set_ylabel('$Y$')
        self.ax3dshow.set_zlabel('$Z$')
        self.ax3dshow.axis('equal')

        self.ax3dshow.view_init(elev=0, azim=0)

        self.fig3dshow.canvas.mpl_connect(
            'key_press_event', ThreeD.return_update_func(self.fig3dshow, self.ax3dshow))


def get_frame_fromPath(img_path: str, rotate=False) -> np.ndarray:
    '''
    When user wants to set the frame from a path.
    Sets 
    - frame
    - rotate
    '''
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


if __name__ == "__main__":
    # Test the ImageData class
    # frame = get_frame_fromPath("data_test/test10_vert.png", rotate=False)
    # frame = get_frame_fromPath("data_test/test10_horiz.png", rotate=True)

    frame = get_frame_fromPath("testfilter.png", rotate=True)
    img_data = ImageData(frame, hyperbolic_threshold=0.25, filterred=True)
    img_data.run()
    img_data.draw()
    img_data.plot()
    plt.show()
