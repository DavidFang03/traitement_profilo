import Bb_tools_img as tools_img
import cv2
import numpy as np
import matplotlib.pyplot as plt
import A_utilit as utilit
import D_ThreeD as ThreeD


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
    image.new_S_list = image.Xline
    image.new_Y_list = image.rangeY

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
        self.theta = kwargs.get("theta", np.radians(30))
        self.mass = kwargs.get("mass", None)
        self.height = kwargs.get("height", None)

        self.frame_nb = kwargs.get("frame_nb", 0)
        self.filterred = kwargs.get("filterred", False)

        self.original_frame = frame
        self.frame = frame  # frame with eventual red filter
        self.gray_frame = []

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
        self.draw_frame[self.rangeY, Xline_int, :] = [0, 255, 0]
        # ! draw baseline
        try:
            self.draw_frame[self.rangeY, Xbaseline_int, :] = [0, 0, 255]
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

        self.fig, self.ax = plt.subplots(1, 3)

        self.ax[0].imshow(self.original_frame)
        self.ax[0].set_title("Original Frame")
        self.ax[0].axis('off')
        self.ax[0].set_aspect('equal')

        self.ax[1].imshow(self.frame)
        self.ax[1].set_title("Frame")
        self.ax[1].axis('off')
        self.ax[1].set_aspect('equal')

        step = 12
        self.ax[2].axis('off')
        self.ax[2] = self.fig.add_subplot(1, 3, 3, projection='3d')
        # print(np.shape(self.new_Y_list), np.shape(
        #     self.new_S_list), np.shape(self.points_list))
        self.ax[2].scatter(
            self.new_S_list[::step], self.new_Y_list[::step], self.points_list[::step], marker="o")
        self.ax[2].set_xlabel('S')
        self.ax[2].set_ylabel('Y')
        self.ax[2].set_zlabel('Z')
        self.ax[2].axis('equal')

        self.ax[2].view_init(elev=0, azim=0)

        self.fig.canvas.mpl_connect(
            'key_press_event', ThreeD.return_update_func(self.fig, self.ax[2]))


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
    # print(type(cv2.imread("test.png", cv2.IMREAD_COLOR)))
    # Test the ImageData class
    # frame = get_frame_fromPath("data_test/test10_vert.png", rotate=False)
    # frame = get_frame_fromPath("data_test/test10_horiz.png", rotate=True)
    frame = get_frame_fromPath("testfilter.png", rotate=True)
    img_data = ImageData(frame, hyperbolic_threshold=0.25, filterred=True)
    img_data.run()
    img_data.draw()
    img_data.plot()
    plt.show()
