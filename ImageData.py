import profilo_frame
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utilit


class ImageData:
    def __init__(self, frame: np.ndarray, **kwargs) -> None:
        '''
        Parameters:
        - mandatory
            - frame
        - Recommended
            - rangeX
            - rangeY
            - res
        - Misc
            - theta
            - mass
            - height
            - hyperbolic_threshold
        '''
        self.theta = kwargs.get("theta", np.radians(30))
        self.mass = kwargs.get("mass", None)
        self.height = kwargs.get("height", None)

        self.frame_nb = kwargs.get("frame_nb", 0)

        self.frame = frame
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
        self.profile = []
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

        # * gray_frame
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2GRAY)
        # * profile, rangeX, rangeY
        self.profile = np.zeros(self.Y)

        # * Boundaries
        self.y1, self.y2, self.ya, self.yb = profilo_frame.boundaries(
            self.gray_frame)

        return

    def get_profile(self) -> None:
        profilo_frame.frame_to_profile(self)
        return

    def get_fit(self) -> None:
        self.i1, self.i2 = profilo_frame.boundaries_Hyperbolic(
            self.profile, self.hyperbolic_threshold)
        profilo_frame.fit_hyperbolic(self)
        return

    def run(self) -> None:
        '''
        Run the full process, once frame is defined
        '''
        self.set_up()
        self.get_profile()
        self.get_fit()
        self.process_data()
        return

    def process_data(self) -> None:
        self.XMAX = np.max(self.Xline)
        self.YMAX = self.rangeY[np.argmax(self.Xline)]

    def draw(self) -> None:
        '''
        Edit the image to draw the following:
        - Original image
        - Argmax
        - Baseline
        '''
        # for i in range(self.Y):  # le premier indice (i) est la coordonnée verticale.
        #     if self.ya > i > self.y1 or self.yb < i < self.y2:
        #         # !draw areas where baseline is fit
        #         self.frame[i, self.Xline[i], :] = [255, 255, 0]
        # ! draw argmax        
        self.frame[self.rangeY, self.Xline, :] = [0, 255, 0]  
        # ! draw baseline
        try:
            self.frame[self.rangeY, self.Xbaseline_int, :] = [0, 0, 255] 
        except IndexError:
            print("Error: Index out of bounds while drawing baseline. Rotate ?")
        # ! draw areas where baseline fit is done
        self.frame[self.y1, : , :] = [195, 100, 25]
        self.frame[self.ya, : , :] = [195, 100, 25]
        self.frame[self.yb, : , :] = [195, 100, 25]
        self.frame[self.y2-1, : , :] = [195, 100, 25]
        



    def plot(self) -> None:
        '''
        Plot. Starts to draw, then proceed to plot 
        - Frame (edited)
        and on the other side:
        - Profile
        - Hyperbolic fit
        '''

        self.fig, self.ax = plt.subplots(1, 2)

        self.ax[0].imshow(self.frame)
        self.ax[0].set_title("Frame")
        self.ax[0].axis('off')
        self.ax[0].set_aspect('equal')

        self.plot_profile_on_ax(self.ax[1])
        self.ax[1].set_title("Profil")
        self.ax[1].set_xlabel("x")
        self.ax[1].set_ylabel("z")
        self.ax[1].legend()
        # self.ax[1].set_aspect('equal')

    def plot_profile_on_ax(self, ax) -> None:
        ax.plot(self.rangeY, self.profile, label="Profil")
        ax.plot(self.rangeY[self.i1:self.i2], utilit.hyperbolic(
            self.rangeY[self.i1:self.i2], *self.popt), label="Hyperbolic fit")

    def set_profile_on_lines(self, line, linefit) -> None:
        '''
        set the profile data on the given lines.
        '''
        line.set_ydata(self.profile)
        linefit.set_ydata(utilit.hyperbolic(
            self.rangeY[self.i1:self.i2], *self.popt))


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
    frame = get_frame_fromPath("data_test/test10_horiz.png", rotate=True)
    img_data = ImageData(frame, hyperbolic_threshold=0.25)
    img_data.run()
    img_data.draw()
    img_data.plot()
    plt.show()
