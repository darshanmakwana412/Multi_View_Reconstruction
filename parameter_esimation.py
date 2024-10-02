from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import glob
import os
import cv2

class ObjView:
    def __init__(self, image_path: str) -> None:
        self.root_dir = "/".join(image_path.split("/")[:-1])
        self.img_name = int(image_path.split("/")[-1][:-4])
        self.img_array = cv2.imread(image_path)

    def detectFeatures(self, detector: cv2.SIFT) -> None:
        self.kp, self.desc = detector.detectAndCompute(self.img_array, None)

    @staticmethod
    def load_from(images_dir: str) -> list[ObjView]:
        views = []
        detector = cv2.SIFT_create()
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")), key= lambda s: int(s.split("/")[-1][:-4]))
        for image_path in tqdm(image_paths, desc="Initializing Object Views from Images"):
            view = ObjView(image_path)
            view.detectFeatures(detector)
            views.append(view)
        return views

    def plot_keypoints(self) -> None:
        plot = cv2.drawKeypoints(self.img_array, self.kp, self.img_array, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(plot)
        plt.show()