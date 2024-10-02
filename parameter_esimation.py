from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import glob
import os
import cv2
import numpy as np

class ObjView:
    def __init__(self, image_path: str) -> None:
        self.root_dir = "/".join(image_path.split("/")[:-1])
        self.img_name = int(image_path.split("/")[-1][:-4])
        self.img_array = cv2.imread(image_path)
        self.kp = None
        self.desc = None

    def detectFeatures(self, detector: cv2.SIFT) -> None:
        keypoints_dir = os.path.join(self.root_dir, "keypoints")
        desc_dir = os.path.join(self.root_dir, "descriptors")
        os.makedirs(keypoints_dir, exist_ok=True)
        os.makedirs(desc_dir, exist_ok=True)

        keypoints_path = os.path.join(keypoints_dir, f"{self.img_name}_keypoints.npy")
        desc_path = os.path.join(desc_dir, f"{self.img_name}_descriptors.npy")

        if os.path.exists(keypoints_path) and os.path.exists(desc_path):
            self.kp, self.desc = self.load(keypoints_path, desc_path)
        else:
            self.kp, self.desc = detector.detectAndCompute(self.img_array, None)
            self.save(keypoints_path, desc_path)

    def save(self, keypoints_path: str, desc_path: str) -> None:
        kp_array = np.array([(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id) for k in self.kp], dtype=np.float32)
        desc_array = np.array(self.desc, dtype=np.float32)
        np.save(keypoints_path, kp_array)
        np.save(desc_path, desc_array)

    def load(self, keypoints_path: str, desc_path: str):
        kp_array = np.load(keypoints_path, mmap_mode='r')
        desc_array = np.load(desc_path, mmap_mode='r')
        keypoints = [cv2.KeyPoint(x=f[0], y=f[1], _size=f[2], _angle=f[3], _response=f[4], _octave=f[5], _class_id=int(f[6])) for f in kp_array]
        return keypoints, desc_array

    @staticmethod
    def load_from(images_dir: str) -> list['ObjView']:
        views = []
        detector = cv2.SIFT_create()
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")), key=lambda s: int(s.split("/")[-1][:-4]))

        for image_path in tqdm(image_paths, desc="Initializing Object Views from Images"):
            view = ObjView(image_path)
            view.detectFeatures(detector)
            views.append(view)
        return views

    def plot_keypoints(self) -> None:
        plot = cv2.drawKeypoints(self.img_array, self.kp, self.img_array, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(plot)
        plt.show()

class Scene:
    def __init__(self, views: list[ObjView]) -> None:
        self.views = views
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.computeMatches()

    def computeMatches(self) -> None:
        self.matches = {}
        num_views = len(self.views)
        for i in tqdm(range(num_views), desc="Matching Object Views"):
            for j in range(num_views):
                match = self.matcher.match(self.views[i].desc, self.views[j].desc)
                match = sorted(match, key=lambda x: x.distance)
                self.matches[(self.views[i].img_name, self.views[j].img_name)] = match

    def plot_matches(self, objview1: ObjView, objview2: ObjView, top_k: int = 100) -> None:
        plot = cv2.drawMatches(
                objview1.img_array, 
                objview1.kp, 
                objview2.img_array, 
                objview1.kp, 
                self.matches[(objview1.img_name, objview2.img_name)][:top_k], 
                None, flags = 
                cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        plt.imshow(plot)
        plt.show()