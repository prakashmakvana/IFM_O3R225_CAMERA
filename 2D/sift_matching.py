import numpy as np
import cv2

class SiftKpDesc:
    def __init__(self, kp, desc):
        self.kp = kp  # List of keypoints in (x, y) coordinates
        self.desc = desc  # List of descriptors at keypoints

class SiftMatching:
    _BLUE = [255, 0, 0]
    _CYAN = [255, 255, 0]
    _line_thickness = 2
    _radius = 5
    _circ_thickness = 2

    def __init__(self, img1, img2, nfeatures=2000, gamma=0.8):
        self.img1_bgr = img1
        self.img2_bgr = img2
        self.nfeatures = nfeatures
        self.gamma = gamma

    def get_sift_features(self, img_bgr, nfeatures=2000):
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        sift_obj = cv2.SIFT_create(nfeatures)
        kp_list_obj, desc = sift_obj.detectAndCompute(image=img_gray, mask=None)
        kp = [x.pt for x in kp_list_obj]
        return SiftKpDesc(kp, desc)

    def match_features(self, sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=0.8):
        correspondence = []  # list of lists of [x1, y1, x2, y2]
        for i in range(len(sift_kp_desc_obj1.kp)):
            sc = np.linalg.norm(sift_kp_desc_obj1.desc[i] - sift_kp_desc_obj2.desc, axis=1)
            idx = np.argsort(sc)
            val = sc[idx[0]] / sc[idx[1]]
            if val <= gamma:
                correspondence.append([*sift_kp_desc_obj1.kp[i], *sift_kp_desc_obj2.kp[idx[0]]])
        return correspondence

    def draw_correspondence(self, correspondence, img1, img2):
        h, w, _ = img1.shape
        img_stack = np.hstack((img1, img2))
        for x1, y1, x2, y2 in correspondence:
            x1_d = int(round(x1))
            y1_d = int(round(y1))
            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))
            cv2.circle(img_stack, (x1_d, y1_d), radius=self._radius, color=self._BLUE,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)
            cv2.circle(img_stack, (x2_d, y2_d), radius=self._radius, color=self._BLUE,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)
            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=self._CYAN,
                     thickness=self._line_thickness)
        return img_stack

    def run(self):
        sift_kp_desc_obj1 = self.get_sift_features(self.img1_bgr, nfeatures=self.nfeatures)
        sift_kp_desc_obj2 = self.get_sift_features(self.img2_bgr, nfeatures=self.nfeatures)
        correspondence = self.match_features(sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=self.gamma)
        return correspondence
