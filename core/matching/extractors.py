import cv2

class FeatureExtractor:
    def extract(self, img_gray):
        raise NotImplementedError

class SIFTExtractor(FeatureExtractor):
    def __init__(self, nfeatures=2000):
        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)
        
    def extract(self, img_gray):
        kp, des = self.sift.detectAndCompute(img_gray, None)
        return kp, des

    def __getstate__(self):
        return {'nfeatures': self.nfeatures}

    def __setstate__(self, state):
        self.nfeatures = state.get('nfeatures', 2000)
        self.sift = cv2.SIFT_create(nfeatures=self.nfeatures)

class ORBExtractor(FeatureExtractor):
    def __init__(self, nfeatures=2000):
        self.nfeatures = nfeatures
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures)
        
    def extract(self, img_gray):
        kp, des = self.orb.detectAndCompute(img_gray, None)
        return kp, des

    def __getstate__(self):
        return {'nfeatures': self.nfeatures}

    def __setstate__(self, state):
        self.nfeatures = state.get('nfeatures', 2000)
        self.orb = cv2.ORB_create(nfeatures=self.nfeatures)
