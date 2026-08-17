import cv2
import numpy as np
import torch
from .extractors import FeatureExtractor

class MatchingResult:
    def __init__(self, score: float, inliers: int, spatial_coverage: float, mean_distance: float, inlier_src_pts=None, inlier_dst_pts=None):
        self.score = score
        self.inliers = inliers
        self.spatial_coverage = spatial_coverage
        self.mean_distance = mean_distance
        self.inlier_src_pts = inlier_src_pts
        self.inlier_dst_pts = inlier_dst_pts

def compute_robust_score(kp1, kp2, matches, inlier_mask, img1_shape) -> MatchingResult:
    total_matches = len(matches)
    num_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    
    if num_inliers < 10 or total_matches == 0:
        return MatchingResult(0.0, num_inliers, 0.0, float('inf'))
        
    inlier_ratio = num_inliers / total_matches
    
    inlier_src_pts = []
    inlier_dst_pts = []
    distances = []
    
    for i, m in enumerate(matches):
        if inlier_mask is None or inlier_mask[i]:
            inlier_src_pts.append(kp1[m.queryIdx].pt)
            inlier_dst_pts.append(kp2[m.trainIdx].pt)
            distances.append(m.distance)
            
    inlier_src_pts = np.array(inlier_src_pts, dtype=np.float32)
    inlier_dst_pts = np.array(inlier_dst_pts, dtype=np.float32)
    
    min_x, min_y = np.min(inlier_src_pts, axis=0)
    max_x, max_y = np.max(inlier_src_pts, axis=0)
    
    bbox_area = max(1, (max_x - min_x) * (max_y - min_y))
    img_area = img1_shape[0] * img1_shape[1]
    
    spatial_coverage = bbox_area / img_area
    mean_distance = float(np.mean(distances))
    
    score = num_inliers * inlier_ratio * spatial_coverage * (1.0 / (1.0 + mean_distance))
    
    return MatchingResult(score, num_inliers, spatial_coverage, mean_distance, inlier_src_pts, inlier_dst_pts)

class DeepExtractorMatcher:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        print(f"Inicializando DeepExtractorMatcher en {self.device}...")
        try:
            from lightglue import SuperPoint, LightGlue
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self._use_kornia = False
        except ImportError:
            import kornia.feature as KF
            self.extractor = KF.SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = KF.LightGlue("superpoint").eval().to(self.device)
            self._use_kornia = True
        
    def extract(self, img_gray):
        t1 = torch.from_numpy(img_gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            if self._use_kornia:
                dict1 = self.extractor(t1)
            else:
                dict1 = self.extractor({"image": t1})
        return dict1["keypoints"], dict1["descriptors"]
        
    def match_images(self, img1_gray, img2_gray):
        kp1, des1 = self.extract(img1_gray)
        kp2, des2 = self.extract(img2_gray)
        return self.match_keypoints(kp1, des1, kp2, des2, img1_gray.shape)
        
    def match_keypoints(self, kp1, des1, kp2, des2, img1_shape):
        dummy_img = torch.empty((1, 1, img1_shape[0], img1_shape[1]), device=self.device)
        la1 = {"image0": {"keypoints": kp1, "descriptors": des1, "image": dummy_img}}
        la2 = {"image1": {"keypoints": kp2, "descriptors": des2, "image": dummy_img}}
        
        with torch.no_grad():
            out = self.matcher({"image0": la1["image0"], "image1": la2["image1"]})
            
            matches = out["matches"][0]
            scores = out["scores"][0]
            
            num_inliers = matches.shape[0]
            if num_inliers < 10:
                return MatchingResult(0.0, num_inliers, 0.0, float('inf'))
                
            kp1_matched = kp1[0, matches[:, 0]].cpu().numpy()
            kp2_matched = kp2[0, matches[:, 1]].cpu().numpy()
            
            min_x, min_y = np.min(kp1_matched, axis=0)
            max_x, max_y = np.max(kp1_matched, axis=0)
            bbox_area = max(1, (max_x - min_x) * (max_y - min_y))
            img_area = img1_shape[0] * img1_shape[1]
            spatial_coverage = bbox_area / img_area
            
            mean_dist = float(1.0 - torch.mean(scores).cpu().numpy())
            score_final = num_inliers * (1.0 + spatial_coverage)
            
            return MatchingResult(score_final, num_inliers, spatial_coverage, mean_dist, np.float32(kp1_matched), np.float32(kp2_matched))

class ClassicalMatcher:
    def __init__(self, method="sift"):
        self.method = method
        if method == "sift":
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif method == "orb":
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
    def __getstate__(self):
        return {'method': self.method}

    def __setstate__(self, state):
        self.method = state.get('method', 'sift')
        self.__init__(self.method)
        
    def match(self, kp1, des1, kp2, des2, img1_shape):
        if des1 is None or len(des1) < 10 or des2 is None or len(des2) < 10:
            return MatchingResult(0.0, 0, 0.0, float('inf'))
            
        knn_matches = self.matcher.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m_obj in knn_matches:
            if len(m_obj) == 2:
                m, n = m_obj
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        if len(good_matches) < 10:
            return MatchingResult(0.0, len(good_matches), 0.0, float('inf'))
            
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
        
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)
        return compute_robust_score(kp1, kp2, good_matches, mask.ravel() if mask is not None else None, img1_shape)
