import cv2
import numpy as np
import spectral
import os
from matplotlib import pyplot as plt


class ImageRegistrator:
    """
    自动识别光谱（*.hdr）或普通图像的通用配准器
    """

    def __init__(self,
                 ref_img: np.ndarray,
                 mov_img: np.ndarray,
                 sift_nfeatures: int = 0,
                 lowe_ratio: float = 0.8,
                 min_match: int = 10,
                 ransac_thresh: float = 5.0,
                 vis: bool = False):
        self.lowe_ratio = lowe_ratio
        self.min_match = min_match
        self.ransac_thresh = ransac_thresh
        self.vis = vis

        # 载入灰度图
        is_spectral_ref = ref_img.shape[2] > 3
        is_spectral_mov = mov_img.shape[2] > 3
        self.ref_gray = self._load_gray(ref_img, is_spectral_ref)
        self.mov_gray = self._load_gray(mov_img, is_spectral_mov)

        # 计算单应矩阵
        self.M = self._compute_homography()

    # ---------- 内部 ----------
    @staticmethod
    def _is_spectral(path: str) -> bool:
        return os.path.splitext(path)[1].lower() == '.hdr'

    @staticmethod
    def _load_gray(img: np.ndarray, is_spectral: bool):
        if is_spectral:
            # img = spectral.open_image(path).load()
            mid = img.shape[2] // 2
            gray = img[:, :, mid]
            gray = (gray - gray.min()) / (gray.max() - gray.min())
            gray = (gray * 255).astype(np.uint8)
            gray = cv2.flip(gray, 1)  # 与之前的处理保持一致
        else:
            raw = img
            # raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if len(raw.shape) == 3 else raw
        return gray

    @staticmethod
    def normalized_img(img):
        """
        归一化图像到0-255范围并转换为8位无符号整数格式。
        
        参数:
        img: 输入图像，假设为多通道（如RGB或其他）格式。
        
        返回:
        img_8bit: 归一化后的图像，格式为8位无符号整数。
        """
        # 提取中间通道并归一化到0-1范围
        mid_channel_index = int(img.shape[2] / 2)
        img_normalized = img[:, :, mid_channel_index]
        img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())

        # 限制归一化后的值到0-1范围
        img_clipped = np.clip(img_normalized, 0, 1)

        # 转换为无符号8位整数格式（0-255）
        img_8bit = (img_clipped * 255).astype(np.uint8)

        return img_8bit
    def _compute_homography(self) -> np.ndarray:
        sift = cv2.SIFT_create(nfeatures=0)
        kp1, des1 = sift.detectAndCompute(self.mov_gray, None)
        kp2, des2 = sift.detectAndCompute(self.ref_gray, None)

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)

        good = [m for m, n in matches if m.distance < self.lowe_ratio * n.distance]
        if len(good) < self.min_match:
            raise RuntimeError(f"Matches {len(good)} < {self.min_match}")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)  
        if self.vis:
            self._vis_matches(kp1, kp2, good, mask)
        return M

    def _vis_matches(self, kp1, kp2, good, mask):
        img = cv2.drawMatches(self.mov_gray, kp1, self.ref_gray, kp2, good,
                              None, matchColor=(0, 255, 0),
                              singlePointColor=None,
                              matchesMask=mask.ravel().tolist(), flags=2)
        plt.figure(figsize=(12, 6))
        plt.title("Feature Matches")
        plt.imshow(img, cmap='gray')
        plt.show()

    # ---------- 公共接口 ----------
    def warp_image(self, img: np.ndarray) -> np.ndarray:
        img = cv2.flip(img, 1)
        h, w = self.ref_gray.shape[:2]

        warped = cv2.warpAffine(img, self.M, (w, h))
        return warped

    def warp_mov_file(self, out_path: str = None) -> np.ndarray:
        if self.is_spectral_mov:
            img = spectral.open_image(self.mov_path).load()
        else:
            img = cv2.imdecode(np.fromfile(self.mov_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        warped = self.warp_image(img)

        if out_path:
            # 保存
            if self.is_spectral_mov:
                save_img = self.normalized_img(warped)
                cv2.imwrite(out_path, save_img)
                print('光谱图像已经输出中间波段配准图像提供查看')
            else:
                cv2.imwrite(out_path, warped)
        return warped
    
if __name__ == '__main__':
    f1 = rf'samples\clean\1_20250807_092406\hk\20250807T092411_483.png'
    f2 = rf'samples\clean\1_20250807_092406\vSpex\vSpex_20250807T092409_053.hdr'
    im1 = cv2.imdecode(np.fromfile(f1, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    hsi = spectral.open_image(f2).load()
    reg1 = ImageRegistrator(
        ref_img=hsi,
        mov_img=im1,
        vis=True
    )
    warped1 = reg1.warp_image(im1)
    print(warped1.shape, hsi.shape)

    plt.subplot(121)
    plt.imshow(warped1)
    plt.subplot(122)
    im = hsi[:,:,3]
    im = (im - im.min())/(im.max()-im.min())*255
    im = im.astype(np.uint8)
    # im = cv2.imread(f1)
    plt.imshow(im)
    plt.show()

    # 原始
    plt.subplot(121)
    plt.imshow(im1)
    plt.subplot(122)
    plt.imshow(im)
    plt.show()

