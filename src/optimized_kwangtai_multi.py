import cv2
import numpy as np
import threading
from collections import deque
from objloader_simple import OBJ  # OBJ 모델 로더

class VideoCapture:
    """Optimized VideoCapture to limit deque size."""
    def __init__(self, name, res=(320, 240), max_frames=2):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        self.q = deque(maxlen=max_frames)
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "Cannot open capture."

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.status = "failed"
                break

            self.q.append(frame)
            self.status = "capture"

    def read(self):
        return self.q[-1] if self.q else None

    def release(self):
        self.cap.release()

def projection_matrix(camera_parameters, homography):
    """
    Calculate the 3D projection matrix from the camera calibration matrix and the estimated homography.
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    l = np.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def render(frame, obj, projection, referenceImage, scale3d, color=False):
    """
    Render the loaded obj model onto the current video frame.
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        if color:
            cv2.fillConvexPoly(frame, framePts, (137, 27, 211))
        else:
            cv2.fillConvexPoly(frame, framePts, (255, 255, 255))

    return frame
class ReferenceObject:
    def __init__(self, image_path, model_path, scale, sift):
        self.image = cv2.imread(image_path, 0)
        self.model = OBJ(model_path, swapyz=True)
        self.scale = scale
        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)

def main():
    # Initialize SIFT
    sift = cv2.SIFT_create()

    # Load the 3D models and reference images
    references = [
        ReferenceObject("./img/referenceImage.jpg", "./models/Woman.obj", 2, sift),
        ReferenceObject("./img/referenceImage3.jpg", "./models/Man.obj", 2, sift)
    ]

    # Camera parameters matrix
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # Minimum number of matches for homography
    MIN_MATCHES = 15

    # Initialize SIFT and FLANN
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Initialize video capture
    cap = VideoCapture(0)

    while True:
        frame = cap.read()
        if frame is None:
            break

        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

        if frame_descriptors is not None:
            for ref in references:
                if ref.descriptors is not None:
                    matches = flann.knnMatch(ref.descriptors, frame_descriptors, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                    if len(good_matches) > MIN_MATCHES:
                        src_pts = np.float32([ref.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if homography is not None:
                            # inlier_ratio 계산
                            inlier_ratio = np.sum(mask) / len(good_matches)
                            print(f"inlier_ratio: {inlier_ratio}")

                            # 인라이어 비율이 특정 임계값 이상인 경우에만 렌더링
                            if inlier_ratio > 0.5:
                                projection = projection_matrix(camera_parameters, homography)
                                frame = render(frame, ref.model, projection, ref.image, ref.scale, False)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()