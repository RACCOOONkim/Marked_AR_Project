import cv2
import numpy as np
import threading
from collections import deque
from objloader_simple import OBJ  # OBJ 모델 로더

# 비디오 캡처를 위한 최적화된 클래스 정의
class VideoCapture:
    """ 최적화된 VideoCapture로 deque 크기를 제한합니다. """
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

# 카메라 매개변수와 호모그래피를 사용하여 3D 투영 행렬 계산
def projection_matrix(camera_parameters, homography):
    """
    카메라 보정 매트릭스와 추정된 호모그래피를 사용하여 3D 프로젝션 매트릭스를 계산합니다.
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

# 3D 모델을 현재 비디오 프레임에 렌더링
def render(frame, obj, projection, referenceImage, scale3d, rotation_angle, color=False):
    """
    현재 비디오 프레임에 로드된 obj 모델을 렌더링합니다.
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape

    # 회전 변환 행렬 생성
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                [0, 0, 1]])

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # 회전 적용
        points = np.dot(points, rotation_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        if color:
            cv2.fillConvexPoly(frame, framePts, (137, 27, 211))
        else:
            cv2.fillConvexPoly(frame, framePts, (255, 255, 255))

    return frame

# 참조 이미지와 OBJ 모델 정보를 저장하기 위한 클래스
class ReferenceObject:
    def __init__(self, image_path, model_path, scale, sift):
        self.image = cv2.imread(image_path, 0)
        self.model = OBJ(model_path, swapyz=True)
        self.scale = scale
        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)

# 주 실행 로직을 포함하는 메인 함수
def main():
    # SIFT 초기화
    sift = cv2.SIFT_create()

    # 3D 모델 및 참조 이미지 로드
    references = [
        ReferenceObject("./img/ref1.jpg", "./models/Woman.obj", 1, sift),
        ReferenceObject("./img/ref2.jpg", "./models/Man.obj", 1, sift)
    ]

    # 카메라 매개변수 매트릭스
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # 호모그래피에 필요한 최소 매칭 수
    MIN_MATCHES = 15

    # SIFT 및 FLANN 초기화
    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 비디오 캡처 초기화
    cap = VideoCapture(0)

    rotation_angle = 0  # 초기 회전 각도
    rotation_speed = np.radians(1)  # 각 프레임당 회전 속도 (라디안 단위)
    
    while True:
        frame = cap.read()
        if frame is None:
            break

        frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

        if frame_descriptors is not None:
            for ref in references:
                # 각 참조 이미지에 대한 매칭 및 호모그래피 계산
                if ref.descriptors is not None:
                    matches = flann.knnMatch(ref.descriptors, frame_descriptors, k=2)
                    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

                    if len(good_matches) > MIN_MATCHES:
                        src_pts = np.float32([ref.keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if homography is not None:
                            inlier_ratio = np.sum(mask) / len(good_matches)
                            if inlier_ratio > 0.5:
                                projection = projection_matrix(camera_parameters, homography)
                                frame = render(frame, ref.model, projection, ref.image, ref.scale, rotation_angle, False)

        rotation_angle += rotation_speed
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()