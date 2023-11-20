#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""파이썬 3 및 OpenCV 4.2를 사용한 증강 현실
"""

__author__ = "ma. fernanda rodriguez r."
__email__ = "mafda13@gmail.com"
__created__ = "Thu 14 May 2020 11:40:54 -0300"
__modified__ = "Thu 29 May 2020 15:13:00 -0300"


import cv2
import math
import threading
import numpy as np
import matplotlib.pyplot as plt
from objloader_simple import *
from collections import deque


class VideoCapture:
    """버퍼 없는 VideoCapture
    """

    def __init__(self, name, res=(320, 240)):
        self.cap = cv2.VideoCapture(name)
        self.cap.set(3, res[0])
        self.cap.set(4, res[1])
        self.q = deque()
        self.status = "init"

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

        while self.status == "init":
            pass

        assert self.status == "capture", "캡처를 열 수 없습니다."

    def _reader(self):
        """프레임이 사용 가능한 즉시 읽어들이고, 가장 최근의 것만 유지
        """

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[에러] ret")
                break

            self.q.append(frame)

            self.status = "capture"

            while len(self.q) > 1:
                self.q.popleft()

        self.status = "failed"

    def read(self):
        return self.q[-1]

    def release(self):
        self.cap.release()


def projection_matrix(camera_parameters, homography):
    """
    카메라 보정 행렬 및 추정된 호모그래피에서 3D 투영 행렬 계산
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # 벡터 정규화
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # 직교 정규 기저 계산
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(
        c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_2 = np.dot(
        c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2)
    )
    rot_3 = np.cross(rot_1, rot_2)

    # 모델에서 현재 프레임으로의 3D 투영 행렬 계산
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)


def render(frame, obj, projection, referenceImage, scale3d, color=False):
    """
    로드된 obj 모델을 현재 비디오 프레임으로 렌더링
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # 모델을 참조 표면의 중간에 렌더링합니다.
        # 이를 위해서는 모델의 포인트를 이동해야 합니다.
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv2.fillConvexPoly(frame, framePts, (137, 27, 211))

    return frame


def main():

    # ============== 데이터 읽기 ==============

    # OBJ 파일에서 3D 모델 로드
    obj = OBJ("./models/fox.obj", swapyz=True)

    # 3D 모델 스케일
    scale3d = 8

    # 카메라 파라미터 행렬
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

    # 최소 일치 수
    MIN_MATCHES = 15

    # ============== 참조 이미지 ==============

    # 참조 이미지 로드 및 그레이 스케일로 변환
    referenceImage = cv2.imread("./img/referenceImage.jpg", 0)

    # =============== 인식 ==============

    # ORB 검출기 초기화
    orb = cv2.ORB_create()

    # 브루트 포스 매처 객체 생성
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 모델의 키포인트와 디스크립터 계산
    referenceImagePts, referenceImageDsc = orb.detectAndCompute(referenceImage, None)

    # =============== 소스 이미지 ==============

    # 비디오 캡처 초기화 (소스 이미지 로드)
    cap = VideoCapture(0)

    while True:
        # 현재 프레임 읽기
        frame = cap.read()

        # =============== 인식 ==============

        # 씬의 키포인트와 디스크립터 계산
        sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)

        # 디스크립터가 존재하는지 확인
        if referenceImageDsc is not None and sourceImageDsc is not None:
            # ============== 일치 =============

            # 프레임 디스크립터를 모델 디스크립터와 일치시킵니다.
            matches = bf.match(referenceImageDsc, sourceImageDsc)

            # 거리 순으로 정렬
            matches = sorted(matches, key=lambda x: x.distance)

            # ============== 호모그래피 =============

            # 충분한 좋은 일치가 있으면 호모그래피 변환을 적용합니다.
            if len(matches) > MIN_MATCHES:
                # 좋은 키포인트 위치 가져오기
                sourcePoints = np.float32(
                    [referenceImagePts[m.queryIdx].pt for m in matches]
                ).reshape(-1, 1, 2)
                destinationPoints = np.float32(
                    [sourceImagePts[m.trainIdx].pt for m in matches]
                ).reshape(-1, 1, 2)

                # 호모그래피 행렬 얻기
                homography, mask = cv2.findHomography(
                    sourcePoints, destinationPoints, cv2.RANSAC, 5.0
                )

                # 적합성을 향상시키기 위해 변환된 코너들만 사용
                inlier_ratio = np.sum(mask) / len(matches)
                print(f"inlier_ratio: {inlier_ratio}")
                # 인라이어 비율이 80% 이상인 경우에만 ROI 그리고 증강
                if inlier_ratio > 0.25:
                    # 변환된 모서리를 연결하여 두 번째 이미지에 다각형을 그립니다.
                    h, w = referenceImage.shape
                    corners = np.float32(
                        [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                    ).reshape(-1, 1, 2)
                    transformedCorners = cv2.perspectiveTransform(corners, homography)

                    frame = cv2.polylines(
                        frame, [np.int32(transformedCorners)], True, (255, 0, 0), 3, cv2.LINE_AA,
                    )

                    # ================= 포즈 추정 ================

                    # 호모그래피 행렬과 카메라 파라미터로부터 3D 투영 행렬 얻기
                    projection = projection_matrix(camera_parameters, homography)

                    # 큐브 또는 모델 투영
                    frame = render(frame, obj, projection, referenceImage, scale3d, False)


        # ===================== 디스플레이 ====================

        # 결과 표시
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
