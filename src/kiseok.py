import cv2
import numpy as np
from objloader_simple import *
import math

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]

    # Normalize vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    # Compute the orthonormal basis
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

    # Compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)


def render(frame, obj,obj_2,projection, referenceImage, scale3d,scale3d_2,radian, color=True):
    """
    Render a loaded obj model into the current video frame
    """
    
    vertices = obj.vertices
    scale_matrix = np.eye(3) * scale3d
    h, w = referenceImage.shape
    # 프레임당 회전 각도 (라디안으로 변환)
    rotation_angle_rad = np.radians(radian)

    # 회전 변환 행렬 생성
    rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad), 0],
                                [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad), 0],
                                [0, 0, 1]])
    
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.dot(points, rotation_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] , p[1] , p[2]+w/2] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        framePts = np.int32(dst)

        cv2.fillConvexPoly(frame, framePts, (137, 27, 211))


    vertices_2 = obj_2.vertices
    scale_matrix_2 = np.eye(3) * scale3d_2
    for face in obj_2.faces:
        face_vertices_2 = face[0]
        points_2 = np.array([vertices_2[vertex - 1] for vertex in face_vertices_2])
        points_2 = np.dot(points_2, scale_matrix_2)
        
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points_2 = np.array([[p[0] + w, p[1] + h, p[2]] for p in points_2])   #projection 위치
        
        dst = cv2.perspectiveTransform(points_2.reshape(-1, 1, 3), projection)
        framePts_2 = np.int32(dst)
        cv2.fillConvexPoly(frame, framePts_2, (137, 27, 211))
      
    return frame
obj = OBJ("./models/chair.obj", swapyz=True)
obj_2 =OBJ("./models/fox.obj", swapyz=True)
# Scale 3D model
scale3d = 8
scale3d_2 = 1.8

referenceImage = cv2.imread("./img/referenceImage1.jpg", 0)
frame = cv2.imread('./img/test.jpg',cv2.IMREAD_COLOR)

h, w = referenceImage.shape
camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
src = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
src_gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
_ , src_bin = cv2.threshold(src_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(src_bin,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#컨투어를 반복하며 각 컨투어를 다각형으로 근사화하고 컨투어 영역 및 꼭지점 수에 따라 필터링
for pts in contours:
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts,True)*0.02,True)
    if cv2.contourArea(pts) < 1000:
        continue
    if len(approx) !=4:
        continue

#소스 및 대상 사각형을 정의하고 퍼스펙티브 변환 행렬을 계산
    srcQuad = np.array([[approx[0,0, :]], [approx[1,0, :]],
                                            [approx[2,0, :]], [approx[3,0,:]]]).astype(np.float32)
    dstQuad = np.array([[0,0],[0,h],[w,h],[w,0]]).astype(np.float32)
    pers = cv2.getPerspectiveTransform(srcQuad,dstQuad)
    #dst = cv2.warpPerspective(src,pers,(w,h))

 #Draw a polygon on the second image joining the transformed corners
frame = cv2.polylines(
        src,[np.int32(srcQuad)],True, 1, 3, cv2.LINE_AA
)

# obtain 3D projection matrix from homography matrix and camera parameters
projection = projection_matrix(camera_parameters, pers)
frame = render(frame, obj,obj_2, projection, referenceImage, scale3d,scale3d_2,0,True)

print(pers)

cv2.imshow('src',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()