import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt

boards = glob.glob('/home/firefox/bag_files/calibration2/*.png')
boards = [cv2.imread(board) for board in boards]
obj_point = np.zeros((7*7, 3), np.float32)
obj_point[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
obj_points = []
corner_points = []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
print(f'{len(boards)} boards in folder')
for board in boards:
    # print('board:' ,board)
    cols, rows = board.shape[:2]
    gray_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray_board, (7,7), None)
    if retval:
        refined_corners = cv2.cornerSubPix(gray_board, corners, (11,11), (-1, -1), criteria)
        corner_points.append(refined_corners)
        obj_points.append(obj_point)
        plt.figure()
        cv2.drawChessboardCorners(board, (7,7), corners, retval)
        plt.imshow(board)

ret, camera_matrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, corner_points, boards[0].shape[1::-1], None, None)
print(camera_matrix)
print(distCoeffs)
        



plt.show()