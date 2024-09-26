import cv2
import numpy as np
import os

def calibrate_camera(images_dir, chessboard_size):
    # Prepare object points and image points
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Read images
    images = [os.path.join(images_dir, fname) for fname in sorted(os.listdir(images_dir)) if fname.endswith('.png')]

    for image_file in images:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)
        else:
            print(f"Chessboard corners not found in {image_file}")

    cv2.destroyAllWindows()

    # Perform camera calibration
    if len(objpoints) > 0 and len(imgpoints) > 0:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            print("Calibration successful.")
            print("Camera matrix:\n", K)
            print("Distortion coefficients:\n", dist)

            # Print and save the extrinsic parameters
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                print(f"Image {i+1} - Rotation Matrix:\n", rotation_matrix)
                print(f"Image {i+1} - Translation Vector:\n", tvec)

            return K, dist, rvecs, tvecs
        else:
            print("Calibration failed.")
            return None, None, None, None
    else:
        print("Not enough points to perform calibration.")
        return None, None, None, None

def main():
    try:
        # Directory containing chessboard images
        calibration_images_dir = "/home/demo/Desktop/calibration_50013"
        chessboard_size = (8, 6)  # Number of inner corners per chessboard row and column

        # Perform camera calibration
        K, dist, rvecs, tvecs = calibrate_camera(calibration_images_dir, chessboard_size)

        if K is not None and dist is not None:
            # Save calibration data
            np.savez(os.path.join(calibration_images_dir, 'calibration_data.npz'), camera_matrix=K, distortion_coefficients=dist, rvecs=rvecs, tvecs=tvecs)
            print("Calibration data saved.")
        else:
            print("Calibration data not saved.")

    except Exception as e:
        print(f"Exception in main: {e}")

if __name__ == "__main__":
    main()
