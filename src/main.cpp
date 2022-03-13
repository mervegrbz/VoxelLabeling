#include <cstdlib>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc.hpp>
using namespace nl_uu_science_gmt;
using namespace std;
using namespace cv;

string data_path = "data" + string(PATH_SEP);

void Calibrate_Cameras(int num_cams) {
    FileStorage fs;
    fs.open(data_path + "checkerboard.xml", FileStorage::READ);
    int BORDER_SIZE = fs["CheckerBoardSquareSize"];
    int CHECKERBOARD[2];
    CHECKERBOARD[0] = fs["CheckerBoardWidth"];
    CHECKERBOARD[1] = fs["CheckerBoardHeight"];
    fs.release();
    vector<Point3f> objp;
    for (int i{ 0 }; i < CHECKERBOARD[1]; i++)
    {
        for (int j{ 0 }; j < CHECKERBOARD[0]; j++)
        {
            objp.push_back(Point3f(BORDER_SIZE * j, BORDER_SIZE * i, 0));
        }
    }
    TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
    const std::string cam_path = data_path + "cam";
    VideoCapture in_video;
	for (int v = 0; v < num_cams; ++v)
	{
	    stringstream full_path;
		full_path << cam_path << (v + 1) << PATH_SEP;
		cout << "Cam " << full_path.str() << endl;
        // open camera
        in_video.open(full_path.str()+ "intrinsics.avi");
        Mat cameraMatrix;
        Mat distCoeffs, R, T;
        Mat rvec, tvec;
        int search_frame = 50 * 2; // every 3 seconds;
        if (in_video.isOpened()) {
            // Creating vector to store vectors of 3D points for each checkerboard image
            vector<vector<Point3f>> objpoints;
            // Creating vector to store vectors of 2D points for each checkerboard image
            vector<vector<Point2f>> imgpoints;
            Mat frame, gray;
            vector<Point2f> corner_pts;
            bool success;
            int framecount = 0;
            vector<Mat> used_images;
            while (1){


                in_video >> frame;
                if (frame.empty()) {
                    break; 
                }
                if (framecount % search_frame != 0) {
                    framecount++;
                    continue;
                    
                }
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);
                if (success)
                {

                    used_images.push_back(frame.clone());


                    cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);
                    drawChessboardCorners(frame, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
                    objpoints.push_back(objp);
                    imgpoints.push_back(corner_pts);
                }
                framecount++;
            }


            cout << "Images found: " << used_images.size() << endl;
            float myerror = calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
            cout << "Original Error: " << myerror << endl;
            // optomize
            int count = 0;
            int target = objpoints.size();
            while (count < target)
            {

                objp = objpoints[0];
                corner_pts = imgpoints[0];

                objpoints.erase(objpoints.begin());
                imgpoints.erase(imgpoints.begin());

                float mynewerror = calibrateCamera(objpoints, imgpoints, Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
                if (mynewerror > myerror)
                {
                    objpoints.push_back(objp);
                    imgpoints.push_back(corner_pts);
                }
                else
                {
                    myerror = mynewerror;
                    cout << "New Error: " << myerror << endl;
                } 
                count++;

            }

            Mat test_img = used_images[10];
            cvtColor(test_img, gray, COLOR_BGR2GRAY);
            success = findChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE);

            if (success) {
                
                cornerSubPix(gray, corner_pts, Size(11, 11), Size(-1, -1), criteria);
                solvePnP(objp, corner_pts, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);
                drawFrameAxes(test_img, cameraMatrix, distCoeffs, rvec, tvec, BORDER_SIZE);
            }
            imshow("Test immage", test_img);
           
        }
        else {
            continue;
        }

        fs.open(full_path.str() + "intrinsics.xml", FileStorage::WRITE);

        fs << "CameraMatrix" << cameraMatrix;
        fs << "DistortionCoeffs" << distCoeffs;
        in_video.release();
	}

}

int main(
		int argc, char** argv)
{
	VoxelReconstruction::showKeys();
	VoxelReconstruction vr(data_path, 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}