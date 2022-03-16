/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

	/**
	 * Constructor
	 * Voxel reconstruction class
	 */
	Reconstructor::Reconstructor(
		const vector<Camera *> &cs) : m_cameras(cs),
									  m_height(2048),
									  m_step(32)
	{
		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (m_plane_size.area() > 0)
				assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
			else
				m_plane_size = m_cameras[c]->getSize();
		}

		const size_t edge = 2 * m_height;
		m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

		initialize();
	}

	/**
	 * Deconstructor
	 * Free the memory of the pointer vectors
	 */
	Reconstructor::~Reconstructor()
	{
		for (size_t c = 0; c < m_corners.size(); ++c)
			delete m_corners.at(c);
		for (size_t v = 0; v < m_voxels.size(); ++v)
			delete m_voxels.at(v);
	}

	/**
	 * Create some Look Up Tables
	 * 	- LUT for the scene's box corners
	 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
	 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
	 */
	void Reconstructor::initialize()
	{
		// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
		const int xL = -m_height;
		const int xR = m_height;
		const int yL = -m_height;
		const int yR = m_height;
		const int zL = 0;
		const int zR = m_height;
		const int plane_y = (yR - yL) / m_step;
		const int plane_x = (xR - xL) / m_step;
		const int plane = plane_y * plane_x;

		// Save the 8 volume corners
		// bottom
		m_corners.push_back(new Point3f((float)xL, (float)yL, (float)zL));
		m_corners.push_back(new Point3f((float)xL, (float)yR, (float)zL));
		m_corners.push_back(new Point3f((float)xR, (float)yR, (float)zL));
		m_corners.push_back(new Point3f((float)xR, (float)yL, (float)zL));

		// top
		m_corners.push_back(new Point3f((float)xL, (float)yL, (float)zR));
		m_corners.push_back(new Point3f((float)xL, (float)yR, (float)zR));
		m_corners.push_back(new Point3f((float)xR, (float)yR, (float)zR));
		m_corners.push_back(new Point3f((float)xR, (float)yL, (float)zR));

		// Acquire some memory for efficiency
		cout << "Initializing " << m_voxels_amount << " voxels ";
		m_voxels.resize(m_voxels_amount);

		int z;
		int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
		for (z = zL; z < zR; z += m_step)
		{
			const int zp = (z - zL) / m_step;
			int done = cvRound((zp * plane / (double)m_voxels_amount) * 100.0);

#pragma omp critical
			if (done > pdone)
			{
				pdone = done;
				cout << done << "%..." << flush;
			}

			int y, x;
			for (y = yL; y < yR; y += m_step)
			{
				const int yp = (y - yL) / m_step;

				for (x = xL; x < xR; x += m_step)
				{
					const int xp = (x - xL) / m_step;

					// Create all voxels
					Voxel *voxel = new Voxel;
					voxel->x = x;
					voxel->y = y;
					voxel->z = z;
					voxel->camera_projection = vector<Point>(m_cameras.size());
					voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

					const int p = zp * plane + yp * plane_x + xp; // The voxel's index

					for (size_t c = 0; c < m_cameras.size(); ++c)
					{
						Point point = m_cameras[c]->projectOnView(Point3f((float)x, (float)y, (float)z));

						// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
						voxel->camera_projection[(int)c] = point;

						// If it's within the camera's FoV, flag the projection
						if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
							voxel->valid_camera_projection[(int)c] = 1;
					}

					// Writing voxel 'p' is not critical as it's unique (thread safe)
					m_voxels[p] = voxel;
				}
			}
		}
		
		path_image = Mat(m_height/4, m_height/4, CV_8UC3);
		path_image = Scalar(255, 255, 255);

		directions = vector<Vec2f>(4);

		directions_found = false;


		frame_count = 0;

		cout << "done!" << endl;
	}

	void Reconstructor::offlinePhase()
	{
		vector<Point2f> m_groundCoordinates(m_visible_voxels.size());
		vector<Point3f> m_coordinates(m_visible_voxels.size());

		for (int i = 0; i < (int)m_visible_voxels.size(); i++)
		{
			m_groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
			m_coordinates[i] = Point3f(m_visible_voxels[i]->x, m_visible_voxels[i]->y, m_visible_voxels[i]->z);
		}
		Mat labels;

		vector<Mat> clusters(4);
		vector<Point2f> centers;
		kmeans(m_groundCoordinates, 4, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
		
		
		last_centers = centers;
		
		for (int j = 0; j < m_groundCoordinates.size(); j++)
		{
			int flag = labels.at<int>(j);
			clusters[flag].push_back(Point3d(m_visible_voxels[j]->x, m_visible_voxels[j]->y, m_visible_voxels[j]->z));
		}
		Mat frame = imread(m_cameras[0]->getDataPath() + "video.png");
		resize(frame,frame,Size(644,486));
		vector<Point2d> imagepoints;
		vector<vector<cv::Point2d>> imagepointList;
		int width = frame.cols;
		int height = frame.rows;
		int dims = frame.channels();
		int nsamples = width * height;
		Mat result = Mat::zeros(Size(644, 486), CV_8UC3);
		for (int j = 0; j < 4; j++)
		{
			projectPoints(clusters[j], m_cameras[0]->m_rotation_values, m_cameras[0]->m_translation_values, m_cameras[0]->m_camera_matrix, m_cameras[0]->m_distortion_coeffs, imagepoints, noArray(), 0);
			imagepointList.push_back(imagepoints);
			Mat points(nsamples, dims, CV_64FC1);
			int cov_mat_type = cv::ml::EM::COV_MAT_GENERIC;
			for (int i = 0; i < imagepoints.size(); i++)
			{
				double row = imagepoints[i].y;
				double col = imagepoints[i].x;
				Vec3b rgb = frame.at<Vec3b>(row, col);
				
				points.at<double>(i, 0) = static_cast<int>(rgb[0]);
				points.at<double>(i, 1) = static_cast<int>(rgb[1]);
				points.at<double>(i, 2) = static_cast<int>(rgb[2]);
				result.at<Vec3b>(row, col)[0] = rgb[0];
				result.at<Vec3b>(row, col)[1] = rgb[1];
				result.at<Vec3b>(row, col)[2] = rgb[2];
			}
			imshow("EM-Segmentation", result);
			Mat notUsed;
			waitKey(1);
			cv::TermCriteria term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15000, 1e-6);
			Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
			gmm->setClustersNumber(2);
			gmm->setTermCriteria(term);
			gmm->trainEM(points, noArray(), notUsed, noArray());
			Gmms.push_back(gmm);
		}

		Scalar color_tab[] = {
			Scalar(255, 0, 0),
			Scalar(0, 255, 0),
			Scalar(0, 0, 255),
			Scalar(255, 0, 255),
		};
		vector<int> matches(4, -1);
		// iterate over gmms
		for (int gm = 0; gm < 4; gm++)
		{
			int person = 0;
			double mostprob = 10000;
			Ptr<cv::ml::EM> gmm = Gmms[gm];
			// iterate over people
			for (int m = 0; m < 4; m++)
			{
				vector<Point2d> newpoints = imagepointList[m];

				vector<Mat> samples;
				Mat sample(1, 3, CV_64FC1);

				double r = 0, g = 0, b = 0;
				double totaldist = 0;
				for (int i = 0; i < newpoints.size(); i++)
				{
					// Get the color of each channel
					double row = newpoints[i].y;
					double col = newpoints[i].x;
					Vec3b rgb = frame.at<Vec3b>(row, col);
					b = (rgb[2]);
					g = (rgb[1]);
					r = (rgb[0]);
					
					// Put pixels in sample data
					sample.at<double>(0, 0) = static_cast<double>(r);
					sample.at<double>(0, 1) = static_cast<double>(g);
					sample.at<double>(0, 2) = static_cast<double>(b);
					totaldist += gmm->predict2(sample, noArray())[0];
					samples.push_back(sample);
				}

				if (mostprob > abs(totaldist / samples.size()))
				{
					if (matches[m] != -1)
					{
						continue;
					}
					person = m;
					mostprob = abs(totaldist / samples.size());
				}
			}
			matches[person] = gm;
		}
		// iterate over person
		for (int h = 0; h < 4; h++)
		{

			int gm = matches[h];
			cout << gm << "this is the person I label" << h << endl;
			for (int r = 0; r < imagepointList[h].size(); r++)
			{
				double row = imagepointList[h][r].y;
				double col = imagepointList[h][r].x;
				Scalar c = color_tab[gm];
				result.at<Vec3b>(row, col)[0] = c[0];
				result.at<Vec3b>(row, col)[1] = c[1];
				result.at<Vec3b>(row, col)[2] = c[2];
			}
		}

		for (int j = 0; j < m_groundCoordinates.size(); j++)
		{
			int flag = labels.at<int>(j);
			m_visible_voxels[j]->color = color_tab[matches[flag]];
		}
		imshow("EM-Segmentation", result);
		waitKey(1);
	};

	/**
	 * Count the amount of camera's each voxel in the space appears on,
	 * if that amount equals the amount of cameras, add that voxel to the
	 * visible_voxels vector
	 */
	void Reconstructor::update()
	{
		m_visible_voxels.clear();
		std::vector<Voxel*> visible_voxels;

		int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
		for (v = 0; v < (int)m_voxels_amount; ++v)
		{
			int camera_counter = 0;
			Voxel* voxel = m_voxels[v];

			for (size_t c = 0; c < m_cameras.size(); ++c)
			{
				if (voxel->valid_camera_projection[c])
				{
					const Point point = voxel->camera_projection[c];

					// If there's a white pixel on the foreground image at the projection point, add the camera
					if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255)
						++camera_counter;
				}
			}

			// If the voxel is present on all cameras
			if (camera_counter == m_cameras.size())
			{
#pragma omp critical // push_back is critical
				visible_voxels.push_back(voxel);
			}
		}
		m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

		vector<Point2f> m_groundCoordinates(m_visible_voxels.size());
		vector<Point3f> m_coordinates(m_visible_voxels.size());

		for (int i = 0; i < (int)m_visible_voxels.size(); i++)
		{
			m_groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
			m_coordinates[i] = Point3f(m_visible_voxels[i]->x, m_visible_voxels[i]->y, m_visible_voxels[i]->z);
		}
		if (!isTrained)
		{
			offlinePhase();
			isTrained = true;
		}
		Mat labels;
		vector<Voxel> clusterpoints;
		vector<vector<Point3d>> clusters(4);
		vector<Point2f> centers;
		vector<vector<cv::Point2d>> imagepointList;
		kmeans(m_groundCoordinates, 4, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
		for (int j = 0; j < m_groundCoordinates.size(); j++)
		{
			int flag = labels.at<int>(j);
			clusters[flag].push_back(Point3d(m_visible_voxels[j]->x, m_visible_voxels[j]->y, m_visible_voxels[j]->z));

		}
		for (int f = 0; f < 4; f++)
		{
			Mat imagepoints;
			projectPoints(clusters[f], m_cameras[0]->m_rotation_values, m_cameras[0]->m_translation_values, m_cameras[0]->m_camera_matrix, m_cameras[0]->m_distortion_coeffs, imagepoints, noArray(), 0);
			imagepointList.push_back(imagepoints);
		}
		Mat frame = m_cameras[0]->getFrame();
		resize(frame, frame, Size(644, 486));
		Mat result = Mat::zeros(Size(644, 486), CV_8UC3);

		Scalar color_tab[] = {
			Scalar(255, 0, 0),
			Scalar(0, 255, 0),
			Scalar(0, 0, 255),
			Scalar(255, 0, 255),
		};
		vector<int> matches(4, -1);
		// iterate over gmms
		for (int gm = 0; gm < 4; gm++)
		{
			int person = 0;
			double mostprob = 10000;
			// iterate over people
			for (int m = 0; m < 4; m++)
			{
				vector<Point2d> newpoints = imagepointList[m];

				vector<Mat> samples;
				Mat sample(1, 3, CV_64FC1);
				Ptr<cv::ml::EM> gmm = Gmms[gm];
				double r = 0, g = 0, b = 0;
				double totaldist = 0;
				for (int i = 0; i < newpoints.size(); i++)
				{
					// Get the color of each channel
					double row = newpoints[i].y;
					double col = newpoints[i].x;
					Vec3b rgb = frame.at<Vec3b>(row, col);
					b = (rgb[2]);
					g = (rgb[1]);
					r = (rgb[0]);

					// Put pixels in sample data
					sample.at<double>(0, 0) = static_cast<double>(r);
					sample.at<double>(0, 1) = static_cast<double>(g);
					sample.at<double>(0, 2) = static_cast<double>(b);
					totaldist += gmm->predict2(sample, noArray())[0];
					//cout<<gmm->predict2(sample, noArray())[1]<<endl;
					samples.push_back(sample);
				}
				if (mostprob > abs(totaldist / samples.size()))
				{
					if (matches[m] != -1)
					{
						continue;
					}
					person = m;
					mostprob = abs(totaldist / samples.size());
				}
			}

			matches[person] = gm;
			cout << "bu kacinci gm" << gm << "bu kacinci insan" << person << endl;
		}

		for (int h = 0; h < 4; h++)
		{
			int gm = matches[h];
			cout << h << "this is the person I label" << h << endl;
			for (int r = 0; r < imagepointList[h].size(); r++)
			{
				double row = imagepointList[h][r].y;
				double col = imagepointList[h][r].x;
				Scalar c = color_tab[gm];
				result.at<Vec3b>(row, col)[0] = c[0];
				result.at<Vec3b>(row, col)[1] = c[1];
				result.at<Vec3b>(row, col)[2] = c[2];
			}
		}
		for (int j = 0; j < m_groundCoordinates.size(); j++)
		{
			int flag = labels.at<int>(j);

			m_visible_voxels[j]->color = color_tab[matches[flag]];
		}
		imshow("EM-Segmentation", result);


		// last centers

		frame_count++;


		// if uncomented, will do regular path tracing
		//directions_found = false;



		if (frame_count % 25 == 0) {

			if (directions_found) {

				for (int c = 0; c < last_centers.size(); c++) {

					int best_c = -1;
					float smallest_dist = FLT_MAX;

					Vec2f direction;


					Point2f ajustment(m_height, m_height);


					for (int nc = 0; nc < matches.size(); nc++) {
					
						Vec2f new_vec(centers[nc] - last_centers[c]);
						new_vec = normalize(new_vec);

						float cosine_dist = 1  - new_vec.dot(directions[c]); // normalized, magnitude should both be 1


						norm(centers[nc] - last_centers[c]);

						float distance = norm(centers[nc] - last_centers[c]);


						if (smallest_dist > (cosine_dist * distance * distance) ){
							
							direction = new_vec;
							smallest_dist = cosine_dist;
							best_c = nc;
						}
					}

					Scalar boop(100, 0, 0);

					cout << (last_centers[c] + ajustment) / 8 << " " << (centers[best_c] + ajustment) / 8 << endl;

					line(path_image, (last_centers[c] + ajustment) / 8, (centers[best_c] + ajustment) / 8, color_tab[c], 5, FILLED);

					last_centers[c] = centers[best_c];


				}
			}
			// get first direction or do regular path finding
			else {
			


				for (int c = 0; c < last_centers.size(); c++) {

					int gm = matches[c];

					Point2f ajustment(m_height, m_height);

					cout << (last_centers[gm] + ajustment) / 8 << " " << (centers[c] + ajustment) / 8 << endl;

					line(path_image, (last_centers[gm] + ajustment) / 8, (centers[c] + ajustment) / 8, color_tab[gm], 2, LINE_AA);

					last_centers[gm] = centers[c];

					directions[c] = normalize(Vec2f(centers[gm] - last_centers[c]));
					last_centers[gm] = centers[c];

					
				}

				directions_found = true;
			
			}
			imshow("Path", path_image);

		}

		if (frame_count == 1000) {
			
			imwrite("path.png", path_image);
		
		}


	}


} /* namespace nl_uu_science_gmt */
