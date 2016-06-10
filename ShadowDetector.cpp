// teszt.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include "Math.h"
//#include <cv.h>
//#include <cxcore.h>
//#include <highgui.h>
#include <iostream>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv\cv.h"

using namespace cv;
using namespace std;

Mat sobel(Mat gray);
Mat canny(Mat src);


class color {
public:
	float y;
	float u;
	float v;
};

IplImage* HistrogramSyn(IplImage* img) {
	int x = img->width;
	int y = img->height;
	int step = img->widthStep;
	uchar* data = (uchar*)img->imageData;
	IplImage* NewImg = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
	uchar* newData = (uchar*)NewImg->imageData;
	int i, j;

	double min = 255;
	double max = 0;

	for (j = 0; j < y; j++) {
		for (i = 0; i < x; i++) {
			if (data[j * step + i] > max)
			{
				max = data[j * step + i];
			}
			if (data[j * step + i] < min)
			{
				min = data[j * step + i];
			}
		}
	}

	for (j = 0; j < y; j++) {
		for (i = 0; i < x; i++) {
			newData[j * step + i] = (int)(255.0 / (max - min)*(data[j * step + i]) - min);
		}
	}
	return NewImg;


}
IplImage* CreateHistogram(IplImage* img, bool exceptnull = false, bool tisztitas = false) {
	int imHeight = img->height;
	int imWidth = img->width;
	int channels = img->nChannels;
	int step = img->widthStep;
	uchar* data = (uchar*)img->imageData;
	IplImage* hist = cvCreateImage(cvSize(256, 200), 8, 1);
	int hImHeight = hist->height;
	int hImWidth = hist->width;
	int hChannels = hist->nChannels;
	int hStep = hist->widthStep;
	uchar* hData = (uchar*)hist->imageData;

	unsigned int histData[256];
	for (int i = 0;i < 256;i++) { histData[i] = 0; }
	for (int i = 0;i < imHeight;i++) {
		for (int j = 0;j < imWidth;j++) {
			histData[(int)data[i*step + j]]++;
		}
	}
	int max = 0;
	if (exceptnull) {
		for (int i = 1;i < 256;i++) { max = histData[i]>max ? histData[i] : max; }
	}
	else {
		for (int i = 0;i < 256;i++) { max = histData[i]>max ? histData[i] : max; }
	}
	if (tisztitas) {
		for (int i = 0;(i < 256) && (histData[i] != max);i++) {
			histData[i] = 0;
		}
	}
	for (int i = 0;i < 256;i++) { histData[i] = (int)(((float)histData[i] / (float)(max)) * 100); }

	for (int i = 0;i < hImHeight;i++) {
		for (int j = 0;j < hImWidth;j++) {
			hData[i*hStep + j] = 0;
			if (i>(4 * hImHeight / 5)) {
				hData[i*hStep + j] = j;
			}
			else {
				if (histData[j] >= ((4 * hImHeight / 5) - i)) {
					hData[i*hStep + j] = 200;
				}
			}

		}
	}

	return hist;
}

void blur(IplImage* image, int size) {

	IplImage* pad = cvCreateImage(cvSize(image->width + size * 2, image->height + size * 2), 8, image->nChannels);
	int imHeight, imWidth, imWidthpad, imHeightpad, pstep, step, channels;
	uchar *data, *pdata;

	imHeight = image->height;
	imWidth = image->width;
	step = image->widthStep;
	pstep = pad->widthStep;
	channels = image->nChannels;
	data = (uchar *)image->imageData;
	imHeightpad = pad->height;
	imWidthpad = pad->width;
	pdata = (uchar*)pad->imageData;

	//manualPad
	for (int i = 0;i<imHeightpad;i++) {
		for (int j = 0;j<imWidthpad;j++) {
			for (int k = 0;k<channels;k++) {
				if (((i - size) >= 0) && ((j - size) >= 0) && ((i - size)<imHeight) && ((j - size)<imWidth)) {
					pdata[i*pstep + j*channels + k] = data[(i - size)*step + (j - size)*channels + k];
				}
				else {
					pdata[i*pstep + j*channels + k] = 255;
				}
			}
		}
	}

	double sum;
	for (int k = 0;k<channels;k++) {
		for (int i = size;i<(imHeightpad - size);i++) {
			for (int j = size;j<(imWidthpad - size);j++) {
				sum = 0.;
				for (int ii = -size / 2;ii<size / 2 + 1;ii++) {
					for (int jj = -size / 2;jj<size / 2 + 1;jj++) {
						sum += (double)pdata[(i + ii)*pstep + (j + jj)*channels + k] / (double)(size*size);
					}
				}


				//std::cout<<"\n"<<sum;
				pdata[i*pstep + j*channels + k] = (int)sum;
			}
		}
	}

	//cvShowImage("padded", pad);

	//manualBlur
	for (int i = size;i<(imHeightpad - size);i++) {
		for (int j = size;j<(imWidthpad - size);j++) {
			for (int k = 0;k<channels;k++) {
				data[(i - (size))*step + (j - (size))*channels + k] = pdata[i*pstep + j*channels + k];
			}
		}
	}

	//cvShowImage("blured", image);

	cvReleaseImage(&pad);
}

IplImage* findShadow(char* filename) {
	IplImage* shadowDet = cvLoadImage(filename, 1);
	IplImage* originImage = cvLoadImage(filename, 1);
	int imHeight, imWidth, step, channels, stepm;
	uchar *data, *data2, *datam, *datag, *datah;
	imHeight = shadowDet->height;
	imWidth = shadowDet->width;
	step = shadowDet->widthStep;
	channels = shadowDet->nChannels;
	data = (uchar *)shadowDet->imageData;


	IplImage* edge = cvCreateImage(cvSize(shadowDet->width, shadowDet->height), 8, 1);
	IplImage* gray = cvCreateImage(cvSize(shadowDet->width, shadowDet->height), 8, 1);
	IplImage* blured = cvCreateImage(cvSize(shadowDet->width, shadowDet->height), 8, 3);
	IplImage* histo = cvCreateImage(cvSize(shadowDet->width, shadowDet->height), 8, 1);
	IplImage* mask = cvCreateImage(cvSize(shadowDet->width, shadowDet->height), 8, 1);

	cvCopy(shadowDet, blured);
	blur(blured, 9);
	cvCvtColor(shadowDet, gray, CV_BGR2GRAY);
	//cvCanny(gray, edge, 100, 200);
	data = (uchar *)shadowDet->imageData;
	data2 = (uchar *)blured->imageData;
	datah = (uchar *)histo->imageData;
	datam = (uchar *)mask->imageData;
	datag = (uchar *)gray->imageData;
	stepm = mask->widthStep;
	color* orig = new color[step*imHeight];

	for (int j = 0;j < imHeight;j++) {
		for (int i = 0;i < imWidth;i++) {
			datam[j * stepm + i] = 0;
		}
	}

	//Full
	double Faverage;
	int Fcount;
	int maxa, maxb;
	Faverage = 0.;
	Fcount = 0;
	int maxwindow = 31;
	double average;
	int count;

	for (int j = 0;j < imHeight;j++) {
		for (int i = 0;i < imWidth;i++) {
			orig[j * step + i].y = (float)data[j * step + i * channels + 0] * 0.114 + (float)data[j * step + i * channels + 1] * 0.587 + (float)data[j * step + i * channels + 2] * 0.299;
			orig[j * step + i].u = (float)data[j * step + i * channels + 0] * 0.436 + (float)data[j * step + i * channels + 1] * -0.289 + (float)data[j * step + i * channels + 2] * -0.147;
			orig[j * step + i].v = (float)data[j * step + i * channels + 0] * -0.1 + (float)data[j * step + i * channels + 1] * -0.515 + (float)data[j * step + i * channels + 2] * 0.615;

			datah[j * stepm + i] = orig[j * step + i].y;
		}
	}

	int a, b;
	IplImage* histogramSynz = HistrogramSyn(histo);
	uchar* dataForHist = (uchar*)histogramSynz->imageData;
	for (int j = 0;j < imHeight;j++) {
		for (int i = 0;i < imWidth;i++) {
			orig[j * step + i].y = dataForHist[j * stepm + i];
			Faverage += orig[j * step + i].y;
			Fcount++;
		}
	}
	Faverage /= Fcount;

	for (int window = 81;window>3;window -= 16) {
		maxa = maxb = window;
		for (int j = 0;j < imHeight;j++) {
			for (int i = 0;i < imWidth;i++) {
				for (a = 0;a < maxa;a++) {
					for (b = 0;b < maxb;b++) {
						average = 0.;
						count = 0;
					}
				}

				for (a = 0;a < maxa;a++) {
					for (b = 0;b < maxb;b++) {

						if ((j + a)<imHeight && (i + b)<imWidth) {
							if (datam[(j + a)*stepm + (i + b)] == 0) {
								average += orig[(j + a)*step + (i + b)].y;
								count++;
							}
						}

					}
				}
				average /= count;
				for (a = 0;a < maxa;a++) {
					for (b = 0;b < maxb;b++) {
						if ((j + a)<imHeight && (i + b)<imWidth) {
							if (Faverage*0.6>orig[(j + a)*step + (i + b)].y) {
								datam[(j + a)*stepm + (i + b)] = 255;
								//data[(j + a)*step + (i + b)*channels + 1] = 255;
							}
							else {
								if (average*0.2>orig[(j + a)*step + (i + b)].y) {
									if (datam[(j + a)*stepm + (i + b)] == 0) {
										datam[(j + a)*stepm + (i + b)] = 255;
										//data[(j + a)*step + (i + b)*channels + 2] = 255;
									}
								}
							}
						}
					}
				}

			}
		}



	}

	//cvSaveImage("shadowDectected.jpg", mask);
	cvShowImage("Shadow", mask);

	//cvSaveImage("edge.jpg", shadowDet);
	//cvShowImage("edge", shadowDet);


	Mat mat_mask = cvarrToMat(mask);

	//Edge detector
	/*
	GaussianBlur(mat_mask, mat_mask, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat edges;
	bool useCanny = false;
	if (useCanny) {
	edges = canny(mat_mask);
	}
	else {
	//Use Sobel filter and thresholding.
	edges = sobel(mat_mask);
	//Automatic thresholding
	//threshold(edges, edges, 0, 255, cv::THRESH_OTSU);
	//Manual thresholding
	threshold(edges, edges, 25, 255, cv::THRESH_BINARY);
	}

	imshow("Edge Detector", edges);
	*/

	//try to increase contrast
	Mat mat_image = cvarrToMat(originImage);
	//Mat mat_image_conv;
	//cvtColor(mat_image, mat_image_conv, CV_BGR2HLS_FULL);
	imshow("before?", mat_image);

	Mat new_image = Mat::zeros(mat_image.size(), mat_image.type());

	Mat channel[3];


	mat_image.convertTo(new_image, -1, 2.5, 0);
	cvtColor(new_image, new_image, CV_RGB2HLS);
	split(new_image, channel);
	imshow("channel:L", channel[1]);
	channel[1].convertTo(channel[1], -1, 1, 90);
	merge(channel, 3, new_image);
	new_image.copyTo(channel[1], mat_mask);
	imshow("new?", channel[1]);
	cvtColor(channel[1], channel[1], CV_HLS2RGB);
	imshow("newmask", channel[1]);
	//Mat channelZero[3];
	//split(channel[0], channelZero);
	//channel[0].convertTo(channelZero[0], -1, 1, 90);
	//merge(channelZero, 3, channel[0]);
	channel[1].copyTo(mat_image, mat_mask);
	imshow("now?", mat_image);

	cout << Faverage;


	cvWaitKey(0);
	cvDestroyAllWindows();

	return mask;
}

int _tmain(int argc, _TCHAR* argv[])
{

	findShadow("..\\Debug\\shadow.jpg");

	system("pause");
	return 0;
}

Mat sobel(Mat gray) {
	Mat edges;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Mat edges_x, edges_y;
	Mat abs_edges_x, abs_edges_y;

	Sobel(gray, edges_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(edges_x, abs_edges_x);
	Sobel(gray, edges_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(edges_y, abs_edges_y);
	addWeighted(abs_edges_x, 0.5, abs_edges_y, 0.5, 0, edges);

	return edges;
}

Mat canny(Mat src)
{
	Mat detected_edges;

	int edgeThresh = 1;
	int lowThreshold = 250;
	int highThreshold = 750;
	int kernel_size = 5;

	Canny(src, detected_edges, lowThreshold, highThreshold, kernel_size);

	return detected_edges;
}
