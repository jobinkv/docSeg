#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "lbp.hpp"
#include "histogram.hpp"

using namespace cv;
using namespace std;




Mat patchpos(Rect imageprp)
{
// this function will give the starting xy position of all aptch from the given image
	int p_size = imageprp.x;
	int nobk=0;
	nobk = floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size)) +floor( ((double) imageprp.height/p_size))+floor( ((double) imageprp.width/p_size))+1;
	
	cout<<" currect no p ="<<floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size))<<endl;

	cout<<" total p= "<<nobk<<endl;
	cout<<"height= "<<imageprp.height<<endl;
	cout<<"width= "<<imageprp.width<<endl;
	Mat codi_list(nobk,3, CV_32F,Scalar(0));
	int cnt=0;
	for (int i =0;i<imageprp.height-1-p_size;i=i+p_size)//Row
		{
		for (int j =0;j<imageprp.width-1-p_size;j=j+p_size)//Col
			{
			codi_list.at<float>(cnt,0)=cnt;
			codi_list.at<float>(cnt,1)=i;
			codi_list.at<float>(cnt,2)=j;
			cnt=cnt+1;
			}
		}
	//cout<<"1st "<<cnt<<endl;
	for (int i =0;i<imageprp.height-1-p_size;i=i+p_size)//Row
		{

		int j=imageprp.width-2-p_size; // -2 because the lbp function gives out image 2 row/col less than the original
		codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;
		}	
	//cout<<"2nd "<<cnt<<endl;
	for (int j =0;j<imageprp.width-1-p_size;j=j+p_size)//Row
		{

		int i=imageprp.height-2-p_size;
		codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;

		}

	//cout<<"3rd "<<cnt<<endl;
	codi_list.at<float>(cnt,0)=cnt;
	codi_list.at<float>(cnt,1)=imageprp.height-p_size-2;
	codi_list.at<float>(cnt,2)=imageprp.width-p_size-2;
	//cout<<"4th "<<cnt<<endl;
	return codi_list;

/*	usage of this function listofpatch = patchpos(imageprp);

Ploting all patches boundary on image.
	for (int i=0;i<listofpatch.rows;i++)
	{
	int psiz = listofpatch.at<float>(1,0)-listofpatch.at<float>(0,0);
	Rect roi(listofpatch.at<float>(i,0),listofpatch.at<float>(i,1),psiz,psiz);
	rectangle(image, roi, Scalar(255*rand(),255*rand(),255*rand()), 1, 8, 0 );	
	}
*/
}

//============***************** ctreating training features*****************////////
Mat crtTrainFet(Mat listofpatch,Mat locbp,Mat gt_img)
{
    /// Establish the number of bins
    	int histSize = 256;
    	float range[] = { 0, 256 } ; //the upper boundary is exclusive
    	const float* histRange = { range };
    	Mat hist;
    	bool uniform = true; bool accumulate = false;
	Mat fetr_lst, main_lst;
	Mat patch,label,channel[3],label_list(listofpatch.rows,2, CV_32F,Scalar(0)), lbl(1,1, CV_32F,Scalar(0));
	int psiz = listofpatch.at<float>(1,2)-listofpatch.at<float>(0,2);
	//cout<<"====="<<psiz<<endl;

	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
 		patch= locbp(ross);
//	--------creating labels for each patch-----------------------
		label = gt_img(ross);
		 split(label, channel);
		double sum_b = cv::sum( channel[0] )[0];// text region 		=>label 1
		double sum_g = cv::sum( channel[1] )[0];// graphics region	=>label 2
		double sum_r = cv::sum( channel[2] )[0];// background region	=>label 3
//
		if (sum_b>=sum_g and sum_b>=sum_r)
			lbl.at<float>(0,0) = 1;
		if (sum_g>=sum_b and sum_g>=sum_r)
			lbl.at<float>(0,0) = 2;
		if (sum_r>=sum_b and sum_r>=sum_g)
			lbl.at<float>(0,0) = 3;

   		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		
		hconcat(lbl, hist.t(), fetr_lst);
		if (i==0)
		{
			main_lst = fetr_lst.clone();
			continue;
		}
		vconcat(main_lst,fetr_lst,main_lst);

	}
	//cout<<"list = "<<main_lst<<endl;
	return main_lst;
}
Mat makeLbpImg(Mat image, int p_size)
{
// ------------------Taking lbp from the image--------------------------------------
    // initial values
    	int radius = 1;
    	int neighbors = 8;
	Mat dst,locbp; 
    // matrices used
    

    // just to switch between possible lbp operators
/*  vector<string> lbp_names;
    lbp_names.push_back("Extended LBP"); // 0
    lbp_names.push_back("Fixed Sampling LBP"); // 1
    lbp_names.push_back("Variance-based LBP"); // 2
    int lbp_operator=0;*/
// pre-processing part
    	cvtColor(image, dst, CV_BGR2GRAY);
    	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    // -----------------------different lbp functions-----------------------------------
    //lbp::ELBP(dst, locbp, radius, neighbors); // use the extended operator
    	lbp::OLBP(dst, locbp); // use the original operator
    	//lbp::VARLBP(dst, locbp, radius, neighbors);
    // a simple min-max norm will do the job...
    	normalize(locbp, locbp, 0, 255, NORM_MINMAX, CV_8UC1);
//-----------------------geting patch list ------------------------------------
	return locbp;

}

///////////////=======+++++++++++++ MAIN PROGRAM +++++++++++=============//////////////
int main( int argc, char** argv )
{
	if( argc != 3)
    	{
     		cout <<" Usage: train original_image ground_truth_img" << endl;
     		return -1;
    	}
	
    	Mat image,gt_img;
    	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file
	gt_img = imread(argv[2], CV_LOAD_IMAGE_COLOR);   // Read the file

	//cout<<"size of gt = "<<gt_img.size()<<endl;
	//cout<<"size of org = "<<image.size()<<endl;
	if( gt_img.size() != image.size())
    	{
     		cout <<"ERROr : Image dimentios of the given images are not matching" << endl;
     		return -1;
    	}

// ------------------Taking lbp from the image--------------------------------------
/*     // initial values
    	int radius = 1;
    	int neighbors = 8;
	Mat dst,locbp; 
    // matrices used
    

    // just to switch between possible lbp operators
 vector<string> lbp_names;
    lbp_names.push_back("Extended LBP"); // 0
    lbp_names.push_back("Fixed Sampling LBP"); // 1
    lbp_names.push_back("Variance-based LBP"); // 2
    int lbp_operator=0;
// pre-processing part
    	cvtColor(image, dst, CV_BGR2GRAY);
    	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    // -----------------------different lbp functions-----------------------------------
    //lbp::ELBP(dst, locbp, radius, neighbors); // use the extended operator
    	lbp::OLBP(dst, locbp); // use the original operator
    	//lbp::VARLBP(dst, locbp, radius, neighbors);
    // a simple min-max norm will do the job...
    	normalize(locbp, locbp, 0, 255, NORM_MINMAX, CV_8UC1);*/
//-----------------------geting patch list ------------------------------------
	int p_size = 70;
	Rect imageprp(p_size,0,image.cols,image.rows); 
	Mat listofpatch;
	// calling patch listing function---
	listofpatch = patchpos(imageprp);
	// making of lbp image
	Mat locbp = makeLbpImg(image,p_size);
	// creating feature and labels
	Mat train_feture = crtTrainFet(listofpatch,locbp,gt_img);
	cout<<"size 4 trining"<<train_feture.size()<<endl;
	//cout<<"list of patch = "<<listofpatch<<endl;
/*
	for (int i=0;i<listofpatch.rows;i++)
	{
	//cout<<"iiiiiii    ="<<codi_list.at<float>(0,0)-codi_list.at<float>(1,1)<<endl;
	int psiz = listofpatch.at<float>(1,0)-listofpatch.at<float>(0,0);
	Rect roi(listofpatch.at<float>(i,0),listofpatch.at<float>(i,1),psiz,psiz);
	rectangle(image, roi, Scalar(255*rand(),255*rand(),255*rand()), 1, 8, 0 );	
	}
	//imwrite( "./image.png", image );
*/
//---------------------------------end getting patch list-------------------------

//---------------------------------Compute the histograms-------------------------
/*
    /// Establish the number of bins
    	int histSize = 256;
    	float range[] = { 0, 256 } ; //the upper boundary is exclusive
    	const float* histRange = { range };
    	Mat hist;
    	bool uniform = true; bool accumulate = false;
	Mat fetr_lst, main_lst;
	Mat patch,label,channel[3],label_list(listofpatch.rows,2, CV_32F,Scalar(0)), lbl(1,1, CV_32F,Scalar(0));
	int psiz = listofpatch.at<float>(1,2)-listofpatch.at<float>(0,2);
	//cout<<"====="<<psiz<<endl;

	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
 		patch= locbp(ross);
//	--------creating labels for each patch-----------------------
		label = gt_img(ross);
		 split(label, channel);
		double sum_b = cv::sum( channel[0] )[0];// text region 		=>label 1
		double sum_g = cv::sum( channel[1] )[0];// graphics region	=>label 2
		double sum_r = cv::sum( channel[2] )[0];// background region	=>label 3
//
		if (sum_b>=sum_g and sum_b>=sum_r)
			lbl.at<float>(0,0) = 1;
		if (sum_g>=sum_b and sum_g>=sum_r)
			lbl.at<float>(0,0) = 2;
		if (sum_r>=sum_b and sum_r>=sum_g)
			lbl.at<float>(0,0) = 3;
		//label_list.at<float>(i,0)=i;
		//label_list.at<float>(i,1)=lbl;		
		

/*		cout<<"sum_b = "<<sum_b<<endl;
		cout<<"sum_g = "<<sum_g<<endl;
		cout<<"sum_r = "<<sum_r<<endl;
		cout<<"--------------label of above = "<<lbl<<endl;
		namedWindow("chanel",CV_WINDOW_NORMAL);
		imshow( "chanel", channel[2] ); 
		cout<<"size = "<<locbp.size()<<endl;
*//*
   		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		
		hconcat(lbl, hist.t(), fetr_lst);
		if (i==0)
		{
			main_lst = fetr_lst.clone();
			continue;
		}
		vconcat(main_lst,fetr_lst,main_lst);
		//fetr_lst.at<float>(i,0)=

		//cout <<main_lst<<endl;
		//cout <<"iiiiiiii"<<i<<endl;
		//fetr_lst.at<float>(i,0)=listofpatch.at<float>(i,0);
		//fetr_lst.at<float>(i,1:256)=hist;
	}
	cout<<"list = "<<main_lst<<endl;
	//cout <<listofpatch<<endl;
	

	for (int i=0;i<listofpatch.rows;i++)
	{
	Rect roi(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz+2,psiz+2);
	rectangle(image, roi, Scalar(255*rand(),255*rand(),255*rand()), 1, 1, 0 );	
	}


    //================================= Draw histogram=============================

    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(hist,hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    for( int i = 1; i < histSize; i++ )
    {
       line( histImage, Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                     Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                     Scalar( 0, 0, 255), 2, 8, 0  );
		     cout<<"hist value = "<<cvRound(hist.at<float>(i-1))<<", i= "<<i<<endl;
       
    }

    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );
    //============================drow hist end====================================
*/    //lbp::spatial_histogram();


//      // windows
    namedWindow("original",CV_WINDOW_NORMAL);
    imshow( "original", image ); 
    namedWindow( "lbp window", WINDOW_NORMAL );// Create a window for display.
    imshow("lbp window", locbp );                   // Show our image inside it.



    waitKey(0);                                  // Wait for a keystroke in the window
    return 0;
}
