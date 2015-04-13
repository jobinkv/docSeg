#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include "lbp.hpp"
#include "histogram.hpp"
#include "GCoptimization.h"
#include "LinkedBlockList.h"
#include <errno.h>

// yaaaahhhooooooooooooooo//
#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>



using namespace cv;
using namespace std;

int p_size = 50;
// FUNCTION FROM ALPHA EXPANSION

struct ForDataFn{
	int numLab;
	int *data;
};


int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

int dataFn(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;

	return( myData->data[p*numLab+l] );
}



////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually
// grid neighborhood structure is assumed
//
Mat GridGraph_Individually(int num_labels,Mat img,int lambda)
{

	int height=img.rows;//HEIGHT
	int width=img.cols;//width
	int num_pixels=height*width;

	int *result = new int[num_pixels];   // stores result of optimization
	int rw;
	int col;
	Mat  opimage =img.clone();
//image is transformed int 1 drow in row major order

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually


		for ( int i = 0; i < num_pixels; i++ )
		{
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;

			}	
			else
			{
			rw=(i+1)/width;
			col=((i+1)%width)-1;
			}

			int blue=img.at<cv::Vec3b>(rw,col)[0];
			int green=img.at<cv::Vec3b>(rw,col)[1];
			int red=img.at<cv::Vec3b>(rw,col)[2];



			for (int l = 0; l < num_labels; l++ )
			{
				if(l==0)
					 gc->setDataCost(i,l,(255-blue)/*+red+green*/);
			 	if(l==1)
			 		gc->setDataCost(i,l,(255-green)/*+red+blue*/);
		 		if(l==2)
		 			gc->setDataCost(i,l,(255-red)/*+blue+green*/);

			}
		}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{

				if(l1==l2)
				//int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
				gc->setSmoothCost(l1,l2,0);

				else

				gc->setSmoothCost(l1,l2,lambda);


			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());


		

		for ( int  i = 0; i < num_pixels; i++ )
		{
			result[i] = gc->whatLabel(i);
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;
			}
			else
			{
				rw=(i+1)/width;
				col=((i+1)%width)-1;
			}
			if(result[i]==0) //sky
			{
		//cout<<"label 0 \n";
				opimage.at<cv::Vec3b>(rw,col)[0]=255;//blue
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=0;
			}
			if(result[i]==1) // grass
			{
			opimage.at<cv::Vec3b>(rw,col)[0]=0;
			opimage.at<cv::Vec3b>(rw,col)[1]=255;
			opimage.at<cv::Vec3b>(rw,col)[2]=0;
			//cout<<"label 1 \n";
			}
			if(result[i]==2) //third object
			{
				opimage.at<cv::Vec3b>(rw,col)[0]=0;
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=255;//red
			}
		}





		//imwrite( "outputimage.png", opimage );


		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}
	delete [] result;
	return opimage;
}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);
		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void GridGraph_DfnSfn(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// set up the needed data to pass to function for the data costs
		ForDataFn toFn;
		toFn.data = data;
		toFn.numLab = num_labels;

		gc->setDataCost(&dataFn,&toFn);

		// smoothness comes from function pointer
		gc->setSmoothCost(&smoothFn);

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// Uses spatially varying smoothness terms. That is
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1
void GridGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;

	// next set up spatially varying arrays V and H

	int *V = new int[num_pixels];
	int *H = new int[num_pixels];


	for ( int i = 0; i < num_pixels; i++ ){
		H[i] = i+(i+1)%3;
		V[i] = i*(i+width)%7;
	}


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCostVH(smooth,V,H);
		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually"
//
void GeneralGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ )
				gc->setNeighbors(x+y*width,x-1+y*width);

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ )
				gc->setNeighbors(x+y*width,x+(y-1)*width);

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually". Uses spatially varying terms. Namely
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

void GeneralGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ ){
				int p1 = x-1+y*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1+p2);
			}

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ ){
				int p1 = x+(y-1)*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1*p2);
			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}

////////////////////////////////////////////



struct ThreeMatstr 
{
	Mat background;
	Mat figure;
	Mat text;
} ;

Mat patchpos(Rect imageprp)
{
// this function will give the starting xy position of all aptch from the given image
	//int p_size = imageprp.x;
	int nobk=0;
	nobk = floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size)) +floor( ((double) imageprp.height/p_size))+floor( ((double) imageprp.width/p_size))+1;
	
	// cout<<" currect no p ="<<floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size))<<endl;

	// cout<<" total p= "<<nobk<<endl;
	// cout<<"height= "<<imageprp.height<<endl;
	// cout<<"width= "<<imageprp.width<<endl;
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
ThreeMatstr crtTrainFet(Mat& listofpatch,Mat& locbp,Mat& gt_img)
{

	ThreeMatstr outFeture;
	Mat initial(256,1,CV_32F,Scalar(0));
	//cout<<"text size = "<<initial.size()<<endl;
	outFeture.background = initial.clone();
	outFeture.figure	= initial.clone();
	outFeture.text		= initial.clone();
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
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//calculating label
		double sum_b = cv::sum( channel[0] )[0];// text region 			=>label 0
		double sum_g = cv::sum( channel[1] )[0];// figure region		=>label 1
		double sum_r = cv::sum( channel[2] )[0];// background region	=>label 2
		//Assigning to curresponding lablel
		if (sum_b>=sum_g and sum_b>=sum_r)
		{
			hconcat(outFeture.text,hist,outFeture.text);
			// cout<<"text size = "<<outFeture.text.size()<<endl;
		}
		if (sum_g>=sum_b and sum_g>=sum_r)

		{
			hconcat(outFeture.figure,hist,outFeture.figure);
			// cout<<"figure size = "<<outFeture.figure.size()<<endl;
		}
		if (sum_r>=sum_b and sum_r>=sum_g)
			
		{
			hconcat(outFeture.background,hist,outFeture.background);
			// cout<<"background size = "<<outFeture.background.size()<<endl;
		}
		
		
		//hconcat(lbl, hist.t(), fetr_lst);

		//fetr_lst = hist.clone();
		//if (i==0)
		//{
		//	outFeture.background = hist.clone();
		//	continue;
		//}


	}
	
	return outFeture;
}
Mat makeLbpImg(Mat& image)
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
//@@@@@@@@@@@@@@@@@@@@@ read all file from the folder@@@@@@@@@@@@@@@@@@@@@@


int getdir (string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) 
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) 
	{
		if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 )
		{
    		//cout<<dirp->d_name<<"hohohhhh"<<endl;
			files.push_back(string(dirp->d_name));	
		}

	}
	closedir(dp);
	return 0;
}
//!!!!!!!!!!!!!!!!!!!!Cleaning feature vector!!!!!!!!!!!++++++++++++@@@@@@@@@@@@@
Mat RmZeroCols(Mat input)
{
	Mat outMat,temp;
	outMat = input.col(10);
	//temp = train_feture.text.col(10);
	//cout<<"dimentios= "<<input.cols<<endl;
	for (int i = 0; i < input.cols; ++i)
	{
		temp = input.col(i);
	// cout<<"dimentios= "<<train_feture.text.col(1)<<endl;
		//cout<<"dimentios= "<<sum(temp)[0]<<endl;
		if(sum(temp)[0]!=0)
		{
			hconcat(outMat,temp,outMat);
		}
		//else
			//cout<<"Skipped ="<<i<<endl;
		//cout<<"dimentios= "<<cleaned.size()<<""<<i<<endl;
	}
	return outMat;

}

ThreeMatstr cleanFet(ThreeMatstr train_feture)
{
	ThreeMatstr clean_feture;
	clean_feture.text = RmZeroCols(train_feture.text);
	clean_feture.figure = RmZeroCols(train_feture.figure);
	clean_feture.background = RmZeroCols(train_feture.background);
	// make equal dimension
		// cout<<"dimentios of clean_feture.text= "<<clean_feture.text.cols<<endl;
		// cout<<"dimentios of clean_feture.figure= "<<clean_feture.figure.cols<<endl;
		// cout<<"dimentios of clean_feture.background= "<<clean_feture.background.cols<<endl;
		// int mmin = std::min(clean_feture.text.cols, clean_feture.figure.cols);
		// mmin = std::min(clean_feture.background.cols, mmin);
		// cout<<"mim = "<<mmin<<endl;
	//cout<<"dimentios last= "<<clean_feture.text.size()<<endl;
	return clean_feture;

}
/* Function maximum definition */
/* x, y and z are parameters */
int maximum(int x, int y, int z) {
	int max = x; /* assume x is the largest */

	if (y > max) { /* if y is larger than max, assign y to max */
	max = y;
	} /* end if */

	if (z > max) { /* if z is larger than max, assign z to max */
	max = z;
	} /* end if */

	return max; /* max is the largest value */
} /* end function maximum */
//############### SVM training function ###################
	void TrainTheModel(string org_folder,string gt_folder)
	{
		ThreeMatstr finalFet;
		Mat initial(256,1,CV_32F,Scalar(0));
	//cout<<"text size = "<<initial.size()<<endl;
		finalFet.background = initial.clone();
		finalFet.figure		= initial.clone();
		finalFet.text		= initial.clone();

		vector<string> files = vector<string>();
		getdir(gt_folder,files);
//	final feture list
		Mat final_lst;
		for (unsigned int i = 0;i < files.size();i++) 
		{
			// if (i==1)// looop for sample run
			// {
			// 	break;
			// }
			string gt_path =  gt_folder + string("/") + files[i] ;
			string org_path = org_folder + string("/") + files[i] ;

			Mat image,gt_img;
	    	image = imread(org_path, CV_LOAD_IMAGE_COLOR);   // Reading original image
			gt_img = imread(gt_path, CV_LOAD_IMAGE_COLOR);   // Reading gt image
			if( gt_img.size() != image.size())
			{
				cout <<"ERROr : Image dimentios of the given images "<<files[i]<<" are not matching" << endl;
				continue;
			}
        	// cout << gt_path << endl;
        	// cout << org_path << endl;
			cout<<"Now running "<<files[i]<<endl;
			Rect imageprp(p_size,0,image.cols,image.rows); 
		// making of lbp image
			Mat locbp = makeLbpImg(image);
		// calling patch listing function---
			Mat listofpatch = patchpos(imageprp);
		// creating feature and labels
			ThreeMatstr train_feture = crtTrainFet(listofpatch,locbp,gt_img);

		// concatinating the out put features
			hconcat(finalFet.text,train_feture.text,finalFet.text);
			hconcat(finalFet.figure,train_feture.figure,finalFet.figure);
			hconcat(finalFet.background,train_feture.background,finalFet.background);
		// cout<<"str ="<<finalFet.text.row(3)<<endl;
		}
	// cleaning feature data
		// ThreeMatstr clean_feture = cleanFet(finalFet);
		ThreeMatstr clean_feture = finalFet;
		int maxx = maximum(clean_feture.text.cols,clean_feture.figure.cols,clean_feture.background.cols);
		cout<<"maximum val= "<<maxx<<endl;

		cout<<"text size= "<<clean_feture.text.cols<<endl;
		cout<<"figure size= "<<clean_feture.figure.cols<<endl;
		cout<<"background size= "<<clean_feture.background.cols<<endl;


		Mat trainData;
		hconcat(clean_feture.text,clean_feture.figure,trainData);
		hconcat(trainData,clean_feture.background,trainData);
	// making of labels
		Mat labels;
		Mat lab_text(clean_feture.text.cols,1,CV_32F,Scalar(0));
		Mat lab_figure(clean_feture.figure.cols,1,CV_32F,Scalar(1));
		Mat lab_background(clean_feture.background.cols,1,CV_32F,Scalar(2));
		vconcat(lab_text,lab_figure,labels);
		vconcat(labels,lab_background,labels);

	//cout<<"labels= "<<labels.size()<<endl;
	//cout<<"lab_figure= "<<lab_figure.size()<<endl;
		trainData = trainData.t();
	//cout<<"trainData= "<<trainData.size()<<endl;
	// train SVM classifier
	//Set up the support vector machines parameters 
    // CvSVMParams params;
    // params.svm_type    = SVM::C_SVC;
    // params.C           = 0.1;
    // params.kernel_type = SVM::RBF;
    // params.term_crit   = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);

    // CvMat *m=cvCreateMat(3,1,CV_32FC1);
    // cvmSet(m,0,0,.5);
    // cvmSet(m,1,0,1);
    // cvmSet(m,2,0,1);

		Mat R = Mat(1, 3, CV_32FC1);
		R.at<float>(0,0)=.166;
		R.at<float>(0,1)=.666;
		R.at<float>(0,2)=1;

		cout<< "R vector"<<R<<endl;

		CvMat weights = R;

		CvSVMParams  param = CvSVMParams();
		param.svm_type = CvSVM::C_SVC;
		param.kernel_type = CvSVM::RBF;
	param.degree = 0; // for  poly
	param.gamma = 20; // for poly/rbf/sigmoid
	param.coef0 = 0; // for  poly/sigmoid
	param.C = 7; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR  and  CV_SVM_NU_SVR
	param.nu = 0.0; // for  CV_SVM_NU_SVC, CV_SVM_ONE_CLASS , and  CV_SVM_NU_SVR
	param.p = 0.0; // for CV_SVM_EPS_SVR
	param.class_weights = &weights;//[(.6, 0.3,0.1);//NULL;//for CV_SVM_C_SVC
	param.term_crit.type = CV_TERMCRIT_ITER;	 //| CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-6;
	CvParamGrid gammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA);
	CvParamGrid pGrid;
	CvParamGrid nuGrid,degreeGrid;
	//gammaGrid.step=0;

	cout << "Starting training process" << endl;
	cout<<"trainData= "<<trainData.rows<<endl;
	cout<<"labels = "<<labels.size()<<endl;
	CvSVM svm;
	svm.train_auto(trainData, labels, Mat(), Mat(),param, 4, 
		CvSVM::get_default_grid(CvSVM::C), gammaGrid, pGrid, nuGrid, CvSVM::get_default_grid(CvSVM::COEF), degreeGrid, true);
	svm.save("classifier_w_s1.xml");

    //svm.save("classifier.xml");
    // svm.load("classifier.xml");
	cout << "Finished training process" << endl;
}
//--------SVM testing patch creator---------------------//
Mat crtTestFet(Mat& image)
{

	Mat locbp = makeLbpImg(image);
	// calling patch listing function---
	Rect imageprp(p_size,0,image.cols,image.rows); 
	Mat listofpatch = patchpos(imageprp);

	Mat outImage(locbp.rows,locbp.cols, CV_8UC3, Scalar(0,0,0));
    /// Establish the number of bins
	int histSize = 256;
	float range[] = { 0, 256 } ; //the upper boundary is exclusive
	const float* histRange = { range };
	Mat hist;
	bool uniform = true; bool accumulate = false;
	Mat patch;
	int psiz = listofpatch.at<float>(1,2)-listofpatch.at<float>(0,2);
	//cout<<"====="<<psiz<<endl;
	CvSVM svm;
	svm.load("classifier_w_s1.xml");

	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		// Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
		patch= locbp(ross);
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//hconcat(outFeture.feature,hist,outFeture.feature);
		float response = svm.predict(hist);
		if (response==0)
			rectangle(outImage, ross, Scalar(255,0,0), -1, 8, 0 );
		if (response==1)
			rectangle(outImage, ross, Scalar(0,255,0), -1, 8, 0 );
		if (response==2)
			rectangle(outImage, ross, Scalar(0,0,255), -1, 8, 0 );

		//cout<<"predicted values = "<<response<<endl;
		//hconcat(outFeture.rectBox,ross,outFeture.rectBox);
	}
	
	return outImage;
}

///////////////=======+++++++++++++ MAIN PROGRAM +++++++++++=============//////////////
Mat docLayotSeg(Mat image)
{

	Mat enerfyMin;
	Mat outImage = crtTestFet(image);
//////////////////////// Alpha expansion //////////////////////
	int num_labels = 3;
	int lambada=.45*255;
	Mat downSmp;
	resize(outImage, downSmp, Size(),(double)1/p_size, (double)1/p_size, INTER_NEAREST);
		// smoothness and data costs are set up one by one, individually
	enerfyMin = GridGraph_Individually(num_labels,downSmp,lambada);
	// resize as the size of the original image
	resize(enerfyMin, enerfyMin, image.size(), INTER_NEAREST);
	return enerfyMin;

}

int main( int argc, char** argv )
{
	if( argc == 1)
	{
		cout <<" Usage of the software pakage!" <<"\n"<<"\n"<< endl;
		cout <<" ==============For Training the layout model==========" <<"\n"<< endl;
		cout <<" <Binary> <train> <original image folder> <ground truth image folder>" <<"\n"<<"\n"<< endl;
		cout <<" ==============For Testing the layout model===========" <<"\n"<< endl;
		cout <<" <Binary> <test> <testing image file>" << "\n"<<"\n"<<endl;
		cout <<" =====================================================" <<"\n"<< endl;
		// cout <<" Usage: train original_image ground_truth_img" << endl;
		// cout <<" Usage: train original_image ground_truth_img" << endl;
		return -1;
	}

	string mode = string(argv[1]);
	
	if (mode =="test")
	{
		cout<<"tesing started"<<endl;
		Mat image = imread(argv[2], CV_LOAD_IMAGE_COLOR); 
		Mat layout = docLayotSeg(image);
		namedWindow("outImage",WINDOW_NORMAL);
		imshow( "outImage", layout ); 
   		namedWindow( "lbp window", WINDOW_NORMAL );// Create a window for display.
    	imshow("lbp window", image );                   // Show our image inside it.
    	waitKey(0);
	}
	else if(mode =="train")
	{
		cout<<"training  started"<<endl;
		string org_folder = string(argv[2]);
		string gt_folder = string(argv[3]);
		// it will create a classifier.xml file
		TrainTheModel(org_folder,gt_folder);
	}

    return 0;
}
