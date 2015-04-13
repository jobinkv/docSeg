#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include "lbp.hpp"
#include "histogram.hpp"
#include <errno.h>


#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>



using namespace cv;
using namespace std;

int p_size = 50;

struct ThreeMatstr 
{
	Mat background;
	Mat figure;
	Mat text;
} ;

Mat patchpos(Rect imageprp)
{
// this function will give the starting xy position of all aptch from the given image
	int p_size = imageprp.x;
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
Mat makeLbpImg(Mat& image, int p_size)
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
			if (i==10)// looop for sample run
			{
				break;
			}
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
		int p_size = 50;
		Rect imageprp(p_size,0,image.cols,image.rows); 
		// making of lbp image
		Mat locbp = makeLbpImg(image,p_size);
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
	ThreeMatstr clean_feture = cleanFet(finalFet);
	Mat trainData;
	hconcat(clean_feture.text,clean_feture.figure,trainData);
	hconcat(trainData,clean_feture.background,trainData);
	// making of labels
	Mat labels;
	Mat lab_text(clean_feture.text.cols,1,CV_32F,Scalar(1));
	Mat lab_figure(clean_feture.figure.cols,1,CV_32F,Scalar(2));
	Mat lab_background(clean_feture.background.cols,1,CV_32F,Scalar(3));
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
	R.at<float>(0,0)=.5;
	R.at<float>(0,1)=.3;
	R.at<float>(0,2)=.2;

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
    cout<<"trainData= "<<trainData.size()<<endl;
    cout<<"labels = "<<labels.size()<<endl;
    CvSVM svm;
    svm.train_auto(trainData, labels, Mat(), Mat(),param, 4, 
    	CvSVM::get_default_grid(CvSVM::C), gammaGrid, pGrid, nuGrid, CvSVM::get_default_grid(CvSVM::COEF), degreeGrid, true);
    svm.save("classifier.xml");

    // svm.save("classifier.xml");
    // svm.load("classifier.xml");
    cout << "Finished training process" << endl;
}
//--------SVM testing patch creator---------------------//
Mat crtTestFet(Mat& listofpatch,Mat& locbp)
{

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
	svm.load("classifier.xml");

	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
		patch= locbp(ross);
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//hconcat(outFeture.feature,hist,outFeture.feature);
		float response = svm.predict(hist);
		if (response==1)
			rectangle(outImage, ross, Scalar(255,0,0), -1, 8, 0 );
		if (response==2)
			rectangle(outImage, ross, Scalar(0,255,0), -1, 8, 0 );
		if (response==3)
			rectangle(outImage, ross, Scalar(0,0,255), -1, 8, 0 );

		//cout<<"predicted values = "<<response<<endl;
		//hconcat(outFeture.rectBox,ross,outFeture.rectBox);
	}
	
	return outImage;
}

///////////////=======+++++++++++++ MAIN PROGRAM +++++++++++=============//////////////
int main( int argc, char** argv )
{
	if( argc != 4)
	{
		cout <<" Usage: train original_image ground_truth_img" << endl;
		return -1;
	}
	

	// string org_folder = string(argv[3]);
	// string gt_folder = string(argv[2]);
	// // it will create a classifier.xml file
	// TrainTheModel(org_folder,gt_folder);

	// svm testing module
	// input image
	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
	
	// making of lbp image
	Mat locbp = makeLbpImg(image,p_size);
	// calling patch listing function---
	Rect imageprp(p_size,0,image.cols,image.rows); 
	Mat listofpatch = patchpos(imageprp);
	// svm  testing
	Mat outImage = crtTestFet(listofpatch,locbp);




	//svm.load("classifier.xml");



//      // windows
    namedWindow("outImage",WINDOW_NORMAL);
    imshow( "outImage", outImage ); 
   namedWindow( "lbp window", WINDOW_NORMAL );// Create a window for display.
    imshow("lbp window", image );                   // Show our image inside it.



    waitKey(0);                                  // Wait for a keystroke in the window
    return 0;
}
