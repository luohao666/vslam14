#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;

//2d-2d
const int mMaxIterations=8;
vector< vector<size_t> >  mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

void generateRandomSets(const std::vector< DMatch > &matches);
void Normalize(const vector<KeyPoint> &vKeys, const vector<Point2f> &NPoints, Mat &T);//归一化到相同尺度

Mat computeH21(const vector<Point2f> &vP1,const vector<Point2f> &vP2);//SVD求解
void findH21(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,Mat &H21);//ransac
float checkH(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, 
    const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,float sigma);//卡方检验

Mat computeF21(const vector<Point2f> &vP1,const vector<Point2f> &vP2);//SVD求解
void findF21(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,Mat &F21);//RANSAC
float checkF(const cv::Mat &F21, vector<bool> &vbMatchesInliers, 
    const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,float sigma);//卡方检验


//2d-3d
void triangulation_dlt(const KeyPoint &keypoints_1,const KeyPoint &keypoints_2,const Mat &P1,const Mat &P2,Mat &x3D);
void triangulation_dlt(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,
    const Mat &P1,const Mat &P2,vector<Mat> &vx3D);
void triangulation_dlt(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,
    const Mat &P1,const Mat &P2,Mat &x3Ds);


//3d-2d
void PnPSolver_DLT(const vector<Point2f> &vP2d,const vector<Point3f> &vP3d,const Mat &K,Mat &R,Mat &t);//SVD求解//局部最小值

//3d-3d
void ICPSovler_DLT(const vector<Point3f> &vP3d1,const vector<Point3f> &vP3d2,Mat &R,Mat &t);//SVD求解

void generateRandomSets(const std::vector< DMatch > &matches)
{
    srand((unsigned)time(NULL));

    const int N=(int)matches.size();
    // 新建一个容器vAllIndices，生成0到N-1的数作为特征点的索引
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    for(int it=0; it<mMaxIterations; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            // 产生0到N-1的随机数
            int randi = rand()%(vAvailableIndices.size()-1);
            // idx表示哪一个索引对应的特征点被选中
            int idx = vAvailableIndices[randi];

            mvSets[it][j] = idx;

            // randi对应的索引已经被选过了，从容器中删除
            // randi对应的索引用最后一个元素替换，并删掉最后一个元素
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }
    cout<<mvSets.size()<<endl;
    cout<<mvSets[0].size()<<endl;
    
}

//均值为0，绝对矩为1
void Normalize(const vector<KeyPoint> &vKeys, vector<Point2f> &NPoints, Mat &T)
{
    const int N=vKeys.size();
    NPoints.resize(N);

    float meanX=0.0;
    float meanY=0.0;

    for(int i=0;i<N;i++)
    {
        meanX+=vKeys[i].pt.x;
        meanY+=vKeys[i].pt.y;
    }
    meanX/=N;
    meanY/=N;

    //均值为0，绝对矩为1
    float meanDevX=0.0;
    float meanDevY=0.0;
    for(int i=0;i<N;i++)
    {
        NPoints[i].x=vKeys[i].pt.x-meanX;
        NPoints[i].y=vKeys[i].pt.y-meanY;
        meanDevX+=fabs(NPoints[i].x);
        meanDevY+=fabs(NPoints[i].y);
    }
    meanDevX/=N;
    meanDevY/=N;

    float sX=1.0/meanDevX;
    float sY=1.0/meanDevY;

    for(int i=0;i<N;i++)
    {
        NPoints[i].x*=sX;
        NPoints[i].y*=sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

Mat computeH21(const vector<Point2f> &vP1,const vector<Point2f> &vP2)
{
    //Ah=0,SVD求解
    //A=2N*9,h=1*9
    const int N = vP1.size();
    cv::Mat A(2*N,9,CV_32F); // 2N*9

    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    cv::Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    return vt.row(8).reshape(0, 3); // v的最后一列
}

float checkH(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, 
    const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,float sigma)
{
    const int N=(int) matches.size();
    //symmetric transfor error
    const float h11=H21.at<float>(0,0);
    const float h12=H21.at<float>(0,1);
    const float h13=H21.at<float>(0,2);
    const float h21=H21.at<float>(1,0);
    const float h22=H21.at<float>(1,1);
    const float h23=H21.at<float>(1,2);
    const float h31=H21.at<float>(2,0);
    const float h32=H21.at<float>(2,1);
    const float h33=H21.at<float>(2,2);

    const float h11inv=H12.at<float>(0,0);
    const float h12inv=H12.at<float>(0,1);
    const float h13inv=H12.at<float>(0,2);
    const float h21inv=H12.at<float>(1,0);
    const float h22inv=H12.at<float>(1,1);
    const float h23inv=H12.at<float>(1,2);
    const float h31inv=H12.at<float>(2,0);
    const float h32inv=H12.at<float>(2,1);
    const float h33inv=H12.at<float>(2,2);

    float score=0;
     // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 5.991;
    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);
    vbMatchesInliers.resize(N);

    for(int i=0;i<N;i++)
    {
        bool bIn = true;

        //get pix
        const float u1=keypoints_1[matches[i].queryIdx].pt.x;
        const float v1=keypoints_1[matches[i].trainIdx].pt.y;
        const float u2=keypoints_2[matches[i].queryIdx].pt.x;
        const float v2=keypoints_2[matches[i].trainIdx].pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|
        // |v1| = |h21inv h22inv h23inv||v2|
        // |1 |   |h31inv h32inv h33inv||1 |
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        // 根据方差归一化误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // 将图像1中的特征点单应到图像2中
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }
    return score;
}

void findH21(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,Mat &H21)
{
    cout<<"find H21"<<endl;
    
    const int N=(int)matches.size();

    //normalize
    vector<Point2f> vPn1;
    vector<Point2f> vPn2;
    Mat T1;
    Mat T2;
    Normalize(keypoints_1,vPn1,T1);
    Normalize(keypoints_2,vPn2,T2);
    cv::Mat T2inv = T2.inv();

    //iter vars
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8);
    cv::Mat H21i, H12i;
    vector<bool> vbCurrentInliers(N,false);
    float mSigma=1.0;

    float currentScore=0.0;
    float score=0.0;

    cout<<mvSets.size()<<endl;
    cout<<mvSets[0].size()<<endl;
    
    for(int i=0;i<mMaxIterations;i++)
    {
        //get random set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[i][j];

            // vPn1i和vPn2i为匹配的特征点对的坐标
            vPn1i[j] = vPn1[matches[idx].queryIdx];
            vPn2i[j] = vPn2[matches[idx].trainIdx];
        }

        cv::Mat Hn = computeH21(vPn1i,vPn2i);

        // 恢复原始的均值和尺度
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        // 利用重投影误差为当次RANSAC的结果评分
        currentScore = checkH(H21i, H12i, vbCurrentInliers,keypoints_1,keypoints_2,matches,mSigma);

        // 得到最优的vbMatchesInliers与score
        if(currentScore>score)
        {
            H21 = H21i.clone();
            score = currentScore;
        }    
    }
}

Mat computeF21(const vector<Point2f> &vP1,const vector<Point2f> &vP2)
{
    //Ah=0,SVD求解
    //A=2N*9,h=1*9
    const int N=vP1.size();
    cv::Mat A(N,9,CV_32F); // N*9
    for(int i=0;i<N;i++)
    {
        const float u1=vP1[i].x;
        const float v1=vP1[i].y;
        const float u2=vP2[i].x;
        const float v2=vP2[i].y;

        A.at<float>(i,0)=u1*u2;
        A.at<float>(i,1)=v1*u2;
        A.at<float>(i,2)=u2;
        A.at<float>(i,3)=u1*v2;
        A.at<float>(i,4)=v1*v2;
        A.at<float>(i,5)=v2;
        A.at<float>(i,6)=u1;
        A.at<float>(i,7)=v1;
        A.at<float>(i,8)=1;
    }
    Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    Mat Fred=vt.row(8).reshape(0,3);
    cv::SVDecomp(Fred,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;//zhi=2的约束
    return u*cv::Mat::diag(w)*vt;
}

float checkF(const cv::Mat &F21, vector<bool> &vbMatchesInliers, 
    const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,float sigma)
{
    const int N=(int)matches.size();
    //point2line error
    const float f11=F21.at<float>(0,0);
    const float f12=F21.at<float>(0,1);
    const float f13=F21.at<float>(0,2);
    const float f21=F21.at<float>(1,0);
    const float f22=F21.at<float>(1,1);
    const float f23=F21.at<float>(1,2);
    const float f31=F21.at<float>(2,0);
    const float f32=F21.at<float>(2,1);
    const float f33=F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);
    for(int i=0;i<N;i++)
    {
        bool bIn = true;

        //get pix
        const float u1=keypoints_1[matches[i].queryIdx].pt.x;
        const float v1=keypoints_1[matches[i].trainIdx].pt.y;
        const float u2=keypoints_2[matches[i].queryIdx].pt.x;
        const float v2=keypoints_2[matches[i].trainIdx].pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // F21x1可以算出x1在图像中x2对应的线l
        const float a2=f11*u1+f12*v1+f13;
        const float b2=f21*u1+f22*v1+f23;
        const float c2=f31*u1+f32*v1+f33;

        // x2应该在l这条线上:x2点乘l = 0 
        const float num2 = a2*u2+b2*v2+c2;
        const float squareDist1 = num2*num2/(a2*a2+b2*b2); // 点到线的几何距离 的平方
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;
        const float squareDist2 = num1*num1/(a1*a1+b1*b1);
        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

void findF21(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,Mat &F21)
{
    cout << "findF21" << endl;
    const int N=(int) matches.size();

    //normalize
    vector<Point2f> vPn1;
    vector<Point2f> vPn2;
    Mat T1;
    Mat T2;
    Normalize(keypoints_1,vPn1,T1);
    Normalize(keypoints_2,vPn2,T2);
    cv::Mat T2t = T2.t();

    //iter vars
    vector<Point2f> vPn1i(8);
    vector<Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);
    float mSigma=1.0;

    float currentScore=0.0;
    float score=0.0;

    for(int i=0;i<mMaxIterations;i++)
    {
        //get random set
        for(size_t j=0; j<8; j++)
        {
            int idx = mvSets[i][j];

            // vPn1i和vPn2i为匹配的特征点对的坐标
            vPn1i[j] = vPn1[matches[idx].queryIdx];
            vPn2i[j] = vPn2[matches[idx].trainIdx];
        }

        cv::Mat Hn = computeF21(vPn1i,vPn2i);

        // 恢复原始的均值和尺度
        F21i = T2t*Hn*T1;

        // 利用重投影误差为当次RANSAC的结果评分
        currentScore = checkF(F21i,vbCurrentInliers,keypoints_1,keypoints_2,matches,mSigma);

        // 得到最优的vbMatchesInliers与score
        if(currentScore>score)
        {
            F21 = F21i.clone();
            score = currentScore;
        }    
    }
}

//DLT triangulation
// Trianularization: 已知匹配特征点对{x x'} 和 各自相机矩阵{P P'}, 估计三维点 X
// x' = P'X  x = PX
// 它们都属于 x = aPX模型
//                         |X|
// |x|     |p1 p2  p3  p4 ||Y|     |x|    |--p0--||.|
// |y| = a |p5 p6  p7  p8 ||Z| ===>|y| = a|--p1--||X|
// |z|     |p9 p10 p11 p12||1|     |z|    |--p2--||.|
// 采用DLT的方法：x叉乘PX = 0
// |yp2 -  p1|     |0|
// |p0 -  xp2| X = |0|
// |xp1 - yp0|     |0|
// 两个点:
// |yp2   -  p1  |     |0|
// |p0    -  xp2 | X = |0| ===> AX = 0
// |y'p2' -  p1' |     |0|
// |p0'   - x'p2'|     |0|
// 变成程序中的形式：
// |xp2  - p0 |     |0|
// |yp2  - p1 | X = |0| ===> AX = 0
// |x'p2'- p0'|     |0|
// |y'p2'- p1'|     |0|
void triangulation_dlt(const KeyPoint &keypoints_1,const KeyPoint &keypoints_2,const Mat &P1,const Mat &P2,Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);
    
    float x1=keypoints_1.pt.x;
    float y1=keypoints_1.pt.y;
    float x2=keypoints_2.pt.x;
    float y2=keypoints_2.pt.y;

    A.row(0)=x1*P1.row(2)-P1.row(0);
    A.row(1)=y1*P1.row(2)-P1.row(1);
    A.row(2)=x2*P1.row(2)-P2.row(0);
    A.row(3)=y2*P1.row(2)-P2.row(1);

    Mat u,w,vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    x3D=vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void triangulation_dlt(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,
    const Mat &P1,const Mat &P2,vector<Mat> &vx3D)
{
    cv::Mat A(4,4,CV_32F);
    const int N=(int) matches.size();

    for(int i=0;i<N;i++)
    {
        //get pix
        const float u1=keypoints_1[matches[i].queryIdx].pt.x;
        const float v1=keypoints_1[matches[i].trainIdx].pt.y;
        const float u2=keypoints_2[matches[i].queryIdx].pt.x;
        const float v2=keypoints_2[matches[i].trainIdx].pt.y;

        A.row(0)=u1*P1.row(2)-P1.row(0);
        A.row(1)=v1*P1.row(2)-P1.row(1);
        A.row(2)=u2*P1.row(2)-P2.row(0);
        A.row(3)=v2*P1.row(2)-P2.row(1);

        Mat u,w,vt;
        cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

        Mat x3D=vt.row(3).t();
        x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
        vx3D.push_back(x3D);
    }
}

void triangulation_dlt(const vector<KeyPoint> &keypoints_1,const vector<KeyPoint> &keypoints_2,const vector< DMatch > &matches,
    const Mat &P1,const Mat &P2,Mat &x3Ds)
{
    cv::Mat A(4,4,CV_32F);
    const int N=(int) matches.size();
    x3Ds.create(4,N,CV_32F);

    for(int i=0;i<N;i++)
    {
        //get pix
        const float u1=keypoints_1[matches[i].queryIdx].pt.x;
        const float v1=keypoints_1[matches[i].trainIdx].pt.y;
        const float u2=keypoints_2[matches[i].queryIdx].pt.x;
        const float v2=keypoints_2[matches[i].trainIdx].pt.y;

        A.row(0)=u1*P1.row(2)-P1.row(0);
        A.row(1)=v1*P1.row(2)-P1.row(1);
        A.row(2)=u2*P1.row(2)-P2.row(0);
        A.row(3)=v2*P1.row(2)-P2.row(1);

        cv::Mat u,w,vt;
        cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
        Mat x3D;
        x3D = vt.row(3).t();
        //x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
        //cout << "x3D:" << x3D<<endl;
        x3Ds.col(i)=x3D;
    }
}

void PnPSolver_DLT(const vector<Point2f> &vP2d,const vector<Point3f> &vP3d,const Mat &K,Mat &R,Mat &t)
{
    const int N=vP2d.size();
    Mat A(2*N,12,CV_32F);

    const float fx=K.at<double>(0,0);
    const float fy=K.at<double>(1,1);
    const float cx=K.at<double>(0,2);
    const float cy=K.at<double>(1,2);

    for(int i=0;i<N;i++)
    {
        const float u=vP2d[i].x;
        const float v=vP2d[i].y;
        const float x=vP3d[i].x;
        const float y=vP3d[i].y;
        const float z=vP3d[i].z;

        A.at<float>(2*i,0)=fx*x;
        A.at<float>(2*i,1)=fx*y;
        A.at<float>(2*i,2)=fx*z;
        A.at<float>(2*i,3)=fx;
        A.at<float>(2*i,4)=0;
        A.at<float>(2*i,5)=0;
        A.at<float>(2*i,6)=0;
        A.at<float>(2*i,7)=0;
        A.at<float>(2*i,8)=cx*x-u*x;
        A.at<float>(2*i,9)=cx*y-u*y;
        A.at<float>(2*i,10)=cx*z-u*z;
        A.at<float>(2*i,11)=cx-u;

        A.at<float>(2*i+1,0)=0;
        A.at<float>(2*i+1,1)=0;
        A.at<float>(2*i+1,2)=0;
        A.at<float>(2*i+1,3)=0;
        A.at<float>(2*i+1,4)=fy*x;
        A.at<float>(2*i+1,5)=fy*y;
        A.at<float>(2*i+1,6)=fy*z;
        A.at<float>(2*i+1,7)=fy;
        A.at<float>(2*i+1,8)=cy*x-v*x;
        A.at<float>(2*i+1,9)=cy*y-v*y;
        A.at<float>(2*i+1,10)=cy*z-v*z;
        A.at<float>(2*i+1,11)=cy-v;
    }

    //SVD
    Mat u1,w1,vt1;
    cv::SVDecomp(A,w1,u1,vt1,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    Mat Tpre=vt1.row(11);
    cout<<vt1.row(11).reshape(0,3)<<endl;
    cv::Mat Rpre(3,3,CV_32F);
    Rpre.at<float>(0,0)=Tpre.at<float>(0,0);
    Rpre.at<float>(0,1)=Tpre.at<float>(0,1);
    Rpre.at<float>(0,2)=Tpre.at<float>(0,2);
    Rpre.at<float>(1,0)=Tpre.at<float>(0,4);
    Rpre.at<float>(1,1)=Tpre.at<float>(0,5);
    Rpre.at<float>(1,2)=Tpre.at<float>(0,6);
    Rpre.at<float>(2,0)=Tpre.at<float>(0,8);
    Rpre.at<float>(2,1)=Tpre.at<float>(0,9);
    Rpre.at<float>(2,2)=Tpre.at<float>(0,10);

    cout<<Tpre<<endl;
    cout<<Rpre<<endl;
    //SVD for better R
    Mat u2,w2,vt2;
    cv::SVDecomp(Rpre,w2,u2,vt2,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    //求解的结果没有尺度并且旋转矩阵性质无法保证
    R=u2*vt2;//opencv 已经转职了
    float beta = 1.0/((w2.at<float>(0,0)+w2.at<float>(1,0)+w2.at<float>(2,0))/3.0);

    cv::Mat tpre(3,1,CV_32F);
    tpre.at<float>(0,0)=Tpre.at<float>(0,3);
    tpre.at<float>(0,1)=Tpre.at<float>(0,7);
    tpre.at<float>(0,2)=Tpre.at<float>(0,11);
    cout<<w2<<endl;
    cout<<tpre<<endl;
    t=tpre*beta;

    //通过深度来判断符号
    int num_positive = 0;
    int num_negative = 0;
    for ( int i = 0; i < N; i ++ ) 
    {
        const float x=vP3d[i].x;
        const float y=vP3d[i].y;
        const float z=vP3d[i].z;
        //投影方程的深度
        float lambda = beta * ( x * Tpre.at<float>(0,8) + y* Tpre.at<float>(0,9) + z* Tpre.at<float>(0,10) + Tpre.at<float>(0,11));
        if ( lambda >= 0 )
        {
            num_positive ++;
        } 
        else 
        {
            num_negative ++;
        }
    }
    if ( num_positive < num_negative ) 
    {
        R = -R;
        t = -t;
    }
}

void ICPSovler_DLT(const vector<Point3f> &vP3d1,const vector<Point3f> &vP3d2,Mat &R,Mat &t)
{
    const int N=vP3d1.size();

    //1.获得去质心坐标
    Point3f p1(.0,.0,.0);
    Point3f p2(.0,.0,.0);
    for(int i=0;i<N;i++)
    {
        p1+=vP3d1[i];
        p2+=vP3d2[i];
    }
    p1=p1*(1.0/N);
    p2=p2*(1.0/N);
    cout<<p1<<endl;
    cout<<p2<<endl;
    //2.获得W
    Mat W(3,3,CV_32F);
    for(int i=0;i<N;i++)
    {
        Point3f d_p1=vP3d1[i]-p1;
        Point3f d_p2=vP3d2[i]-p2;
        W.at<float>(0,0) += d_p1.x*d_p2.x;
        W.at<float>(0,1) += d_p1.x*d_p2.y;
        W.at<float>(0,2) += d_p1.x*d_p2.z;
        W.at<float>(1,0) += d_p1.y*d_p2.x;
        W.at<float>(1,1) += d_p1.y*d_p2.y;
        W.at<float>(1,2) += d_p1.y*d_p2.z;
        W.at<float>(2,0) += d_p1.z*d_p2.x;
        W.at<float>(2,1) += d_p1.z*d_p2.y;
        W.at<float>(2,2) += d_p1.z*d_p2.z;
    }

    //SVD分解
    Mat u,w,vt;
    cv::SVDecomp(W,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cout <<u<<endl;
    cout <<vt<<endl;

    if(determinant(u) * determinant(vt) < 0)
	{
        for (int x = 0; x < 3; ++x)
        {
            u.at<float>(x,2) *= -1;
        }
	}

    R=u*vt;

    Mat p1_temp(3,1,CV_32F);
    Mat p2_temp(3,1,CV_32F);

    p1_temp.at<float>(0,0)=p1.x;
    p1_temp.at<float>(1,0)=p1.y;
    p1_temp.at<float>(2,0)=p1.z;
    p2_temp.at<float>(0,0)=p2.x;
    p2_temp.at<float>(1,0)=p2.y;
    p2_temp.at<float>(2,0)=p2.z;
    t=p1_temp-R*p2_temp;
}
