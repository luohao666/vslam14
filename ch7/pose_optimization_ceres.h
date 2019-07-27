#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

//ceres
#include <ceres/ceres.h>
#include <ceres/rotation.h>

//sophus
#include "sophus/so3.h"
#include "sophus/se3.h"

#include <chrono>


using namespace std;
using namespace cv;

void PnPBACeres(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t);
void PnPBACeresJaco(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t);
void ICPBACeres(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t);
void ICPBACeresJaco(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t);

struct PnP_cost_function_defined
{
    PnP_cost_function_defined(Point3f pt3d,Point2f pt2d):_pt3d(pt3d),_pt2d(pt2d){}

    template<typename T>
    bool operator() (
        const T* const cere_r,     // 模型参数，有3维
        const T* const cere_t,     // 模型参数，有3维
        T* residual ) const     // 残差
    {
        T pt3d1[3];//旋转前
        T pt3d2[3];//旋转后
        pt3d1[0]=T(_pt3d.x);
        pt3d1[1]=T(_pt3d.y);
        pt3d1[2]=T(_pt3d.z);

        ceres::AngleAxisRotatePoint(cere_r,pt3d1,pt3d2);//利用轴角来旋转
        pt3d2[0]+=cere_t[0];
        pt3d2[1]+=cere_t[1];
        pt3d2[2]+=cere_t[2];

        const T x=pt3d2[0]/pt3d2[2];
        const T y=pt3d2[1]/pt3d2[2];

        T u=520.9*x+325.1;//Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
        T v=521.0*y+249.7;


        T u0= T(_pt2d.x);
        T v0= T(_pt2d.y);
        //
        residual[0]=u-u0;
        residual[1]=v-v0;
        return true;
    }

    const Point3f _pt3d;
    const Point2f _pt2d;
};

//3d-2d
void PnPBACeres(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t)
{
    Mat r;
    cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    double cere_r[3];
    double cere_t[3];

    //cere_r[0]=r.at<double>(0,0);
    //cere_r[1]=r.at<double>(1,0);
    //cere_r[2]=r.at<double>(2,0);

    cere_r[0]=0;
    cere_r[1]=1;
    cere_r[2]=2;

    cere_t[0]=t.at<double>(0,0);
    cere_t[1]=t.at<double>(1,0);
    cere_t[2]=t.at<double>(2,0);

    for ( auto a:cere_r ) cout<<a<<endl;
    for ( auto a:cere_t ) cout<<a<<endl;

    //构建最小二乘问题
    const int N=pts3d.size();
    ceres::Problem problem;
    for(int i=0;i<N;i++)
    {
        //使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        ceres::CostFunction *cost=new 
            ceres::AutoDiffCostFunction<PnP_cost_function_defined,2,3,3>(
                new PnP_cost_function_defined(pts3d[i],pts2d[i]));

        problem.AddResidualBlock(cost,NULL,cere_r,cere_t);
    }

    //配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated r = ";
    for ( auto a:cere_r ) cout<<a<<endl;

    cout<<"estimated t = ";
    for ( auto a:cere_t ) cout<<a<<endl;

    Mat r2(3,1,CV_32F);
    Mat R2;
    r2.at<float>(0,0)=cere_r[0];
    r2.at<float>(1,0)=cere_r[1];
    r2.at<float>(2,0)=cere_r[2];

    cv::Rodrigues ( r2, R2 ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"estimated R = ";
    cout<<R2<<endl;
}

class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const { return 6; }
    virtual int LocalSize() const { return 6; }
};

bool SE3Parameterization::ComputeJacobian(const double *x, double *jacobian) const {
    ceres::MatrixRef(jacobian, 6, 6) = ceres::Matrix::Identity(6, 6);
    return true;
}

bool SE3Parameterization::Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const {
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(x);
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_lie(delta);

    Sophus::SE3 T = Sophus::SE3::exp(lie);
    Sophus::SE3 delta_T = Sophus::SE3::exp(delta_lie);
    Eigen::Matrix<double, 6, 1> x_plus_delta_lie = (delta_T * T).log();

    for(int i = 0; i < 6; ++i) x_plus_delta[i] = x_plus_delta_lie(i, 0);

    return true;

}

class PnP_cost_function: public ceres::SizedCostFunction<2,6>
{
    public:
        PnP_cost_function(Point3f pt3d,Point2f pt2d):_pt3d(pt3d),_pt2d(pt2d)
        {
        }

        virtual bool Evaluate (double const *const *pose, double *residual, double **jacobians) const
        {
            double pt3d1[3];//旋转前
            double pt3d2[3];//旋转后
            pt3d1[0]=double(_pt3d.x);
            pt3d1[1]=double(_pt3d.y);
            pt3d1[2]=double(_pt3d.z);

            double cere_r[3];
            cere_r[0]=pose[0][0];
            cere_r[1]=pose[0][1];
            cere_r[2]=pose[0][2];

            ceres::AngleAxisRotatePoint(cere_r,pt3d1,pt3d2);//利用轴角来旋转
            pt3d2[0]+=pose[0][3];
            pt3d2[1]+=pose[0][4];
            pt3d2[2]+=pose[0][5];

            double x=pt3d2[0]/pt3d2[2];
            double y=pt3d2[1]/pt3d2[2];

            double u=520.9*x+325.1;//Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
            double v=521.0*y+249.7;

            double u0= double(_pt2d.x);
            double v0= double(_pt2d.y);
            //
            residual[0]=u-u0;
            residual[1]=v-v0;

            if(jacobians)
            {
                double x_trans=pt3d2[0];
                double y_trans=pt3d2[1];
                double z_trans=pt3d2[2];

                Eigen::Matrix<double,2,6> _jacobianOplusXj;
                _jacobianOplusXj(0,0) =  _fx*x_trans*y_trans/(z_trans*z_trans);
                _jacobianOplusXj(0,1) =  (_fx+_fx*x_trans*x_trans/(z_trans*z_trans))*(-1);
                _jacobianOplusXj(0,2) =  _fx*y_trans/z_trans;   
                _jacobianOplusXj(0,3) = (_fx/z_trans)*(-1);
                _jacobianOplusXj(0,4) = 0;
                _jacobianOplusXj(0,5) = _fx*x_trans/(z_trans*z_trans);

                _jacobianOplusXj(1,0) = _fy+_fy*(y_trans*y_trans)/(z_trans*z_trans);
                _jacobianOplusXj(1,1) = _fy*x_trans*y_trans/(z_trans*z_trans)*(-1);
                _jacobianOplusXj(1,2) = _fy*x_trans/z_trans*(-1);
                _jacobianOplusXj(1,3) = 0;
                _jacobianOplusXj(1,4) = _fy/z_trans*(-1);
                _jacobianOplusXj(1,5) = _fy*y_trans/(z_trans*z_trans);

                jacobians[0][0] = _jacobianOplusXj(0,0);
                jacobians[0][1] = _jacobianOplusXj(0,1);
                jacobians[0][2] = _jacobianOplusXj(0,2);
                jacobians[0][3] = _jacobianOplusXj(0,3);
                jacobians[0][4] = _jacobianOplusXj(0,4);
                jacobians[0][5] = _jacobianOplusXj(0,5);

                jacobians[0][6] = _jacobianOplusXj(1,0);
                jacobians[0][7] = _jacobianOplusXj(1,1);
                jacobians[0][8] = _jacobianOplusXj(1,2);
                jacobians[0][9] = _jacobianOplusXj(1,3);
                jacobians[0][10] = _jacobianOplusXj(1,4);
                jacobians[0][11] = _jacobianOplusXj(1,5);

            }
            

            return true;
        }
    private:
        const Point3f _pt3d;
        const Point2f _pt2d;
        const double _fx=520.9;
        const double _fy=521.0;
        const double _cx=325.1;
        const double _cy=249.7;
};

//3d-2d
void PnPBACeresJaco(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t)
{
    Mat r;
    cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    double pose[6];
    pose[0]=r.at<double>(0,0);
    pose[1]=r.at<double>(1,0);
    pose[2]=r.at<double>(2,0);

//    pose[0]=0;
//    pose[1]=1;
//    pose[2]=2;

    pose[3]=t.at<double>(0,0);
    pose[4]=t.at<double>(1,0);
    pose[5]=t.at<double>(2,0);

    for ( auto a:pose ) cout<<a<<endl;

    //构建最小二乘问题
    const int N=pts3d.size();
    ceres::Problem problem;
    for(int i=0;i<N;i++)
    {
        //使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        ceres::CostFunction *cost=new PnP_cost_function(pts3d[i],pts2d[i]);

        problem.AddResidualBlock(cost,NULL,pose);
    }       

    //problem.SetParameterization(pose, new SE3Parameterization());

    //配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated r = "<<endl;
    cout<<"estimated t = "<<endl;

    for ( auto a:pose ) cout<<a<<endl;

    Mat r2(3,1,CV_32F);
    Mat R2;
    r2.at<float>(0,0)=pose[0];
    r2.at<float>(1,0)=pose[1];
    r2.at<float>(2,0)=pose[2];

    cv::Rodrigues ( r2, R2 ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"estimated R = ";
    cout<<R2<<endl;

}

struct ICP_cost_function_defined
{
    ICP_cost_function_defined(Point3f pt1,Point3f pt2):_pt1(pt1),_pt2(pt2){}

    template<typename T>
    bool operator ()(
        const T* const cere_r,
        const T* const cere_t,
        T* residual) const
    {
        T p1[3];//转换前
        T p2[3];//转换后
        p1[0]=T(_pt2.x);
        p1[1]=T(_pt2.y);
        p1[2]=T(_pt2.z);

        ceres::AngleAxisRotatePoint(cere_r,p1,p2);//旋转
        p2[0]+=cere_t[0];//平移
        p2[1]+=cere_t[1];
        p2[2]+=cere_t[2];

        residual[0]=p2[0]-T(_pt1.x);
        residual[1]=p2[1]-T(_pt1.y);
        residual[2]=p2[2]-T(_pt1.z);

        return true;
    }

    Point3f _pt1;
    Point3f _pt2;
};

void ICPBACeres(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t)
{
    Mat r;
    cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    double cere_r[3];
    double cere_t[3];

    //cere_r[0]=r.at<double>(0,0);
    //cere_r[1]=r.at<double>(1,0);
    //cere_r[2]=r.at<double>(2,0);

    cere_r[0]=0;
    cere_r[1]=1;
    cere_r[2]=2;

    cere_t[0]=t.at<double>(0,0);
    cere_t[1]=t.at<double>(1,0);
    cere_t[2]=t.at<double>(2,0);

    //构建最小二乘问题
    ceres::Problem problem;
    const int N=pts3d1.size(); 
    for(int i=0;i<N;i++)
    {
        //使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        ceres::CostFunction *cost=new 
            ceres::AutoDiffCostFunction<ICP_cost_function_defined,3,3,3>(
                new ICP_cost_function_defined(pts3d1[i],pts3d2[i]));

        problem.AddResidualBlock(cost,NULL,cere_r,cere_t);
    }

    //配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated r = ";
    for ( auto a:cere_r ) cout<<a<<endl;

    cout<<"estimated t = ";
    for ( auto a:cere_t ) cout<<a<<endl;

    Mat r2(3,1,CV_32F);
    Mat R2;
    r2.at<float>(0,0)=cere_r[0];
    r2.at<float>(1,0)=cere_r[1];
    r2.at<float>(2,0)=cere_r[2];

    cv::Rodrigues ( r2, R2 ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"estimated R = ";
    cout<<R2<<endl;
}

class ICP_cost_function: public ceres::SizedCostFunction<3,6>
{
    public:
        ICP_cost_function(Point3f pt3d1,Point3f pt3d2):_pt3d1(pt3d1),_pt3d2(pt3d2)
        {
        }

        virtual bool Evaluate (double const *const *pose, double *residual, double **jacobians) const
        {
             
            double pt1[3];//旋转前
            double pt2[3];//旋转后
            pt1[0]=double(_pt3d2.x);
            pt1[1]=double(_pt3d2.y);
            pt1[2]=double(_pt3d2.z);

            double cere_r[3];
            cere_r[0]=pose[0][0];
            cere_r[1]=pose[0][1];
            cere_r[2]=pose[0][2];
            ceres::AngleAxisRotatePoint(cere_r,pt1,pt2);//利用轴角来旋转

            pt2[0]+=pose[0][3];
            pt2[1]+=pose[0][4];
            pt2[2]+=pose[0][5];

            //
            double pt0[3];
            pt0[0]=double(_pt3d1.x);
            pt0[1]=double(_pt3d1.y);
            pt0[2]=double(_pt3d1.z);

            residual[0]=pt0[0]-pt2[0];
            residual[1]=pt0[1]-pt2[1];
            residual[2]=pt0[2]-pt2[2];
        
            if(jacobians)
            {
                double x_trans=pt2[0];
                double y_trans=pt2[1];
                double z_trans=pt2[2];

                Eigen::Matrix<double,3,6> _jacobianOplusXj;
                
                _jacobianOplusXj(0,0) = 0;
                _jacobianOplusXj(0,1) = -z_trans;
                _jacobianOplusXj(0,2) = y_trans;
                _jacobianOplusXj(0,3) = -1;
                _jacobianOplusXj(0,4) = 0;
                _jacobianOplusXj(0,5) = 0;

                _jacobianOplusXj(1,0) = z_trans;
                _jacobianOplusXj(1,1) = 0;
                _jacobianOplusXj(1,2) = -x_trans;
                _jacobianOplusXj(1,3) = 0;
                _jacobianOplusXj(1,4) = -1;
                _jacobianOplusXj(1,5) = 0;

                _jacobianOplusXj(2,0) = -y_trans;
                _jacobianOplusXj(2,1) = x_trans;
                _jacobianOplusXj(2,2) = 0;
                _jacobianOplusXj(2,3) = 0;
                _jacobianOplusXj(2,4) = 0;
                _jacobianOplusXj(2,5) = -1;
                
                jacobians[0][0] = _jacobianOplusXj(0,0);
                jacobians[0][1] = _jacobianOplusXj(0,1);
                jacobians[0][2] = _jacobianOplusXj(0,2);
                jacobians[0][3] = _jacobianOplusXj(0,3);
                jacobians[0][4] = _jacobianOplusXj(0,4);
                jacobians[0][5] = _jacobianOplusXj(0,5);

                jacobians[0][6] = _jacobianOplusXj(1,0);
                jacobians[0][7] = _jacobianOplusXj(1,1);
                jacobians[0][8] = _jacobianOplusXj(1,2);
                jacobians[0][9] = _jacobianOplusXj(1,3);
                jacobians[0][10] = _jacobianOplusXj(1,4);
                jacobians[0][11] = _jacobianOplusXj(1,5);

                jacobians[0][12] = _jacobianOplusXj(2,0);
                jacobians[0][13] = _jacobianOplusXj(2,1);
                jacobians[0][14] = _jacobianOplusXj(2,2);
                jacobians[0][15] = _jacobianOplusXj(2,3);
                jacobians[0][16] = _jacobianOplusXj(2,4);
                jacobians[0][17] = _jacobianOplusXj(2,5);
            }
    /*        
    Eigen::Map<const Eigen::Matrix<double, 6, 1>> lie(pose[0]);
    Sophus::SE3 T = Sophus::SE3::exp(lie);

    //std::cout << T.matrix3x4() << std::endl;
    Eigen::Vector3d Pi(double(_pt3d2.x),double(_pt3d2.y),double(_pt3d2.z));
    Eigen::Vector3d Pj(double(_pt3d1.x),double(_pt3d1.y),double(_pt3d1.z));
    auto Pj_ = T * Pi;
    Eigen::Vector3d err = Pj - Pj_;

    residual[0] = err(0);
    residual[1] = err(1);
    residual[2] = err(2);

    Eigen::Matrix<double, 3, 6> Jac = Eigen::Matrix<double, 3, 6>::Zero();
    Jac.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    Jac.block<3, 3>(0, 3) = Sophus::SO3::hat(Pj_);
    int k = 0;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 6; ++j) {
            if(k >= 18)
                return false;
            if(jacobians) {
                if(jacobians[0])
                    jacobians[0][k] = Jac(i, j);
            }
            k++;
        }
    }

    //printf("jacobian ok!\n");
*/
            return true;
        }
    private:
        const Point3f _pt3d1;
        const Point3f _pt3d2;
};

void ICPBACeresJaco(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t)
{

    Mat r;
    cv::Rodrigues ( R, r ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    double pose[6];

    //pose[0]=double(r.at<float>(0,0));
    //pose[1]=double(r.at<float>(1,0));
    //pose[2]=double(r.at<float>(2,0));

    pose[0]=0;
    pose[1]=1;
    pose[2]=2;

    pose[3]=double(t.at<float>(0,0));
    pose[4]=double(t.at<float>(1,0));
    pose[5]=double(t.at<float>(2,0));

    for ( auto a:pose ) cout<<a<<endl;

    //构建最小二乘问题
    ceres::Problem problem;
    const int N=pts3d1.size(); 
    for(int i=0;i<N;i++)
    {
        //使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
        ceres::CostFunction *cost=new ICP_cost_function(pts3d1[i],pts3d2[i]);

        problem.AddResidualBlock(cost,NULL,pose);
    }

    //problem.SetParameterization(pose, new SE3Parameterization());

    //配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout
    //options.minimizer_type = ceres::TRUST_REGION;
    //options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    //options.trust_region_strategy_type = ceres::DOGLEG;
    //options.minimizer_progress_to_stdout = true;
    //options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // 输出结果
    
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated r = "<<endl;
    cout<<"estimated t = "<<endl;
    for ( auto a:pose ) cout<<a<<endl;

    Mat r2(3,1,CV_32F);
    Mat R2;
    r2.at<float>(0,0)=pose[0];
    r2.at<float>(1,0)=pose[1];
    r2.at<float>(2,0)=pose[2];

    cv::Rodrigues ( r2, R2 ); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout<<"estimated R = ";
    cout<<R2<<endl;
    
//   	Eigen::Map<Eigen::Matrix<double, 6, 1> > se3lie(pose);
//   	std::cout << "esitmate = \n" << Sophus::SE3::exp(se3lie).matrix()<< std::endl;
}