#include <iostream>
#include<opencv2/opencv.hpp>
#include<vector>

//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

//g2o
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <g2o/types/sba/types_six_dof_expmap.h>

#include <chrono>

using namespace std;
using namespace cv;

void PnPBAG2O(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t);
void PnPBA2G2O(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d1,const vector<Point2f>& pts2d2,const Mat& K,Mat& R,Mat& t);

void ICPBAPoseOnlyG2O(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t);
void ICPBAG2O(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t);

//3d-2d
void PnPBAG2O(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d,const Mat& K,Mat& R,Mat& t)
{
    //1.初始化g2o:线性求解器->矩阵块求解器->优化算法->优化器
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); //
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    //3.添加节点,id,estimate,marginal,add2optimizer
    //pose2
    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
    R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
    R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
    R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);
    Eigen::Vector3d t_v(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,t_v));
    optimizer.addVertex(pose);

    //landmarks
    int index=1;
    for(const Point3f p:pts3d)
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //2.相机参数
    g2o::CameraParameters* Camrea = 
        new g2o::CameraParameters(K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0);
    Camrea->setId(0);
    optimizer.addParameter(Camrea);  

    //3.添加误差边:id,vertex,measure
    index=1;
    for(const Point2f p:pts2d)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ));
        edge->setVertex(1,pose);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    //4.开始优化
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
    
}

void PnPBA2G2O(const vector<Point3f>& pts3d,const vector<Point2f>& pts2d1,const vector<Point2f>& pts2d2,const Mat& K,Mat& R,Mat& t)
{
    //1.初始化g2o:线性求解器->矩阵块求解器->优化算法->优化器
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); //
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    //3.添加节点,id,estimate,marginal,add2optimizer
    //pose1
    g2o::VertexSE3Expmap* pose1=new g2o::VertexSE3Expmap();
    pose1->setId(0);
    pose1->setEstimate(g2o::SE3Quat());
    optimizer.addVertex(pose1);

    //pose2
    g2o::VertexSE3Expmap* pose2=new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
    R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
    R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
    R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);
    Eigen::Vector3d t_v(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
    pose2->setId(1);
    pose2->setEstimate(g2o::SE3Quat(R_mat,t_v));
    optimizer.addVertex(pose2);

    //landmarks
    int index=2;
    for(const Point3f p:pts3d)
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //2.相机参数
    g2o::CameraParameters* Camrea = 
        new g2o::CameraParameters(K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0);
    Camrea->setId(0);
    optimizer.addParameter(Camrea);  

    //3.添加误差边:id,vertex,measure
    index=2;
    for(const Point2f p:pts2d2)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ));
        edge->setVertex(1,pose2);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    //pose1 相关边
    index=2;
    for(const Point2f p:pts2d1)
    {
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ));
        edge->setVertex(1,pose1);
        edge->setMeasurement(Eigen::Vector2d(p.x,p.y));
        edge->setParameterId(0,0);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    //4.开始优化
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
    cout<<endl<<"after optimization:"<<endl;
    cout<<"T1="<<endl<<Eigen::Isometry3d ( pose1->estimate() ).matrix() <<endl;
    cout<<"T2="<<endl<<Eigen::Isometry3d ( pose2->estimate() ).matrix() <<endl;
}

//3d-3d
//当深度图已经确定时，我们没有必要去优化地图点
//节点表示要优化的变量
//因此我们可以设计一个一元边来解决ICP问题
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>//观测类型大小，类型，顶点类型
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly( const Eigen::Vector3d& point ) : _point(point) {}

    bool read(std::istream& is){}

    bool write(std::ostream& os) const{}

    virtual void computeError()  
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        _error = _measurement - pose->estimate().map( _point );
    }

    virtual void linearizeOplus()
    {
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[0] );
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x=xyz_trans[0];
        double y=xyz_trans[1];
        double z=xyz_trans[2];

        _jacobianOplusXi(0,0) = 0;
        _jacobianOplusXi(0,1) = -z;
        _jacobianOplusXi(0,2) = y;
        _jacobianOplusXi(0,3) = -1;
        _jacobianOplusXi(0,4) = 0;
        _jacobianOplusXi(0,5) = 0;

        _jacobianOplusXi(1,0) = z;
        _jacobianOplusXi(1,1) = 0;
        _jacobianOplusXi(1,2) = -x;
        _jacobianOplusXi(1,3) = 0;
        _jacobianOplusXi(1,4) = -1;
        _jacobianOplusXi(1,5) = 0;

        _jacobianOplusXi(2,0) = -y;
        _jacobianOplusXi(2,1) = x;
        _jacobianOplusXi(2,2) = 0;
        _jacobianOplusXi(2,3) = 0;
        _jacobianOplusXi(2,4) = 0;
        _jacobianOplusXi(2,5) = -1;
    }
    protected:
        Eigen::Vector3d _point;
};

void ICPBAPoseOnlyG2O(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t)
{
    //1.g2o初始化:线性求解器->矩阵块求解器->
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); //多态多种方式
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); //
    Block* block_ptr=new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* sovler = new g2o::OptimizationAlgorithmLevenberg(block_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(sovler);

    //2.顶点
    Eigen::Matrix3d R_mat(Eigen::Matrix3d::Identity());
    Eigen::Vector3d t_v(0,0,0);
    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,t_v));
    optimizer.addVertex(pose);

    //3.边
    const int N=pts3d1.size();
    int index=1;
    for(int i=0;i<N;i++)
    {
        EdgeProjectXYZRGBDPoseOnly* edge=
            new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts3d2[i].x, pts3d2[i].y, pts3d2[i].z));
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement( Eigen::Vector3d(
            pts3d1[i].x, pts3d1[i].y, pts3d1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        index++;
    }

    //4.开始优化
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}

//如果非要优化地图点，那么要构建一个二元边
class EdgeProjectXYZRGBD : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>//观测类型大小，类型，顶点类型
{
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBD(){}

    bool read(std::istream& is){}

    bool write(std::ostream& os) const{}

    virtual void computeError()  
    {
        const g2o::VertexSBAPointXYZ* point = static_cast<const g2o::VertexSBAPointXYZ*> ( _vertices[0] );
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
        _error = _measurement - pose->estimate().map(point->estimate());
    }

    virtual void linearizeOplus()
    {
        const g2o::VertexSBAPointXYZ* point = static_cast<const g2o::VertexSBAPointXYZ*> ( _vertices[0] );
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> ( _vertices[1] );
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(point->estimate());
        double x=xyz_trans[0];
        double y=xyz_trans[1];
        double z=xyz_trans[2];

        //空间点的雅克比矩阵
        _jacobianOplusXi = -T.rotation().toRotationMatrix();

        _jacobianOplusXj(0,0) = 0;
        _jacobianOplusXj(0,1) = -z;
        _jacobianOplusXj(0,2) = y;
        _jacobianOplusXj(0,3) = -1;
        _jacobianOplusXj(0,4) = 0;
        _jacobianOplusXj(0,5) = 0;

        _jacobianOplusXj(1,0) = z;
        _jacobianOplusXj(1,1) = 0;
        _jacobianOplusXj(1,2) = -x;
        _jacobianOplusXj(1,3) = 0;
        _jacobianOplusXj(1,4) = -1;
        _jacobianOplusXj(1,5) = 0;

        _jacobianOplusXj(2,0) = -y;
        _jacobianOplusXj(2,1) = x;
        _jacobianOplusXj(2,2) = 0;
        _jacobianOplusXj(2,3) = 0;
        _jacobianOplusXj(2,4) = 0;
        _jacobianOplusXj(2,5) = -1;
    }
};

void ICPBAG2O(const vector<Point3f>& pts3d1,const vector<Point3f>& pts3d2,Mat& R,Mat& t)
{
    //1.g2o初始化:线性求解器->矩阵块求解器->
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); //多态多种方式
    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); //
    Block* block_ptr=new Block(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* sovler = new g2o::OptimizationAlgorithmLevenberg(block_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(sovler);

    //2.顶点
    //pose
/*     
    Eigen::Matrix3d R_mat(Eigen::Matrix3d::Identity());
    Eigen::Vector3d t_v(0,0,0);
    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,t_v));
    optimizer.addVertex(pose);
 */   


    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    Eigen::Matrix3d R_mat;
    R_mat <<
    R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
    R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
    R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2);
    Eigen::Vector3d t_v(t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(R_mat,t_v));
    optimizer.addVertex(pose);

    //landmarks
    int index=1;
    for(const Point3d p:pts3d2)
    {
        g2o::VertexSBAPointXYZ* point=new g2o::VertexSBAPointXYZ();
        point->setId(index++);
        point->setEstimate(Eigen::Vector3d(p.x,p.y,p.z));
        point->setMarginalized(true);
        optimizer.addVertex(point);
    }

    //3.边
    const int N=pts3d1.size();
    index=1;
    for(int i=0;i<N;i++)
    {
        EdgeProjectXYZRGBD* edge=new EdgeProjectXYZRGBD();
        edge->setId(index);
        edge->setVertex(0,dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index)));
        edge->setVertex(1,dynamic_cast<g2o::VertexSE3Expmap*>(pose));
        edge->setMeasurement( Eigen::Vector3d(
            pts3d1[i].x, pts3d1[i].y, pts3d1[i].z) );
        edge->setInformation( Eigen::Matrix3d::Identity()*1e4 );
        optimizer.addEdge(edge);
        index++;
    }

    //4.开始优化
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(100);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;
    cout<<endl<<"after optimization:"<<endl;
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}
