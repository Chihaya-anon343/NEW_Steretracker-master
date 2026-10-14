#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
#include "pose/GPnPSolver.hpp"
#include <Eigen/Geometry>
#include <limits>

using namespace gpnp;
namespace {
StereoCameraParams calibration() {
    Eigen::Matrix3d k;k<<1000,0,512,0,1000,512,0,0,1;
    return makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),{120,0,0});
}
std::vector<Eigen::Vector3d> modelPoints() {
    return {{-80,-50,0},{80,-50,0},{80,50,0},{-80,50,0},
            {-50,-25,0},{50,-25,0},{50,25,0},{-50,25,0}};
}
PipelineResult exactStereo(const StereoCameraParams& p,const std::vector<Eigen::Vector3d>& model,
                           const Eigen::Matrix3d& r,const Eigen::Vector3d& t) {
    PipelineResult out;
    for(int i=0;i<static_cast<int>(model.size());++i) {
        const Eigen::Vector3d pc=r*model[i]+t,pr=p.R_rl.transpose()*(pc-p.t_rl);
        const Eigen::Vector3d u=p.K*pc,v=p.K*pr;
        const cv::Point2f l(u.x()/u.z(),u.y()/u.z()),rr(v.x()/v.z(),v.y()/v.z());
        out.pts_left_match.push_back(l);out.pts_left_good.push_back(l);out.pts_right_good.push_back(rr);
        out.idx_from_filtered.push_back(i);out.good_matches.emplace_back(i,i,0);
        out.disparity.push_back(l.x-rr.x);
    }
    return out;
}
void test_gpnp_six_dof_exact_identity_and_tilt() {
    auto p=calibration();auto model=modelPoints();
    for(double angle:{0.0,1e-9,0.3}) {
        Eigen::Matrix3d r=Eigen::AngleAxisd(angle,Eigen::Vector3d::UnitY()).toRotationMatrix();
        Eigen::Vector3d t(0,0,1200);
        auto data=exactStereo(p,model,r,t);GPnPSolver solver(p);double ms=0;
        auto pose=solver.solve(data,model,&r,&t,ms);
        TEST_ASSERT(pose.success&&pose.source==PoseSource::StereoGpnp);
        TEST_ASSERT((pose.t-t).norm()<.01);TEST_ASSERT((pose.R-r).norm()<1e-3);
        TEST_ASSERT(solver.lastMonitor().candidate_valid&&solver.lastMonitor().pose_accepted);
        TEST_ASSERT(data.last_gpnp_monitor.opt_status>=1&&data.last_gpnp_monitor.opt_status<=4);
    }
}
void test_gpnp_invalid_indices_counts_and_nonfinite_values() {
    auto p=calibration();auto model=modelPoints();
    const Eigen::Matrix3d r=Eigen::Matrix3d::Identity();const Eigen::Vector3d t(0,0,1200);
    GPnPSolver solver(p);double ms=0;
    for(int variant=0;variant<7;++variant) {
        auto data=exactStereo(p,model,r,t);
        if(variant==0)data.good_matches[0].queryIdx=-1;
        if(variant==1)data.idx_from_filtered[0]=-1;
        if(variant==2)data.pts_right_good.pop_back();
        if(variant==3)data.pts_right_good[0].x=std::numeric_limits<float>::quiet_NaN();
        if(variant==4)data.good_matches[0].trainIdx=1000000;
        if(variant==5)data.idx_from_filtered[1]=0;
        if(variant==6)data.good_matches[1].trainIdx=0;
        TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
        TEST_ASSERT(!solver.lastMonitor().failure_reason.empty());
    }
}
void test_gpnp_invalid_initial_pose_and_calibration() {
    auto p=calibration();auto model=modelPoints();
    Eigen::Matrix3d r=Eigen::Matrix3d::Identity();Eigen::Vector3d t(0,0,1200);
    auto data=exactStereo(p,model,r,t);GPnPSolver solver(p);double ms=0;
    TEST_ASSERT(!solver.solve(data,model,&r,nullptr,ms).success);
    r.setZero();TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    r.setIdentity();t.x()=std::numeric_limits<double>::infinity();
    TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    t=Eigen::Vector3d(0,0,-1200);
    TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    TEST_ASSERT_EQ(solver.lastMonitor().failure_reason,std::string("nonpositive_initial_depth"));
    p.K_inv.setZero();GPnPSolver bad_camera(p);
    TEST_ASSERT(!bad_camera.solve(data,model,nullptr,nullptr,ms).success);
}
void test_gpnp_budget_exhaustion_is_not_success() {
    auto p=calibration();auto model=modelPoints();
    Eigen::Matrix3d r=Eigen::AngleAxisd(.25,Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Vector3d t(0,0,1200);auto data=exactStereo(p,model,r,t);
    Eigen::Matrix3d seed=Eigen::Matrix3d::Identity();Eigen::Vector3d seed_t(10,10,1500);
    GPnPOptions options;options.max_function_evaluations=1;
    GPnPSolver solver(p);double ms=0;
    TEST_ASSERT(!solver.solve(data,model,&seed,&seed_t,ms,options).success);
    TEST_ASSERT_EQ(solver.lastMonitor().opt_status,5);
    TEST_ASSERT(!solver.lastMonitor().opt_success&&!solver.lastMonitor().pose_accepted);
}
void test_gpnp_rejects_collinear_model() {
    auto p=calibration();auto model=modelPoints();
    for(int i=0;i<static_cast<int>(model.size());++i)model[i]={double(i)*20,0,0};
    const Eigen::Matrix3d r=Eigen::Matrix3d::Identity();const Eigen::Vector3d t(0,0,1200);
    auto data=exactStereo(p,model,r,t);GPnPSolver solver(p);double ms=0;
    TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    TEST_ASSERT_EQ(solver.lastMonitor().failure_reason,std::string("degenerate_model_geometry"));
}
void test_gpnp_geometry_and_explicit_initial_source() {
    auto p=calibration();auto model=modelPoints();
    const Eigen::Matrix3d r=Eigen::Matrix3d::Identity();const Eigen::Vector3d t(0,0,1200);
    auto data=exactStereo(p,model,r,t);for(auto& right:data.pts_right_good)right.x-=50;
    GPnPSolver solver(p);double ms=0;
    TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    GPnPOptions options;options.initial_source=PoseInitializationSource::InitialPnP;
    auto pose=solver.solve(data,model,&r,&t,ms,options);
    TEST_ASSERT(pose.success&&pose.source==PoseSource::LeftPnpFallback);
    TEST_ASSERT(solver.lastMonitor().used_initial_pose_fallback);
}
void test_gpnp_degenerate_image_geometry_is_rejected() {
    auto p=calibration();auto model=modelPoints();
    const Eigen::Matrix3d r=Eigen::Matrix3d::Identity();const Eigen::Vector3d t(0,0,1200);
    auto data=exactStereo(p,model,r,t);
    for(auto& pixel:data.pts_left_good)pixel={512,512};
    GPnPSolver solver(p);double ms=0;
    TEST_ASSERT(!solver.solve(data,model,&r,&t,ms).success);
    TEST_ASSERT_EQ(solver.lastMonitor().failure_reason,std::string("degenerate_image_geometry"));
}
}
REGISTER_TEST(test_gpnp_six_dof_exact_identity_and_tilt);
REGISTER_TEST(test_gpnp_invalid_indices_counts_and_nonfinite_values);
REGISTER_TEST(test_gpnp_invalid_initial_pose_and_calibration);
REGISTER_TEST(test_gpnp_budget_exhaustion_is_not_success);
REGISTER_TEST(test_gpnp_rejects_collinear_model);
REGISTER_TEST(test_gpnp_geometry_and_explicit_initial_source);
REGISTER_TEST(test_gpnp_degenerate_image_geometry_is_rejected);
