#include "../framework/TestAssert.hpp"
#include "common/MonoDualRoiPolicy.hpp"
#include <Eigen/Geometry>
#include <limits>
using namespace gpnp;
namespace {
const Eigen::Matrix3d K=(Eigen::Matrix3d()<<800,0,320,0,800,240,0,0,1).finished();
std::vector<Eigen::Vector3d> model(double half) {
    return {{-half,-half,0},{half,-half,0},{half,half,0},{-half,half,0}};
}
std::vector<cv::Point2f> project(const PoseEstimate& p,const std::vector<Eigen::Vector3d>& m) {
    std::vector<cv::Point2f> uv;
    for(auto v:m){Eigen::Vector3d q=K*(p.R*v+p.t);uv.emplace_back(q.x()/q.z(),q.y()/q.z());}
    return uv;
}
PoseEstimate pose(){PoseEstimate p;p.success=true;p.t={0,0,1000};return p;}
void test_joint_checks_original_primary_and_both_groups() {
    auto p=pose(),q=p;auto a=model(150),b=model(10);auto u=project(p,a),v=project(p,b);
    MonoDualRoiConfig cfg;
    TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg).empty());
    q.t.x()=2;TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg)=="primary_reprojection_worse");
    q=p;v[0].x+=20;TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg)=="group_reprojection");
    q.t.z()=-1;TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg)=="nonfinite_or_depth");
}
void test_joint_pose_jump_checks_are_independent_of_reprojection() {
    auto p=pose(),q=p;auto a=model(150),b=model(10);auto u=project(p,a),v=project(p,b);
    MonoDualRoiConfig cfg;cfg.max_primary_error_increase_px=1000;cfg.max_group_error_px=1000;
    q.R=Eigen::AngleAxisd(20*CV_PI/180,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg)=="rotation_delta");
    q=p;q.t.z()=1200;TEST_ASSERT(checkMonoDualJoint(p,q,u,a,v,b,K,cfg)=="translation_delta");
}
void test_dual_sparse_ids_must_be_unique_and_not_prefix_reindexed() {
    PipelineResult r;r.success=true;
    auto m=model(10);r.pts_left_match=project(pose(),m);
    for(int i=0;i<4;++i)r.good_matches.emplace_back(i,i,0);
    TEST_ASSERT(monoDualMatchesValid(r,m));
    r.good_matches[1].trainIdx=0;TEST_ASSERT(!monoDualMatchesValid(r,m));
    r.good_matches[1].trainIdx=1;r.good_matches[3].trainIdx=4;TEST_ASSERT(!monoDualMatchesValid(r,m));
    r.good_matches[3].trainIdx=3;r.pose_min_inliers=8;TEST_ASSERT(!monoDualMatchesValid(r,m));
    r.pose_min_inliers=4;r.pts_left_match[0].x=std::numeric_limits<float>::quiet_NaN();
    TEST_ASSERT(!monoDualMatchesValid(r,m));
    m.resize(10,Eigen::Vector3d(1,2,0));r.pts_left_match[0].x=1;
    const int ids[]={0,2,7,9};for(int i=0;i<4;++i)r.good_matches[i].trainIdx=ids[i];
    TEST_ASSERT(monoDualMatchesValid(r,m));
}
void test_dual_config_missing_is_inert_and_bad_values_rejected() {
    cv::FileStorage empty("{}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(!readMonoDualRoiConfig(empty["dual"]).enabled);
    for(const auto value:{"[]","{\"enabled\":2}","{\"size_aware_class1\":\"yes\"}",
        "{\"max_group_error_px\":-1}","{\"max_rotation_delta_deg\":181}",
        "{\"max_translation_delta_ratio\":2}","{\"max_group_error_px\":\"3\"}"}) {
        cv::FileStorage fs(std::string("{\"dual\":")+value+"}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
        TEST_ASSERT_THROWS(readMonoDualRoiConfig(fs["dual"]),std::exception);
    }
}
REGISTER_TEST(test_joint_checks_original_primary_and_both_groups);
REGISTER_TEST(test_joint_pose_jump_checks_are_independent_of_reprojection);
REGISTER_TEST(test_dual_sparse_ids_must_be_unique_and_not_prefix_reindexed);
REGISTER_TEST(test_dual_config_missing_is_inert_and_bad_values_rejected);
}
int main(){return gpnp_test::TestRegistry::instance().runAll();}
