#include "../framework/TestAssert.hpp"
#include "pose/MonoPnPSolver.hpp"
#include "pose/BoundedMonoRefinement.hpp"
#include "common/LogConfig.hpp"
#include <Eigen/Geometry>
#include <limits>

using namespace gpnp;
namespace {
constexpr double pi=3.14159265358979323846;
MonoPosePriorConfig prior() { MonoPosePriorConfig p; p.enabled=true; p.max_tilt_deg=30; return p; }
Eigen::Matrix3d rotation(double tilt, double yaw=0) {
    return (Eigen::AngleAxisd(yaw*pi/180,Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(tilt*pi/180,Eigen::Vector3d::UnitY())).toRotationMatrix();
}
std::vector<Eigen::Vector3d> model(int count=10) {
    if(count==4) return {{-150,-150,0},{150,-150,0},{150,150,0},{-150,150,0}};
    return {{-150,-150,0},{-30,-150,0},{150,-150,0},{150,-50,0},{30,-50,0},
            {30,50,0},{150,50,0},{150,150,0},{-150,150,0},{-150,0,0}};
}
Eigen::Matrix3d camera() { Eigen::Matrix3d K; K<<900,0,600,0,900,450,0,0,1; return K; }
std::vector<cv::Point2f> project(const std::vector<Eigen::Vector3d>& points,
        const Eigen::Matrix3d& R,const Eigen::Vector3d& t) {
    std::vector<cv::Point2f> image;
    for(const auto& p:points) { const Eigen::Vector3d q=camera()*(R*p+t); image.emplace_back(q.x()/q.z(),q.y()/q.z()); }
    return image;
}
void test_total_tilt_boundary_and_free_yaw() {
    const Eigen::Vector3d t(20,10,900);
    for(double yaw:{0.,90.,179.,270.}) {
        TEST_ASSERT(planarPosePriorRejection(rotation(30,yaw),t,model(),prior()).empty());
        TEST_ASSERT(planarPosePriorRejection(rotation(30.01,yaw),t,model(),prior())=="tilt_out_of_range");
    }
    const auto composite=(Eigen::AngleAxisd(25*pi/180,Eigen::Vector3d::UnitX())*
                          Eigen::AngleAxisd(25*pi/180,Eigen::Vector3d::UnitY())).toRotationMatrix();
    TEST_ASSERT(planarPosePriorRejection(composite,t,model(),prior())=="tilt_out_of_range");
}
void test_back_face_and_all_point_cheirality() {
    const auto upside=Eigen::AngleAxisd(pi,Eigen::Vector3d::UnitX()).toRotationMatrix();
    TEST_ASSERT(planarPosePriorRejection(upside,{0,0,900},model(),prior())=="camera_on_back_side");
    TEST_ASSERT(planarPosePriorRejection(rotation(20),{0,0,20},model(),prior())=="point_behind_camera");
    TEST_ASSERT(planarPosePriorRejection(Eigen::Matrix3d::Identity(),{0,0,900},model(),prior()).empty());
}
void test_bad_models_rotations_and_config() {
    auto points=model(); points[0].z()=1;
    TEST_ASSERT(planarPosePriorRejection(rotation(0),{0,0,900},points,prior())=="model_not_on_z0");
    auto bad=rotation(0); bad(0,0)=std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT(planarPosePriorRejection(bad,{0,0,900},model(),prior())=="non_finite_pose");
    for(double tilt:{0.,-1.,90.,180.,std::numeric_limits<double>::infinity()}) {
        auto cfg=prior(); cfg.max_tilt_deg=tilt;
        bool threw=false; try { MonoPnPSolver solver(cfg); } catch(const std::invalid_argument&) { threw=true; }
        TEST_ASSERT(threw);
    }
    for(const std::string& json:{"{\"mono_pose_prior\":{\"enabled\":true,\"max_tilt_deg\":30}}",
                                "{\"mono_pose_prior\":{\"enabled\":false,\"max_tilt_deg\":30}}"}) {
        cv::FileStorage fs(json,cv::FileStorage::READ|cv::FileStorage::MEMORY);
        const auto cfg=readMonoPosePriorConfig(fs["mono_pose_prior"]);
        TEST_ASSERT(cfg.max_tilt_deg==30);
    }
    for(const std::string& json:{"{\"mono_pose_prior\":{\"enabled\":\"true\"}}",
                                "{\"mono_pose_prior\":{\"enabled\":2}}",
                                "{\"mono_pose_prior\":{\"max_tilt_deg\":\"30\"}}",
                                "{\"mono_pose_prior\":[]}"}) {
        bool threw=false;
        try { cv::FileStorage fs(json,cv::FileStorage::READ|cv::FileStorage::MEMORY); readMonoPosePriorConfig(fs["mono_pose_prior"]); }
        catch(const std::exception&) { threw=true; }
        TEST_ASSERT(threw);
    }
    cv::FileStorage empty("{}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(!readMonoPosePriorConfig(empty["mono_pose_prior"]).enabled);
}
void test_cold_warm_four_and_ten_points_recover() {
    for(int count:{4,10}) for(double yaw:{0.,90.,179.}) {
        const auto points=model(count); const auto R=rotation(21,yaw);
        const Eigen::Vector3d t(50,25,900); const auto image=project(points,R,t);
        MonoPnPSolver solver(prior());
        PoseSeed seed; seed.R=rotation(75,yaw); seed.t=t;
        for(const PoseSeed* init:{static_cast<const PoseSeed*>(nullptr),static_cast<const PoseSeed*>(&seed)}) {
            const auto pose=solver.solve(image,points,camera(),init,.3);
            TEST_ASSERT(pose.success);
            TEST_ASSERT(planarPosePriorRejection(pose.R,pose.t,points,prior()).empty());
            TEST_ASSERT(Eigen::AngleAxisd(pose.R*R.transpose()).angle()<.001);
        }
    }
}
void test_large_tilt_rejected_without_fallback_escape() {
    for(int count:{4,10}) {
        const auto points=model(count); const auto R=rotation(65,25);
        const Eigen::Vector3d t(20,10,900); const auto image=project(points,R,t);
        MonoPnPSolver unconstrained;
        TEST_ASSERT(unconstrained.solve(image,points,camera()).success);
        MonoPnPSolver constrained(prior()); PoseSeed seed; seed.R=R; seed.t=t;
        TEST_ASSERT(!constrained.solve(image,points,camera()).success);
        TEST_ASSERT(!constrained.solve(image,points,camera(),&seed,.3).success);
    }
}
void test_backside_correspondences_and_invalid_input_rejected() {
    MonoPnPSolver solver(prior()); const auto points=model();
    const auto R=Eigen::AngleAxisd(pi,Eigen::Vector3d::UnitX()).toRotationMatrix();
    TEST_ASSERT(!solver.solve(project(points,R,{0,0,900}),points,camera()).success);
    auto image=project(points,rotation(0),{0,0,900}); image[0].x=std::numeric_limits<float>::quiet_NaN();
    TEST_ASSERT(!solver.solve(image,points,camera()).success);
}
void test_boundary_refit_optimizes_instead_of_switching_mirror() {
    const auto points=model();const Eigen::Vector3d truth_t(40,20,900);
    const auto image=project(points,rotation(29,130),truth_t);
    auto R=rotation(35,133);Eigen::Vector3d t=truth_t+Eigen::Vector3d(12,-5,30);
    TEST_ASSERT(refineMonoWithinTilt(image,points,camera(),30,R,t));
    TEST_ASSERT(planarPosePriorRejection(R,t,points,prior()).empty());
    TEST_ASSERT(Eigen::AngleAxisd(R*rotation(29,130).transpose()).angle()<.01);
    // At 31deg the best feasible result belongs near the 30deg boundary,
    // rather than the opposite small-tilt branch.
    R=rotation(31,130);t=truth_t;
    TEST_ASSERT(refineMonoWithinTilt(project(points,R,t),points,camera(),30,R,t));
    TEST_ASSERT(planarPosePriorRejection(R,t,points,prior()).empty());
    TEST_ASSERT(Eigen::AngleAxisd(R*rotation(31,130).transpose()).angle()<.04);
}
void test_quality_path_requires_enough_pose_support() {
    MonoPnPSolver solver(prior());auto points=model();auto image=project(points,rotation(20),{10,15,1500});
    auto result=solver.solve(image,points,camera(),nullptr,0.,8);TEST_ASSERT(result.success);TEST_ASSERT(result.num_points>=8);
    TEST_ASSERT(solver.lastRejectionReason().empty());
    points.resize(7);image.resize(7);TEST_ASSERT(!solver.solve(image,points,camera(),nullptr,0.,8).success);
    TEST_ASSERT(solver.lastRejectionReason()=="insufficient_pose_points");
    auto square=model(4);auto corners=project(square,rotation(10),{10,15,1500});
    TEST_ASSERT(solver.solve(corners,square,camera()).success);
    TEST_ASSERT(solver.lastRejectionReason().empty());
}
REGISTER_TEST(test_quality_path_requires_enough_pose_support);
REGISTER_TEST(test_total_tilt_boundary_and_free_yaw);
REGISTER_TEST(test_back_face_and_all_point_cheirality);
REGISTER_TEST(test_bad_models_rotations_and_config);
REGISTER_TEST(test_cold_warm_four_and_ten_points_recover);
REGISTER_TEST(test_large_tilt_rejected_without_fallback_escape);
REGISTER_TEST(test_backside_correspondences_and_invalid_input_rejected);
REGISTER_TEST(test_boundary_refit_optimizes_instead_of_switching_mirror);
} // namespace
int main() { gpnp::g_verbose_console=false; cv::setRNGSeed(20261016); return gpnp_test::TestRegistry::instance().runAll(); }
