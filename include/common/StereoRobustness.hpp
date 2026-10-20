#pragma once
#include "common/MonoDualRoiPolicy.hpp"
#include "matching/StereoFlowMatcher.hpp"

namespace gpnp {
inline StereoRobustnessConfig readStereoRobustnessConfig(const cv::FileNode& node) {
    StereoRobustnessConfig c;
    if(node.empty())return c;
    if(!node.isMap())throw std::invalid_argument("stereo_frontend.robustness must be an object");
    auto flag=[&](const char* key,bool& v){
        auto n=node[key];if(n.empty())return;
        if(!n.isInt()||(int(n)!=0&&int(n)!=1))throw std::invalid_argument(std::string(key)+" must be boolean");
        v=int(n)!=0;
    };
    flag("enabled",c.enabled);flag("binary_corner",c.binary_corner);
    flag("tiny_target",c.tiny_target);flag("joint_quality_check",c.joint_quality_check);
    if(!node["min_bc_points"].empty()) {
        if(!node["min_bc_points"].isInt())throw std::invalid_argument("min_bc_points must be integer");
        c.min_bc_points=int(node["min_bc_points"]);
    }
    if(c.min_bc_points<4||c.min_bc_points>10)throw std::invalid_argument("min_bc_points must be in [4,10]");
    c.joint_limits=readMonoDualRoiConfig(node["joint_limits"]);
    return c;
}

// A stereo joint pose must preserve the SAME primary pairs in BOTH cameras.
inline std::string checkStereoDualJoint(const PoseEstimate& primary,const PoseEstimate& joint,
    const PipelineResult& c0,const PipelineResult& c1,const std::vector<Eigen::Vector3d>& model,
    const StereoCameraParams& camera,const MonoDualRoiConfig& c) {
    auto a=checkFlowPose(primary,c0,model,camera,true,1e9,1e9);
    auto b=checkFlowPose(joint,c0,model,camera,true,1e9,1e9);
    auto s=checkFlowPose(joint,c1,model,camera,true,1e9,1e9);
    if(!a.success||!b.success||!s.success)return "nonfinite_or_depth";
    if(b.mean_left_px>a.mean_left_px+c.max_primary_error_increase_px||
       b.mean_right_px>a.mean_right_px+c.max_primary_error_increase_px)return "primary_reprojection_worse";
    if(std::max({b.mean_left_px,b.mean_right_px,s.mean_left_px,s.mean_right_px})>c.max_group_error_px)
        return "group_reprojection";
    const double cosine=std::clamp(((joint.R*primary.R.transpose()).trace()-1)*.5,-1.,1.);
    if(std::acos(cosine)*180/CV_PI>c.max_rotation_delta_deg)return "rotation_delta";
    if((joint.t-primary.t).norm()>c.max_translation_delta_ratio*std::max(primary.t.norm(),1.))return "translation_delta";
    return {};
}
} // namespace gpnp
