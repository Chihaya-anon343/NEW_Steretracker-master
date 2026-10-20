#pragma once
#include "common/Types.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpnp {
inline bool monoDualMatchesValid(const PipelineResult& r,
                                const std::vector<Eigen::Vector3d>& model) {
    if(!r.success || r.pts_left_match.size()!=r.good_matches.size() ||
       r.good_matches.size()<size_t(std::max(4,r.pose_min_inliers)))return false;
    std::vector<bool> used(model.size(),false);
    for(size_t i=0;i<r.good_matches.size();++i) {
        const auto& m=r.good_matches[i];const auto p=r.pts_left_match[i];
        if(m.queryIdx!=int(i)||m.trainIdx<0||m.trainIdx>=int(model.size())||
           used[m.trainIdx]||!model[m.trainIdx].allFinite()||
           !std::isfinite(p.x)||!std::isfinite(p.y))return false;
        used[m.trainIdx]=true;
    }
    return true;
}
inline MonoDualRoiConfig readMonoDualRoiConfig(const cv::FileNode& node) {
    MonoDualRoiConfig c;
    if (node.empty()) return c;
    if (!node.isMap()) throw std::invalid_argument("dual_roi.mono_robustness must be an object");
    auto boolean = [&](const char* key, bool& value) {
        auto n=node[key]; if(n.empty())return;
        if(!n.isInt() || (int(n)!=0 && int(n)!=1))
            throw std::invalid_argument(std::string("dual_roi.mono_robustness.")+key+" must be boolean");
        value=int(n)!=0;
    };
    auto positive = [&](const char* key, double& value) {
        auto n=node[key];if(n.empty())return;
        if(!n.isInt()&&!n.isReal())throw std::invalid_argument(std::string(key)+" must be numeric");
        value=double(n);
        if(!std::isfinite(value)||value<0)throw std::invalid_argument(std::string(key)+" must be finite and nonnegative");
    };
    boolean("enabled",c.enabled);
    boolean("size_aware_class1",c.size_aware_class1);
    boolean("joint_quality_check",c.joint_quality_check);
    positive("max_primary_error_increase_px",c.max_primary_error_increase_px);
    positive("max_group_error_px",c.max_group_error_px);
    positive("max_rotation_delta_deg",c.max_rotation_delta_deg);
    positive("max_translation_delta_ratio",c.max_translation_delta_ratio);
    if(c.max_rotation_delta_deg>180 || c.max_translation_delta_ratio>1)
        throw std::invalid_argument("DualROI joint pose limits out of range");
    return c;
}

inline double monoGroupReprojection(const PoseEstimate& pose,
    const std::vector<cv::Point2f>& pixels,const std::vector<Eigen::Vector3d>& model,
    const Eigen::Matrix3d& K) {
    if(!pose.success || !pose.R.allFinite() || !pose.t.allFinite() ||
       pixels.empty() || pixels.size()!=model.size())return std::numeric_limits<double>::infinity();
    double sum=0;
    for(size_t i=0;i<pixels.size();++i) {
        const Eigen::Vector3d p=pose.R*model[i]+pose.t;
        if(!p.allFinite()||p.z()<=0||!std::isfinite(pixels[i].x)||!std::isfinite(pixels[i].y))
            return std::numeric_limits<double>::infinity();
        const Eigen::Vector3d u=K*p;
        sum+=std::hypot(u.x()/u.z()-pixels[i].x,u.y()/u.z()-pixels[i].y);
    }
    return sum/pixels.size();
}

// Compare the SAME original primary observations under both poses. A lower
// pooled error on a different/larger point set is not evidence of improvement.
inline std::string checkMonoDualJoint(const PoseEstimate& primary,const PoseEstimate& joint,
    const std::vector<cv::Point2f>& primary_pixels,const std::vector<Eigen::Vector3d>& primary_model,
    const std::vector<cv::Point2f>& secondary_pixels,const std::vector<Eigen::Vector3d>& secondary_model,
    const Eigen::Matrix3d& K,const MonoDualRoiConfig& cfg) {
    const double before=monoGroupReprojection(primary,primary_pixels,primary_model,K);
    const double after=monoGroupReprojection(joint,primary_pixels,primary_model,K);
    const double secondary=monoGroupReprojection(joint,secondary_pixels,secondary_model,K);
    if(!std::isfinite(before)||!std::isfinite(after)||!std::isfinite(secondary))return "nonfinite_or_depth";
    if(after>before+cfg.max_primary_error_increase_px)return "primary_reprojection_worse";
    if(after>cfg.max_group_error_px || secondary>cfg.max_group_error_px)return "group_reprojection";
    const double cosine=std::clamp(((joint.R*primary.R.transpose()).trace()-1)*.5,-1.,1.);
    if(std::acos(cosine)*180/CV_PI>cfg.max_rotation_delta_deg)return "rotation_delta";
    if((joint.t-primary.t).norm()>cfg.max_translation_delta_ratio*std::max(primary.t.norm(),1.))
        return "translation_delta";
    return {};
}
} // namespace gpnp
