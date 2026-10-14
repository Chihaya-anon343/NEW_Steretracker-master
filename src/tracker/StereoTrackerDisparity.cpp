#include "tracker/StereoTracker.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace gpnp {
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
cv::Rect padded(const RoiRect& r, cv::Size size, int pad) {
    return cv::Rect(r.x-pad,r.y-pad,r.width+2*pad,r.height+2*pad)&cv::Rect(0,0,size.width,size.height);
}
}

bool StereoTracker::terminalDisparityFailure(const std::string& reason) {
    return reason.find("negative_disparity")!=std::string::npos ||
        reason.rfind("invalid_",0)==0 || reason=="empty_input";
}

void StereoTracker::writeDisparityResult(const PipelineResult& r) const {
    if(output_dir_.empty())return;
    std::ofstream csv(output_dir_+"/disparity_f"+std::to_string(current_frame_)+".csv");
    csv<<"left_index,left_x,left_y,right_x,right_y,disparity_px,region,original_index,template_index,model_frame,strategy,frame_match_ok\n";
    for(size_t k=0;k<r.disparity.size();++k) {
        const int original=r.idx_from_filtered[k];
        const int region=k<r.flow_regions.size()?r.flow_regions[k]:(r.is_class1?1:0);
        const int local=k<r.flow_original_indices.size()?r.flow_original_indices[k]:original;
        const int model=k<r.flow_template_indices.size()?r.flow_template_indices[k]:-1;
        csv<<original<<','<<r.pts_left_good[k].x<<','<<r.pts_left_good[k].y<<','
           <<r.pts_right_good[k].x<<','<<r.pts_right_good[k].y<<','<<r.disparity[k]<<','
           <<region<<','<<local<<','<<model<<','<<r.model_frame<<','<<r.strategy_name<<','
           <<(r.disparity_success?1:0)<<'\n';
    }
}

PipelineResult StereoTracker::runDisparityExtraction(FeatureExtractor& ext, StereoFlowFrame& context,
        const cv::Mat& gray, const cv::Mat& color, const cv::Mat& right_color,
        const cv::Rect& roi, int region, FlowMatchResult& diagnostics) {
    const auto start=Clock::now();
    auto r=ext.extractMono(gray(roi),color.empty()?cv::Mat():color(roi));
    const double extraction_ms=elapsed(start);
    // Extraction flags/model dimensions never authorize a pose in this path.
    r.success=false;r.gpnp_success=false;r.stereo_success=false;r.disparity_only=true;
    r.pose_source=PoseSource::Unspecified;r.pose_model_points.clear();r.model_frame="unresolved";
    r.strategy_name=ext.name();r.stereo_method="LK";r.is_class1=region==1;
    r.left_color=color;r.right_color=right_color;
    r.left_roi_offset_x=roi.x;r.left_roi_offset_y=roi.y;
    r.timing.clear();r.timing["disparity_extract"]=extraction_ms;
    for(auto& p:r.pts_left_match)p+=cv::Point2f(roi.x,roi.y);
    for(auto& p:r.kp_left)p.pt+=cv::Point2f(roi.x,roi.y);
    const bool tiny=ext.strategyType()==StrategyType::TinyTarget;
    if((tiny&&r.pts_left_match.size()!=4)||(!tiny&&r.pts_left_match.size()<4)) {
        r.failure_reason=tiny?"tt_extraction_failed":"bc_extraction_failed";
        diagnostics.failure_reason=r.failure_reason;
        r.extract_ms=extraction_ms;return r;
    }
    diagnostics=context.match(r.pts_left_match,r.pts_left_match,tiny?4:config_.stereo_flow.min_matches);
    r.pts_left_good=diagnostics.left_points;r.pts_right_good=diagnostics.right_points;
    r.idx_from_filtered=diagnostics.left_indices;
    r.pts_left_used=r.pts_left_good;r.pts_right_used=r.pts_right_good;
    r.n_matched=static_cast<int>(r.pts_left_good.size());
    r.dual_c0_pairs=region==0?r.n_matched:0;r.dual_c1_pairs=region==1?r.n_matched:0;
    r.disparity_success=diagnostics.success;r.failure_reason=diagnostics.failure_reason;
    r.flow_planar_status=diagnostics.planar_status;r.flow_planar_rejected=diagnostics.planar_rejected;
    r.flow_planar_ms=diagnostics.planar_ms;
    r.valid_mask.assign(r.pts_left_match.size(),false);
    std::vector<int> models(r.pts_left_match.size(),-1);
    for(const auto& m:r.good_matches)
        if(m.queryIdx>=0&&m.queryIdx<static_cast<int>(models.size()))models[m.queryIdx]=m.trainIdx;
    for(size_t k=0;k<r.idx_from_filtered.size();++k) {
        const int i=r.idx_from_filtered[k];r.valid_mask[i]=true;
        const double d=r.pts_left_good[k].x-r.pts_right_good[k].x;
        r.disparity.push_back(d);r.dx_filtered.push_back(d);
        r.flow_regions.push_back(region);r.flow_original_indices.push_back(i);
        r.flow_template_indices.push_back(models[i]);
    }
    r.flow_roi=diagnostics.search_roi;r.flow_full_fallback=diagnostics.full_frame_fallback;
    r.flow_coarse_attempts=diagnostics.coarse_attempts;r.flow_pyramid_builds=diagnostics.pyramid_builds;
    r.stereo_match_ms=diagnostics.total_ms;
    r.timing["flow"]=diagnostics.total_ms;r.extract_ms=extraction_ms+diagnostics.total_ms;
    return r;
}

PipelineResult StereoTracker::processSingleDisparity(const cv::Mat& left, const cv::Mat& right,
                                                     const RoiGroup& group, bool visualize) {
    PipelineResult result;result.disparity_only=true;result.stereo_method="LK";
    result.is_first_frame=!state_.has_cache;result.is_class1=group.is_class1;
    auto [lc,lg]=loadImage(left,visualize);auto [rc,rg]=loadImage(right,visualize);
    if(lg.empty()||rg.empty()||lg.size()!=rg.size()||lg.type()!=CV_8UC1||rg.type()!=CV_8UC1) {
        result.failure_reason="invalid_stereo_images";result.route_detail="FRAME:invalid_images";
        writeDisparityResult(result);return result;
    }
    const auto base=cv::Rect(group.primary.x,group.primary.y,group.primary.width,group.primary.height);
    if(base.area()<=0||(base&cv::Rect(0,0,lg.cols,lg.rows))!=base) {
        result.failure_reason="invalid_left_roi";writeDisparityResult(result);return result;
    }
    struct Guard {
        BinaryCornerExtractor* bc;TinyTargetExtractor* tt;
        ~Guard(){bc->setUseClass1(false);tt->setUseClass1(false);}
    } guard{binary_extractor_.get(),tiny_extractor_.get()};
    binary_extractor_->setUseClass1(group.is_class1);tiny_extractor_->setUseClass1(group.is_class1);
    configureStrategyChain(base.area(),group.is_class1);
    std::vector<FeatureExtractor*> chain{extractor_};
    for(auto* ext:fallback_extractors_)
        if(std::find(chain.begin(),chain.end(),ext)==chain.end())chain.push_back(ext);
    const auto bc_roi=padded(group.primary,lg.size(),binary_roi_pad_);
    const auto tt_roi=padded(group.primary,lg.size(),tiny_roi_pad_);
    StereoFlowFrame context(camera_,config_.stereo_flow,lg,rg,bc_roi|tt_roi);
    double extraction_ms=0,match_ms=0;
    int builds=0,coarse=0;bool full_retry=false;
    std::string route;
    FlowMatchResult diagnostics;
    FeatureExtractor* last_ext=extractor_;
    cv::Rect last_roi;
    for(size_t attempt=0;attempt<chain.size();++attempt) {
        auto* ext=chain[attempt];const bool tiny=ext->strategyType()==StrategyType::TinyTarget;
        last_ext=ext;last_roi=tiny?tt_roi:bc_roi;diagnostics=FlowMatchResult{};
        result=runDisparityExtraction(*ext,context,lg,lc,rc,last_roi,group.is_class1?1:0,diagnostics);
        extraction_ms+=result.timing["disparity_extract"];match_ms+=result.stereo_match_ms;
        builds+=result.flow_pyramid_builds;coarse+=result.flow_coarse_attempts;
        full_retry|=result.flow_full_fallback;
        if(!route.empty())route+=" -> ";
        route+=(group.is_class1?"C1":"C0")+std::string(tiny?"TT:":"BC:")+
            (result.disparity_success?"OK":result.failure_reason);
        result.fallback_used=attempt>0;
        if(result.disparity_success||terminalDisparityFailure(result.failure_reason))break;
    }
    result.is_first_frame=!state_.has_cache;result.route_detail=route;
    result.extract_ms=extraction_ms+match_ms;result.stereo_match_ms=match_ms;
    result.flow_pyramid_builds=builds;result.flow_coarse_attempts=coarse;result.flow_full_fallback=full_retry;
    result.timing={{"disparity_extract",extraction_ms},{"flow",match_ms}};
    if(result.disparity_success)updateStickinessFromWinner(last_ext->name());
    fillTemporalMeta(result);
    writeDisparityResult(result);
    if(visualize&&!output_dir_.empty())renderBcFlow(result,diagnostics,
        {last_roi.x,last_roi.y,last_roi.width,last_roi.height},lc,rc,last_ext);
    return result;
}
} // namespace gpnp
