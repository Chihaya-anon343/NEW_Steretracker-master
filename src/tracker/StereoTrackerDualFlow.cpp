#include "utils/PoseAxisDisplay.hpp"
#include "tracker/StereoTracker.hpp"
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace gpnp {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
cv::Rect bounded(const RoiRect& r, cv::Size size, int pad=0) {
    if(!r.valid())return {};
    return cv::Rect(r.x-pad,r.y-pad,r.width+2*pad,r.height+2*pad)&cv::Rect(0,0,size.width,size.height);
}
struct GroupFlow {
    int region=0;
    PipelineResult raw;
    std::vector<int> model_indices;
    FlowMatchResult flow;
    cv::Mat binary;
    std::vector<std::pair<std::string,cv::Mat>> stages;
};
cv::Mat color(const cv::Mat& input) {
    cv::Mat output;
    if(input.channels()==1)cv::cvtColor(input,output,cv::COLOR_GRAY2BGR);
    else output=input.clone();
    return output;
}
void panel(const std::string& file, cv::Mat image, const std::string& title) {
    if(image.empty())return;
    if(image.cols>1600)cv::resize(image,image,{1600,std::max(1,image.rows*1600/image.cols)});
    cv::copyMakeBorder(image,image,36,0,0,0,cv::BORDER_CONSTANT,{25,25,25});
    cv::putText(image,title,{8,24},cv::FONT_HERSHEY_SIMPLEX,.5,{255,255,255},1,cv::LINE_AA);
    utils::AsyncImageSaver::write(file,image);
}
void writeDisparity(const PipelineResult& r,const std::string& directory,int frame) {
    if(directory.empty())return;
    std::ofstream csv(directory+"/disparity_f"+std::to_string(frame)+".csv");
    csv<<"left_index,left_x,left_y,right_x,right_y,disparity_px,region,original_index,template_index,model_frame\n";
    for(size_t i=0;i<r.disparity.size();++i)
        csv<<r.idx_from_filtered[i]<<','<<r.pts_left_good[i].x<<','<<r.pts_left_good[i].y<<','
           <<r.pts_right_good[i].x<<','<<r.pts_right_good[i].y<<','<<r.disparity[i]<<','
           <<r.flow_regions[i]<<','<<r.flow_original_indices[i]<<','<<r.flow_template_indices[i]<<','<<r.model_frame<<'\n';
}
}

PipelineResult StereoTracker::processDualRoiFlow(const cv::Mat& left, const cv::Mat& right,
                                                const RoiGroup& group, bool visualize, PoseEstimate& pose) {
    if(config_.stereo_robustness.enabled)return processRobustStereo(left,right,group,visualize,pose);
    const auto& cfg=config_.stereo_flow;
    auto [lc,lg]=loadImage(left,visualize);
    auto [rc,rg]=loadImage(right,visualize);
    PipelineResult result;
    result.stereo_method="LK";result.disparity_only=cfg.disparity_only;
    result.strategy_name="DualRoi_BC";result.is_first_frame=!state_.has_cache;
    result.left_color=lc;result.right_color=rc;
    if(lg.empty()||rg.empty()||lg.size()!=rg.size()||lg.type()!=CV_8UC1||rg.type()!=CV_8UC1) {
        result.failure_reason="invalid_stereo_images";result.route_detail="DualFlow:invalid_images";
        return result;
    }
    const auto pri=bounded(group.primary,lg.size());
    const auto sec=bounded(group.secondary,lg.size(),config_.dual_roi_secondary_expand);
    if(pri.empty()||sec.empty()) {
        result.failure_reason="invalid_dual_roi";result.route_detail="DualFlow:invalid_roi";return result;
    }
    struct ScaleGuard { BinaryCornerExtractor* bc; ~ScaleGuard(){bc->setUseClass1(false);} } guard{binary_extractor_.get()};
    binary_extractor_->setUseClass1(false);
    prepareDualBcTemplate();
    std::vector<Eigen::Vector3d> board=dual_bc_tmpl_pts3d_;
    board.insert(board.end(),dual_c1_tmpl_pts3d_.begin(),dual_c1_tmpl_pts3d_.end());
    const int n0=static_cast<int>(dual_bc_tmpl_pts3d_.size());
    const bool short_circuit=!config_.dual_roi_tier1_only && config_.dual_roi_class1_fallback &&
        config_.dual_roi_primary_span_ratio>0 &&
        pri.width>=lg.cols*config_.dual_roi_primary_span_ratio &&
        pri.height>=lg.rows*config_.dual_roi_primary_span_ratio;
    const auto tt_roi=(cv::Rect(sec.x-tiny_roi_pad_,sec.y-tiny_roi_pad_,
        sec.width+2*tiny_roi_pad_,sec.height+2*tiny_roi_pad_)&cv::Rect(0,0,lg.cols,lg.rows));
    const cv::Rect search_union=cfg.disparity_only&&cfg.disparity_full_chain
        ? ((short_circuit ? sec : (pri|sec))|tt_roi) : (short_circuit ? sec : (pri|sec));
    std::unique_ptr<StereoFlowFrame> context;
    GroupFlow c0,c1;c0.region=0;c1.region=1;
    double bc_ms=0,tt_ms=0,initial_ms=0,flow_ms=0,gpnp_ms=0,association_ms=0;
    double pyramid_ms=0,coarse_ms=0,lk_ms=0,check_ms=0,crop_ms=0;
    int builds=0,coarse_attempts=0;
    bool full_retry=false;
    cv::Rect search_roi;
    std::string route=short_circuit?"C0:short_circuit":"";
    auto step=[&](const std::string& text){if(!route.empty())route+=" -> ";route+=text;};
    auto extract=[&](GroupFlow& g,const cv::Rect& roi) {
        const auto start=Clock::now();
        g.raw=binary_extractor_->extractMono(lg(roi),lc.empty()?cv::Mat():lc(roi));
        for(auto& p:g.raw.pts_left_match)p+=cv::Point2f(roi.x,roi.y);
        for(auto& p:g.raw.kp_left)p.pt+=cv::Point2f(roi.x,roi.y);
        g.model_indices.assign(g.raw.pts_left_match.size(),-1);
        if(visualize && visualize_detailed_) {
            g.binary=binary_extractor_->lastLeftBinary().clone();
            g.stages=binary_extractor_->lastBinaryProcess();
        }
        bc_ms+=ms(start);
    };
    // All solver indices refer to explicit caller-owned models, never ordinary BC scale.
    auto makeResult=[&](const std::vector<const GroupFlow*>& groups,const std::string& frame) {
        PipelineResult r;r.stereo_method="LK";r.disparity_only=cfg.disparity_only;r.model_frame=frame;
        for(const auto* g:groups) {
            if(r.flow_planar_status=="disabled")r.flow_planar_status.clear();
            if(!r.flow_planar_status.empty())r.flow_planar_status+=",";
            r.flow_planar_status+="c"+std::to_string(g->region)+":"+g->flow.planar_status;
            r.flow_planar_rejected+=g->flow.planar_rejected;r.flow_planar_ms+=g->flow.planar_ms;
            const int base=static_cast<int>(r.pts_left_match.size());
            r.pts_left_match.insert(r.pts_left_match.end(),g->raw.pts_left_match.begin(),g->raw.pts_left_match.end());
            r.kp_left.insert(r.kp_left.end(),g->raw.kp_left.begin(),g->raw.kp_left.end());
            r.pts_template_match.resize(r.pts_left_match.size());
            for(int i=0;i<static_cast<int>(g->model_indices.size());++i)
                if(g->model_indices[i]>=0)r.good_matches.emplace_back(base+i,g->model_indices[i],0);
            for(size_t k=0;k<g->flow.left_indices.size();++k) {
                const int original=g->flow.left_indices[k];
                bool duplicate=false;
                for(const auto& p:r.pts_right_good)
                    if(cv::norm(p-g->flow.right_points[k])<cfg.duplicate_distance_px)duplicate=true;
                if(duplicate)continue;
                r.idx_from_filtered.push_back(base+original);
                r.pts_left_good.push_back(g->flow.left_points[k]);r.pts_right_good.push_back(g->flow.right_points[k]);
                r.flow_regions.push_back(g->region);r.flow_original_indices.push_back(original);
                r.flow_template_indices.push_back(g->model_indices[original]);
                const double d=g->flow.left_points[k].x-g->flow.right_points[k].x;
                r.disparity.push_back(d);r.dx_filtered.push_back(d);
            }
        }
        r.valid_mask.assign(r.pts_left_match.size(),false);
        for(int i:r.idx_from_filtered)r.valid_mask[i]=true;
        r.pts_left_used=r.pts_left_good;r.pts_right_used=r.pts_right_good;
        r.n_kp_left=static_cast<int>(r.kp_left.size());r.n_matched=static_cast<int>(r.pts_left_good.size());
        r.n_template_match=static_cast<int>(r.good_matches.size());
        return r;
    };
    auto initialPose=[&](GroupFlow& g,const std::vector<Eigen::Vector3d>& model) {
        PoseEstimate initial;
        if(cfg.disparity_only||!config_.use_initial_pnp)return initial;
        auto r=makeResult({&g},"");
        if(r.good_matches.size()<4)return initial;
        MatchResult matches;matches.good_matches=r.good_matches;matches.pts_left_match=r.pts_left_match;
        matches.pts_template_match=r.pts_template_match;
        const auto start=Clock::now();
        initial=initial_pnp_.solve(matches,model,camera_.K);
        if(!checkFlowPose(initial,r,model,camera_,false,cfg.initial_reprojection_px,cfg.initial_reprojection_px).success)
            initial=PoseEstimate{};
        initial_ms+=ms(start);return initial;
    };
    auto match=[&](GroupFlow& g,const std::vector<Eigen::Vector3d>& model,const PoseEstimate& initial) {
        std::vector<cv::Point2f> points,seeds;
        std::vector<int> original;
        for(int i=0;i<static_cast<int>(g.raw.pts_left_match.size());++i) {
            if(!cfg.disparity_only && g.model_indices[i]<0)continue;
            original.push_back(i);points.push_back(g.raw.pts_left_match[i]);
            if(cfg.disparity_only)seeds.push_back(points.back());
            else if(initial.success) {
                const Eigen::Vector3d p=camera_.R_rl.transpose()*(initial.R*model[g.model_indices[i]]+initial.t-camera_.t_rl);
                const Eigen::Vector3d u=camera_.K*p;
                seeds.push_back(p.allFinite()&&p.z()>0 ? cv::Point2f(u.x()/u.z(),u.y()/u.z()) : cv::Point2f(-1,-1));
            }
        }
        if(points.size()<static_cast<size_t>(cfg.min_matches)) {
            g.flow.failure_reason="insufficient_left_or_model_points";
            g.flow.diagnostics.resize(g.raw.pts_left_match.size());return;
        }
        if(!context)context=std::make_unique<StereoFlowFrame>(camera_,cfg,lg,rg,search_union);
        auto f=context->match(points,seeds);
        auto compact_diagnostics=std::move(f.diagnostics);
        f.diagnostics.resize(g.raw.pts_left_match.size());
        for(size_t i=0;i<original.size();++i)f.diagnostics[original[i]]=std::move(compact_diagnostics[i]);
        for(auto& i:f.left_indices)i=original[i];
        flow_ms+=f.total_ms;pyramid_ms+=f.pyramid_ms;coarse_ms+=f.coarse_ms;lk_ms+=f.lk_ms;
        check_ms+=f.check_ms;crop_ms+=f.crop_ms;coarse_attempts+=f.coarse_attempts;builds+=f.pyramid_builds;
        full_retry|=f.full_frame_fallback;
        if(f.search_roi.area()>0)search_roi=f.search_roi;
        g.flow=std::move(f);
    };
    auto solve=[&](PipelineResult& r,const std::vector<Eigen::Vector3d>& model,const PoseEstimate& initial,bool motion) {
        PoseEstimate p;
        if(cfg.disparity_only||r.pts_left_good.size()<static_cast<size_t>(std::max(cfg.min_matches,config_.gpnp_min_pts)))return p;
        const auto start=Clock::now();double elapsed=0;
        GPnPOptions options;
        options.initial_source=PoseInitializationSource::InitialPnP;
        options.max_mean_reprojection_px=cfg.max_mean_reprojection_px;
        options.max_point_reprojection_px=cfg.max_point_reprojection_px;
        p=gpnp_solver_.solve(r,model,initial.success?&initial.R:nullptr,initial.success?&initial.t:nullptr,elapsed,options);
        if(p.source!=PoseSource::StereoGpnp ||
            !checkFlowPose(p,r,model,camera_,true,cfg.max_mean_reprojection_px,cfg.max_point_reprojection_px).success)
            p=PoseEstimate{};
        if(p.success&&motion&&seedActive()) {
            PoseSeed seed;seed.R=state_.R_prev;seed.t=state_.t_prev;
            if(!motionGatePass(p,seed,nullptr,false,"DualRoi-LK",true))p=PoseEstimate{};
        }
        gpnp_ms+=ms(start);return p;
    };
    PoseEstimate c0_initial;
    if(!short_circuit) {
        extract(c0,pri);
        if(c0.raw.pts_left_match.size()==dual_bc_tmpl_pts3d_.size())
            for(int i=0;i<static_cast<int>(c0.model_indices.size());++i)c0.model_indices[i]=i;
        c0_initial=initialPose(c0,board);
        match(c0,board,c0_initial);
        result=makeResult({&c0},cfg.disparity_only?"unresolved":"board");
        if(c0.flow.success&&!cfg.disparity_only)pose=solve(result,board,c0_initial,true);
        step("C0:"+(cfg.disparity_only?std::string(c0.flow.success?"OK":"FAIL"):std::string(pose.success?"OK":"FAIL")));
    }
    const bool terminal_c0=cfg.disparity_only&&cfg.disparity_full_chain&&
        terminalDisparityFailure(c0.flow.failure_reason);
    // Pose mode retains the existing P0-dependent annotation association.
    // Disparity-only mode tracks both regions without inventing 3D identities.
    if(!terminal_c0 && !config_.dual_roi_tier1_only && ((cfg.disparity_only&&!short_circuit) || (pose.success&&!dual_c1_tmpl_pts3d_.empty()))) {
        extract(c1,sec);
        if(!cfg.disparity_only) {
            const auto start=Clock::now();
            struct Candidate { double distance;int point,model; };
            std::vector<Candidate> candidates;
            for(int j=n0;j<static_cast<int>(board.size());++j) {
                const Eigen::Vector3d p=pose.R*board[j]+pose.t;
                if(p.z()<=0)continue;
                const Eigen::Vector3d u=camera_.K*p;
                const cv::Point2f predicted(u.x()/u.z(),u.y()/u.z());
                for(int i=0;i<static_cast<int>(c1.raw.pts_left_match.size());++i) {
                    const double distance=cv::norm(predicted-c1.raw.pts_left_match[i]);
                    if(distance<=config_.dual_roi_joint_match_gate_px)candidates.push_back({distance,i,j});
                }
            }
            std::sort(candidates.begin(),candidates.end(),[](auto a,auto b){return a.distance<b.distance;});
            std::vector<bool> used(board.size(),false);
            for(auto c:candidates)if(c1.model_indices[c.point]<0&&!used[c.model]) {
                c1.model_indices[c.point]=c.model;used[c.model]=true;
            }
            association_ms+=ms(start);
        }
        match(c1,board,pose);
        if(cfg.disparity_only) {
            std::vector<const GroupFlow*> active;
            if(c0.flow.success)active.push_back(&c0);
            if(c1.flow.success&&(c0.flow.success||config_.dual_roi_class1_fallback))active.push_back(&c1);
            if(active.empty())active=config_.dual_roi_class1_fallback?std::vector<const GroupFlow*>{&c0,&c1}:std::vector<const GroupFlow*>{&c0};
            result=makeResult(active,"unresolved");result.disparity_success=c0.flow.success||(c1.flow.success&&config_.dual_roi_class1_fallback);
            result.strategy_name=c0.flow.success?(c1.flow.success?"DualRoi_BCJ":"DualRoi_BC"):(config_.dual_roi_class1_fallback?"DualRoi_C1BC":"DualRoi_BC");
            result.is_class1=!c0.flow.success&&c1.flow.success&&config_.dual_roi_class1_fallback;
            result.fallback_used=result.is_class1;
        } else if(c1.flow.success) {
            auto joint=makeResult({&c0,&c1},"board");
            if(joint.pts_left_good.size()>result.pts_left_good.size()) {
                const auto combined=solve(joint,board,pose,true);
                if(combined.success){pose=combined;result=std::move(joint);result.strategy_name="DualRoi_BCJ";}
            }
        }
        step("C1:"+std::string(c1.flow.success?"OK":"FAIL"));
    }
    if(cfg.disparity_only && !short_circuit && config_.dual_roi_tier1_only) {
        // Ablation keeps only class0, matching the legacy tier1_only contract.
        result=makeResult({&c0},"unresolved");result.disparity_success=c0.flow.success;result.strategy_name="DualRoi_BC";
    }
    PoseEstimate fallback_initial;
    std::vector<Eigen::Vector3d> fallback_model;
    if(!config_.dual_roi_tier1_only && config_.dual_roi_class1_fallback &&
        (cfg.disparity_only ? short_circuit : !pose.success)) {
        binary_extractor_->setUseClass1(true);
        extract(c1,sec);
        const auto model=binary_extractor_->templateData().pts_3d;
        fallback_model=model;
        if(!cfg.disparity_only && c1.raw.pts_left_match.size()==model.size())
            for(int i=0;i<static_cast<int>(c1.model_indices.size());++i)c1.model_indices[i]=i;
        fallback_initial=initialPose(c1,model);
        match(c1,model,fallback_initial);
        result=makeResult({&c1},cfg.disparity_only?"unresolved":"class1");
        result.is_class1=true;result.fallback_used=true;result.strategy_name="DualRoi_C1BC";
        if(cfg.disparity_only)result.disparity_success=c1.flow.success;
        else if(c1.flow.success)pose=solve(result,model,fallback_initial,false);
        step("C1BC:"+std::string((cfg.disparity_only?c1.flow.success:pose.success)?"OK":"FAIL"));
    }
    if(cfg.disparity_only && result.strategy_name.empty())result.strategy_name="DualRoi_BC";
    const std::string bc_failure=terminal_c0?c0.flow.failure_reason:
        (terminalDisparityFailure(c1.flow.failure_reason)?c1.flow.failure_reason:
         (short_circuit||result.is_class1?c1.flow.failure_reason:c0.flow.failure_reason));
    bool tried_left_tt=false;
    if(cfg.disparity_only&&cfg.disparity_full_chain&&!result.disparity_success&&
       !config_.dual_roi_tier1_only&&config_.dual_roi_class1_fallback&&
       !terminalDisparityFailure(bc_failure)) {
        struct Guard {TinyTargetExtractor* tt;~Guard(){tt->setUseClass1(false);}} guard{tiny_extractor_.get()};
        tiny_extractor_->setUseClass1(true);
        if(!context)context=std::make_unique<StereoFlowFrame>(camera_,cfg,lg,rg,search_union);
        FlowMatchResult f;
        result=runDisparityExtraction(*tiny_extractor_,*context,lg,lc,rc,tt_roi,1,f);
        result.strategy_name="DualRoi_C1TT";result.fallback_used=true;tried_left_tt=true;
        tt_ms+=result.timing["disparity_extract"];
        flow_ms+=f.total_ms;pyramid_ms+=f.pyramid_ms;coarse_ms+=f.coarse_ms;lk_ms+=f.lk_ms;
        check_ms+=f.check_ms;crop_ms+=f.crop_ms;coarse_attempts+=f.coarse_attempts;builds+=f.pyramid_builds;
        full_retry|=f.full_frame_fallback;if(f.search_roi.area()>0)search_roi=f.search_roi;
        step("C1TT_LK:"+std::string(result.disparity_success?"OK":result.failure_reason));
    }
    if(!cfg.disparity_only && !pose.success && cfg.allow_left_pnp) {
        const PoseEstimate initial=result.is_class1?fallback_initial:c0_initial;
        if(initial.success){pose=initial;pose.source=PoseSource::LeftPnpFallback;step("LeftPnP:OK");}
    }
    const bool ok=cfg.disparity_only?result.disparity_success:pose.success;
    if(!ok&&!tried_left_tt)result.failure_reason=bc_failure;
    if(!ok&&result.failure_reason.empty())result.failure_reason="dual_flow_pose_failed";
    result.stereo_method="LK";result.disparity_only=cfg.disparity_only;
    if(!cfg.disparity_only)result.pose_model_points=result.is_class1?fallback_model:board;
    if(result.strategy_name.empty())result.strategy_name="DualRoi_BC";
    result.is_first_frame=!state_.has_cache;result.left_color=lc;result.right_color=rc;
    result.stereo_match_ms=flow_ms;result.flow_coarse_attempts=coarse_attempts;result.flow_pyramid_builds=builds;
    result.flow_roi=search_roi;result.flow_full_fallback=full_retry;
    result.dual_c0_pairs=static_cast<int>(std::count(result.flow_regions.begin(),result.flow_regions.end(),0));
    result.dual_c1_pairs=static_cast<int>(std::count(result.flow_regions.begin(),result.flow_regions.end(),1));
    result.extract_ms=bc_ms+tt_ms+flow_ms+association_ms;result.pnp_ms=initial_ms+gpnp_ms;
    result.timing={{"bc",bc_ms},{"tt",tt_ms},{"flow_crop",crop_ms},{"flow_pyramid",pyramid_ms},{"flow_coarse",coarse_ms},
                   {"flow_lk",lk_ms},{"flow_check",check_ms},{"annotation",association_ms},
                   {"initial_pnp",initial_ms},{"gpnp",gpnp_ms}};
    step(ok?"FRAME:OK":"FRAME:FAIL");result.route_detail=route;
    if(verbose_console_)std::cout<<"[DualFlow] "<<route<<'\n';
    if(cfg.disparity_only) {
        if(cfg.disparity_full_chain)writeDisparityResult(result);
        else writeDisparity(result,output_dir_,current_frame_);
    }
    if(visualize&&visualize_detailed_&&!output_dir_.empty()) {
        const std::string suffix="_f"+std::to_string(current_frame_)+".png";
        for(const auto* g:{&c0,&c1}) {
            cv::Mat diagnostic(std::max(1,static_cast<int>(g->flow.diagnostics.size()))*24+32,1100,CV_8UC3,{25,25,25});
            for(size_t i=0;i<g->flow.diagnostics.size();++i) {
                const auto& d=g->flow.diagnostics[i];std::ostringstream row;
                row<<g->region<<':'<<i<<' '<<d.reason<<" FB="<<d.fb_error<<" epi="<<d.epipolar_error<<" NCC="<<d.ncc<<" plane="<<d.planar_error_px;
                cv::putText(diagnostic,row.str(),{8,25+static_cast<int>(i)*24},cv::FONT_HERSHEY_SIMPLEX,.45,
                    d.accepted?cv::Scalar(100,255,100):cv::Scalar(100,100,255),1);
            }
            panel(output_dir_+"/dual_roi_c"+std::to_string(g->region)+"_flow_status"+suffix,diagnostic,
                  "Dual-ROI LK | region="+std::to_string(g->region)+" | "+g->flow.failure_reason);
        }
        std::vector<cv::Mat> binaries,stages;
        for(const auto* g:{&c0,&c1}) {
            if(!g->binary.empty()) {
                auto b=color(g->binary);cv::resize(b,b,{300,300});binaries.push_back(b);
            }
            for(const auto& stage:g->stages) {
                auto b=color(stage.second);cv::resize(b,b,{240,240});
                cv::copyMakeBorder(b,b,25,0,0,0,cv::BORDER_CONSTANT,{25,25,25});
                cv::putText(b,"c"+std::to_string(g->region)+" "+stage.first,{3,18},cv::FONT_HERSHEY_SIMPLEX,.4,{255,255,255},1);
                stages.push_back(b);
            }
        }
        if(!binaries.empty()){cv::Mat b;cv::hconcat(binaries,b);panel(output_dir_+"/dual_roi_binary"+suffix,b,"Left BC binaries; right uses LK");}
        if(!stages.empty()){cv::Mat b;cv::hconcat(stages,b);panel(output_dir_+"/dual_roi_bc_process"+suffix,b,"Left BC processing");}
    }
    return result;
}

PipelineResult StereoTracker::runDualTinyDisparity(const cv::Mat& left,const cv::Mat& right,
                                                 const RoiGroup& left_group,const RoiGroup& right_group,bool visualize) {
    PipelineResult r;r.stereo_method="LK";r.disparity_only=true;r.is_class1=true;r.fallback_used=true;
    r.strategy_name="DualRoi_C1TT";r.model_frame="unresolved";
    auto [lc,lg]=loadImage(left,visualize);auto [rc,rg]=loadImage(right,visualize);
    if(lg.empty()||rg.empty()||lg.size()!=rg.size()){r.failure_reason="invalid_stereo_images";return r;}
    const RoiRect lr=left_group.is_dual?left_group.secondary:(left_group.is_class1?left_group.primary:RoiRect{});
    const RoiRect rr=right_group.is_dual?right_group.secondary:(right_group.is_class1?right_group.primary:RoiRect{});
    const auto a=bounded(lr,lg.size(),config_.dual_roi_secondary_expand);
    const auto b=bounded(rr,rg.size(),config_.dual_roi_secondary_expand);
    if(a.empty()||b.empty()){r.failure_reason="no_class1_roi_for_tt";return r;}
    struct Guard { TinyTargetExtractor* tt;~Guard(){tt->setUseClass1(false);} } guard{tiny_extractor_.get()};
    tiny_extractor_->setUseClass1(true);
    const auto start=Clock::now();
    auto extracted=tiny_extractor_->extract(lg(a),rg(b),lc.empty()?cv::Mat():lc(a),rc.empty()?cv::Mat():rc(b));
    offsetResultToOriginal(extracted,{double(a.x),double(a.y)},{double(b.x),double(b.y)},lc,rc);
    const double extract_ms=ms(start);
    r.pts_left_match=extracted.pts_left_match;r.kp_left=extracted.kp_left;r.n_kp_left=extracted.n_kp_left;
    if(extracted.pts_left_good.size()!=4||extracted.pts_right_good.size()!=4){r.failure_reason="legacy_tt_no_stereo_pairs";r.extract_ms=extract_ms;return r;}
    auto tt_config=config_.stereo_flow;tt_config.min_matches=4;
    auto checked=StereoFlowMatcher(camera_,tt_config).match(lg,rg,r.pts_left_match,extracted.pts_right_good,a);
    r.pts_left_good=checked.left_points;r.pts_right_good=checked.right_points;r.idx_from_filtered=checked.left_indices;
    r.pts_left_used=r.pts_left_good;r.pts_right_used=r.pts_right_good;
    r.n_matched=static_cast<int>(r.pts_left_good.size());r.disparity_success=checked.success;r.failure_reason=checked.failure_reason;
    r.flow_planar_status=checked.planar_status;r.flow_planar_rejected=checked.planar_rejected;r.flow_planar_ms=checked.planar_ms;
    for(size_t i=0;i<r.pts_left_good.size();++i) {
        const double d=r.pts_left_good[i].x-r.pts_right_good[i].x;
        r.disparity.push_back(d);r.dx_filtered.push_back(d);r.flow_regions.push_back(1);
        r.flow_original_indices.push_back(r.idx_from_filtered[i]);r.flow_template_indices.push_back(-1);
    }
    r.dual_c1_pairs=r.n_matched;r.flow_roi=checked.search_roi;r.flow_full_fallback=checked.full_frame_fallback;
    r.flow_coarse_attempts=checked.coarse_attempts;r.flow_pyramid_builds=checked.pyramid_builds;
    r.stereo_match_ms=checked.total_ms;r.extract_ms=extract_ms+checked.total_ms;
    r.timing={{"tt",extract_ms},{"flow",checked.total_ms}};
    // This branch contains no pose solver or pose-cache update.
    if(r.disparity_success)writeDisparity(r,output_dir_,current_frame_);
    return r;
}

void StereoTracker::renderDualFlowResult(const PipelineResult& result,const cv::Mat& left,
                                       const cv::Mat& right,const RoiGroup& group) const {
    if(left.empty()||output_dir_.empty())return;
    cv::Mat l=color(left),r=right.empty()?cv::Mat(l.size(),CV_8UC3,cv::Scalar(25,25,25)):color(right);
    const auto pri=bounded(group.primary,l.size()),sec=bounded(group.secondary,l.size(),config_.dual_roi_secondary_expand);
    if(!pri.empty())cv::rectangle(l,pri,{0,220,255},2);
    if(!sec.empty())cv::rectangle(l,sec,{80,255,80},2);
    if(result.flow_roi.area()>0)cv::rectangle(r,result.flow_roi&cv::Rect(0,0,r.cols,r.rows),{150,150,150},1);
    if(result.success&&result.pose_source==PoseSource::LeftPnpFallback) {
        for(const auto& m:result.good_matches) {
            if(m.queryIdx<0||m.queryIdx>=static_cast<int>(result.pts_left_match.size()))continue;
            const auto p=result.pts_left_match[m.queryIdx];
            cv::circle(l,p,4,{80,255,80},1);
            cv::putText(l,"L:"+std::to_string(m.queryIdx),p+cv::Point2f(4,-4),cv::FONT_HERSHEY_SIMPLEX,.4,{80,255,80},1);
        }
    }
    for(size_t i=0;i<result.pts_left_good.size()&&i<result.pts_right_good.size();++i) {
        const int region=i<result.flow_regions.size()?result.flow_regions[i]:(static_cast<int>(i)<result.dual_c0_pairs?0:1);
        const int original=i<result.flow_original_indices.size()?result.flow_original_indices[i]:static_cast<int>(i)-(region?result.dual_c0_pairs:0);
        const cv::Scalar c=region?cv::Scalar(80,255,80):cv::Scalar(0,220,255);
        for(auto view:std::vector<std::pair<cv::Mat*,cv::Point2f>>{{&l,result.pts_left_good[i]},{&r,result.pts_right_good[i]}}) {
            cv::circle(*view.first,view.second,4,c,1);
            int label=original;
            if(config_.stereo_robustness.enabled&&i<result.flow_template_indices.size()) {
                label=result.flow_template_indices[i];
                if(region&&group.is_dual&&!result.is_class1&&label>=int(dual_bc_tmpl_pts3d_.size()))label-=int(dual_bc_tmpl_pts3d_.size());
            }
            cv::putText(*view.first,std::to_string(region)+":"+std::to_string(label),view.second+cv::Point2f(4,-4),
                        cv::FONT_HERSHEY_SIMPLEX,.4,c,1);
        }
    }
    cv::Mat paired(std::max(l.rows,r.rows),l.cols+r.cols,CV_8UC3,cv::Scalar(25,25,25));
    l.copyTo(paired(cv::Rect(0,0,l.cols,l.rows)));r.copyTo(paired(cv::Rect(l.cols,0,r.cols,r.rows)));
    for(size_t i=0;i<result.pts_left_good.size()&&i<result.pts_right_good.size();++i)
        cv::line(paired,result.pts_left_good[i],result.pts_right_good[i]+cv::Point2f(l.cols,0),{80,180,255},1);
    const std::string suffix="_f"+std::to_string(current_frame_)+".png";
    std::ostringstream title;title<<result.strategy_name<<" | "<<result.stereo_method<<" | c0="<<result.dual_c0_pairs
        <<" c1="<<result.dual_c1_pairs<<" | "<<(result.disparity_only?"disparity only, PnP skipped":result.success?"pose OK":"pose FAIL")
        <<" | "<<poseSourceName(result.pose_source);
    cv::Mat axes=l.clone();
    cv::Mat reproj=l.clone();
    if(result.success&&!result.disparity_only) {
        cv::Mat k=(cv::Mat_<double>(3,3)<<camera_.K(0,0),camera_.K(0,1),camera_.K(0,2),camera_.K(1,0),camera_.K(1,1),camera_.K(1,2),camera_.K(2,0),camera_.K(2,1),camera_.K(2,2));
        cv::Mat rotation(3,3,CV_64F),rv,tv=(cv::Mat_<double>(3,1)<<result.t.x(),result.t.y(),result.t.z());
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)rotation.at<double>(i,j)=result.R(i,j);
        cv::Rodrigues(rotation,rv);std::vector<cv::Point2d> uv;
        const double axis=config_.stereo_robustness.enabled&&result.is_class1?5:50;
        cv::projectPoints(std::vector<cv::Point3d>{{0,0,0},{axis,0,0},{0,axis,0},{0,0,utils::upNormalZ(axis, result.t.z())}},rv,tv,k,cv::Mat(),uv);
        for(int i=1;i<3;++i)cv::line(axes,uv[0],uv[i],i==1?cv::Scalar(0,0,255):cv::Scalar(0,255,0),2);
        utils::drawUpNormal(axes,uv[0],uv[3]);
        if(config_.stereo_robustness.enabled&&result.pose_source==PoseSource::StereoGpnp) {
            cv::Mat right_axes=r.clone();
            const Eigen::Matrix3d right_rotation=camera_.R_rl.transpose()*result.R;
            const Eigen::Vector3d right_translation=camera_.R_rl.transpose()*(result.t-camera_.t_rl);
            for(int i=0;i<3;++i){tv.at<double>(i)=right_translation(i);for(int j=0;j<3;++j)rotation.at<double>(i,j)=right_rotation(i,j);}
            cv::Rodrigues(rotation,rv);
            cv::projectPoints(std::vector<cv::Point3d>{{0,0,0},{axis,0,0},{0,axis,0},{0,0,utils::upNormalZ(axis,right_translation.z())}},rv,tv,k,cv::Mat(),uv);
            for(int i=1;i<3;++i)cv::line(right_axes,uv[0],uv[i],i==1?cv::Scalar(0,0,255):cv::Scalar(0,255,0),2);
            utils::drawUpNormal(right_axes,uv[0],uv[3]);
            cv::Mat both;cv::hconcat(axes,right_axes,both);
            panel(output_dir_+"/stereo_axes_lr"+suffix,both,"Actual observations + axes | left / right camera projection | -Z up");
        }
        std::vector<int> model_index(result.pts_left_match.size(),-1);
        for(const auto& m:result.good_matches)if(m.queryIdx>=0&&m.queryIdx<static_cast<int>(model_index.size()))model_index[m.queryIdx]=m.trainIdx;
        auto projected_indices=result.idx_from_filtered;
        if(result.pose_source==PoseSource::LeftPnpFallback) {
            projected_indices.clear();
            for(const auto& m:result.good_matches)projected_indices.push_back(m.queryIdx);
        }
        for(int i:projected_indices) {
            if(i<0||i>=static_cast<int>(model_index.size()))continue;
            const int j=model_index[i];
            if(j<0||j>=static_cast<int>(result.pose_model_points.size()))continue;
            const Eigen::Vector3d p=result.R*result.pose_model_points[j]+result.t;
            if(p.z()<=0)continue;
            const Eigen::Vector3d q=camera_.K*p;
            const cv::Point2f projected(q.x()/q.z(),q.y()/q.z());
            cv::circle(reproj,projected,3,{80,255,80},-1);
            cv::line(reproj,result.pts_left_match[i],projected,{255,255,255},1);
        }
    }
    panel(output_dir_+"/stereo"+suffix,result.disparity_only||!result.success?paired:axes,title.str());
    if(visualize_detailed_) {
        panel(output_dir_+"/dual_roi_left"+suffix,l,title.str());panel(output_dir_+"/dual_roi_right"+suffix,r,"Right observations | rectangle: search area");
        panel(output_dir_+"/dual_roi_correspondence"+suffix,paired,title.str());
        panel(output_dir_+"/dual_roi_reproj"+suffix,reproj,result.disparity_only?"PnP skipped":result.success?"Observed corners and model reprojection":"Pose rejected; no projection");
    }
}
} // namespace gpnp
