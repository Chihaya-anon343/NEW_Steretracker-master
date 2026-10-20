#include "tracker/StereoTracker.hpp"
#include "common/StereoRobustness.hpp"
#include "common/TinyTargetPolicy.hpp"
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/SVD>
#include <chrono>
#include <sstream>

namespace gpnp {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
cv::Mat cvK(const Eigen::Matrix3d& k) {
    cv::Mat c(3,3,CV_64F);for(int i=0;i<3;++i)for(int j=0;j<3;++j)c.at<double>(i,j)=k(i,j);return c;
}
// No correspondence is inferred from vector position after any filtering.
PipelineResult joined(const PipelineResult& a,const PipelineResult& b) {
    auto r=a;const int base=int(r.pts_left_match.size());
    r.pts_left_match.insert(r.pts_left_match.end(),b.pts_left_match.begin(),b.pts_left_match.end());
    r.kp_left.insert(r.kp_left.end(),b.kp_left.begin(),b.kp_left.end());
    r.pts_template_match.insert(r.pts_template_match.end(),b.pts_template_match.begin(),b.pts_template_match.end());
    for(auto m:b.good_matches){m.queryIdx+=base;r.good_matches.push_back(m);}
    for(size_t k=0;k<b.idx_from_filtered.size();++k) {
        r.idx_from_filtered.push_back(base+b.idx_from_filtered[k]);
        r.pts_left_good.push_back(b.pts_left_good[k]);r.pts_right_good.push_back(b.pts_right_good[k]);
        r.disparity.push_back(b.disparity[k]);r.dx_filtered.push_back(b.disparity[k]);
        r.flow_regions.push_back(b.flow_regions[k]);r.flow_original_indices.push_back(b.flow_original_indices[k]);
        r.flow_template_indices.push_back(b.flow_template_indices[k]);
    }
    r.valid_mask.insert(r.valid_mask.end(),b.valid_mask.begin(),b.valid_mask.end());
    r.pts_left_used=r.pts_left_good;r.pts_right_used=r.pts_right_good;
    r.n_kp_left=int(r.kp_left.size());r.n_matched=int(r.pts_left_good.size());r.n_template_match=int(r.good_matches.size());
    r.dual_c0_pairs=a.dual_c0_pairs+b.dual_c0_pairs;r.dual_c1_pairs=a.dual_c1_pairs+b.dual_c1_pairs;
    r.flow_planar_rejected+=b.flow_planar_rejected;r.flow_planar_ms+=b.flow_planar_ms;
    r.flow_planar_status+=";"+b.flow_planar_status;
    return r;
}
}

PipelineResult StereoTracker::processRobustStereo(const cv::Mat& left,const cv::Mat& right,
        const RoiGroup& group,bool visualize,PoseEstimate& final_pose) {
    const auto& cfg=config_.stereo_flow;
    const auto& policy=config_.stereo_robustness;
    PipelineResult result;result.stereo_method="LK";result.disparity_only=cfg.disparity_only;
    auto [lc,lg]=loadImage(left,visualize);auto [rc,rg]=loadImage(right,visualize);
    if(lg.empty()||rg.empty()||lg.size()!=rg.size()||lg.type()!=CV_8UC1||rg.type()!=CV_8UC1) {
        result.failure_reason="invalid_stereo_images";result.route_detail="RobustStereo:invalid_images";return result;
    }
    const cv::Rect image(0,0,lg.cols,lg.rows);
    auto rect=[&](const RoiRect& r,int pad) {
        if(!r.valid())return cv::Rect{};
        return cv::Rect(r.x-pad,r.y-pad,r.width+2*pad,r.height+2*pad)&image;
    };
    const auto primary=rect(group.primary,0),secondary=rect(group.secondary,config_.dual_roi_secondary_expand);
    if(primary.empty()||(group.is_dual&&secondary.empty())) {
        result.failure_reason="invalid_left_roi";result.route_detail="RobustStereo:invalid_roi";return result;
    }
    configureStrategyChain(primary.area(),group.is_class1);
    for(auto* e:{robust_bc_.get(),robust_dual_c0_.get(),robust_dual_c1_.get()})e->setDebugCapture(visualize);
    robust_tt_->setDebugCapture(visualize);
    struct ClassGuard {
        BinaryCornerExtractor* bc;TinyTargetExtractor* tt;
        ~ClassGuard(){bc->setUseClass1(false);tt->setUseClass1(false);}
    } guard{robust_bc_.get(),robust_tt_.get()};
    if(group.is_dual)prepareDualBcTemplate();
    std::vector<Eigen::Vector3d> board=dual_bc_tmpl_pts3d_;
    board.insert(board.end(),robust_c1_model_.begin(),robust_c1_model_.end());
    const int n0=int(dual_bc_tmpl_pts3d_.size());
    const bool short_circuit=group.is_dual&&!config_.dual_roi_tier1_only&&config_.dual_roi_class1_fallback&&
        config_.dual_roi_primary_span_ratio>0&&primary.width>=lg.cols*config_.dual_roi_primary_span_ratio&&
        primary.height>=lg.rows*config_.dual_roi_primary_span_ratio;
    const auto search=short_circuit?rect(group.secondary,std::max(binary_roi_pad_,tiny_roi_pad_)+config_.dual_roi_secondary_expand):
        (rect(group.primary,std::max(binary_roi_pad_,tiny_roi_pad_))|(group.is_dual?rect(group.secondary,tiny_roi_pad_):cv::Rect{}));
    StereoFlowFrame context(camera_,cfg,lg,rg,search);
    double extraction_ms=0,flow_ms=0,pnp_ms=0;int builds=0,coarse=0;bool full_retry=false;
    std::map<std::string,double> timing;cv::Rect flow_roi;
    std::string route="RobustStereo";
    auto step=[&](const std::string& s){route+=" -> "+s;};
    auto terminal=[&](const PipelineResult& r){return terminalDisparityFailure(r.failure_reason);};
    auto accepted=[&](const PipelineResult& r,const PoseEstimate& p){return cfg.disparity_only?r.disparity_success:p.success;};

    // Candidate generation is independent of the right image. Every candidate
    // then receives the SAME right observations and the SAME stereo acceptance.
    auto solve=[&](PipelineResult& r,const std::vector<Eigen::Vector3d>& model,
                   const PoseEstimate* preferred=nullptr) {
        PoseEstimate winner;
        if(cfg.disparity_only)return winner;
        if(r.pose_model_points.empty())return winner;
        const int support=std::max({cfg.min_matches,config_.gpnp_min_pts,r.pose_min_inliers});
        if(int(r.pts_left_good.size())<support) {
            r.failure_reason="insufficient_pose_support";
            if(cfg.allow_left_pnp) {
                std::vector<cv::Point2f> px;std::vector<Eigen::Vector3d> object;
                for(auto m:r.good_matches)if(m.queryIdx>=0&&m.queryIdx<int(r.pts_left_match.size())&&m.trainIdx>=0&&m.trainIdx<int(model.size())) {
                    px.push_back(r.pts_left_match[m.queryIdx]);object.push_back(model[m.trainIdx]);
                }
                const auto start=Clock::now();winner=mono_pnp_.solve(px,object,camera_.K,nullptr,0,support);pnp_ms+=ms(start);
                if(winner.success)winner.source=PoseSource::LeftPnpFallback;
            }
            return winner;
        }
        const auto begin=Clock::now();
        std::vector<cv::Point3d> object;std::vector<cv::Point2f> pixel;
        std::vector<int> lut(r.pts_left_match.size(),-1);
        for(auto m:r.good_matches)if(m.queryIdx>=0&&m.queryIdx<int(lut.size()))lut[m.queryIdx]=m.trainIdx;
        for(int q:r.idx_from_filtered) {
            if(q<0||q>=int(lut.size())||lut[q]<0||lut[q]>=int(model.size())) {
                r.failure_reason="invalid_model_index";pnp_ms+=ms(begin);return winner;
            }
            auto p=model[lut[q]];object.emplace_back(p.x(),p.y(),p.z());pixel.push_back(r.pts_left_match[q]);
        }
        struct Candidate {PoseEstimate pose;PoseInitializationSource source;};
        std::vector<Candidate> candidates;
        if(preferred&&preferred->success)candidates.push_back({*preferred,PoseInitializationSource::Provided});
        // A metric stereo seed avoids inheriting a small planar PnP depth bias
        // through the existing GPnP translation prior. No scale is estimated.
        if(r.pts_left_good.size()>=4) {
            cv::Mat pl=cv::Mat::zeros(3,4,CV_64F),pr=cv::Mat::zeros(3,4,CV_64F);
            const Eigen::Matrix3d rr=camera_.R_rl.transpose();const Eigen::Vector3d rt=-rr*camera_.t_rl;
            for(int i=0;i<3;++i){pl.at<double>(i,i)=1;pr.at<double>(i,3)=rt(i);for(int j=0;j<3;++j)pr.at<double>(i,j)=rr(i,j);}
            cv::Mat hp;cv::triangulatePoints(cvK(camera_.K)*pl,cvK(camera_.K)*pr,r.pts_left_good,r.pts_right_good,hp);
            hp.convertTo(hp,CV_64F);std::vector<Eigen::Vector3d> metric;bool valid=true;
            Eigen::Vector3d mc=Eigen::Vector3d::Zero(),pc=Eigen::Vector3d::Zero();
            for(int i=0;i<hp.cols;++i) {
                const double w=hp.at<double>(3,i);if(!std::isfinite(w)||std::abs(w)<1e-12){valid=false;break;}
                Eigen::Vector3d p(hp.at<double>(0,i)/w,hp.at<double>(1,i)/w,hp.at<double>(2,i)/w);
                if(!p.allFinite()||p.z()<=0||(rr*p+rt).z()<=0){valid=false;break;}
                metric.push_back(p);pc+=p;mc+=Eigen::Vector3d(object[i].x,object[i].y,object[i].z);
            }
            if(valid&&metric.size()==object.size()) {
                pc/=metric.size();mc/=metric.size();Eigen::Matrix3d covariance=Eigen::Matrix3d::Zero();
                for(size_t i=0;i<metric.size();++i)covariance+=(Eigen::Vector3d(object[i].x,object[i].y,object[i].z)-mc)*(metric[i]-pc).transpose();
                Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance,Eigen::ComputeFullU|Eigen::ComputeFullV);
                if(svd.singularValues()(1)>1e-8) {
                    Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant()<0?-1:1;
                    PoseEstimate p;p.R=svd.matrixV()*sign*svd.matrixU().transpose();p.t=pc-p.R*mc;p.success=true;
                    if(p.t.norm()<20000)candidates.push_back({p,PoseInitializationSource::StereoTriangulation});
                }
            }
        }
        if(config_.use_initial_pnp) {
            MatchResult m;m.good_matches=r.good_matches;m.pts_left_match=r.pts_left_match;m.pts_template_match=r.pts_template_match;
            auto p=initial_pnp_.solve(m,model,camera_.K);if(p.success)candidates.push_back({p,PoseInitializationSource::InitialPnP});
        }
        std::vector<cv::Mat> rotations,translations;
        try {if(config_.use_initial_pnp)cv::solvePnPGeneric(object,pixel,cvK(camera_.K),cv::Mat(),rotations,translations,false,cv::SOLVEPNP_IPPE);}
        catch(const cv::Exception&) { rotations.clear(); }
        for(size_t k=0;k<rotations.size();++k) {
            cv::Mat rotation;cv::Rodrigues(rotations[k],rotation);PoseEstimate p;p.success=true;
            for(int i=0;i<3;++i){p.t(i)=translations[k].at<double>(i);for(int j=0;j<3;++j)p.R(i,j)=rotation.at<double>(i,j);}
            candidates.push_back({p,PoseInitializationSource::PlanarIppe});
        }
        double best=std::numeric_limits<double>::infinity();GPNPMonitor monitor;
        std::string failure="no_initial_pose";
        for(const auto& candidate:candidates) {
            const auto& seed=candidate.pose;
            double elapsed=0;GPnPOptions options;options.initial_source=candidate.source;
            options.max_mean_reprojection_px=cfg.max_mean_reprojection_px;options.max_point_reprojection_px=cfg.max_point_reprojection_px;
            auto p=gpnp_solver_.solve(r,model,&seed.R,&seed.t,elapsed,options);
            const auto check=checkFlowPose(p,r,model,camera_,true,cfg.max_mean_reprojection_px,cfg.max_point_reprojection_px);
            if(!p.success||p.source!=PoseSource::StereoGpnp||!check.success) {
                failure=r.last_gpnp_monitor.failure_reason.empty()?check.reason:r.last_gpnp_monitor.failure_reason;continue;
            }
            const double score=check.mean_left_px+check.mean_right_px;
            if(score<best){best=score;winner=p;monitor=r.last_gpnp_monitor;}
        }
        pnp_ms+=ms(begin);
        if(winner.success) {
            r.last_gpnp_monitor=monitor;r.stereo_success=true;r.failure_reason.clear();
        } else r.failure_reason=failure.empty()?"stereo_gpnp_failed":failure;
        // Single-camera output is explicitly opt-in and never stereo_success.
        if(!winner.success&&cfg.allow_left_pnp&&!candidates.empty()) {
            double e=std::numeric_limits<double>::infinity();
            for(const auto& candidate:candidates) {
                auto p=candidate.pose;
                auto check=checkFlowPose(p,r,model,camera_,false,cfg.initial_reprojection_px,cfg.initial_reprojection_px);
                if(check.success&&check.mean_left_px<e){e=check.mean_left_px;winner=p;winner.source=PoseSource::LeftPnpFallback;}
            }
        }
        return winner;
    };

    auto run=[&](FeatureExtractor& ext,const cv::Rect& roi,int region,
                 const std::vector<Eigen::Vector3d>& model,int model_offset,const std::string& name) {
        const auto begin=Clock::now();
        auto r=ext.extractMono(lg(roi),lc.empty()?cv::Mat():lc(roi));
        const double extraction=ms(begin);extraction_ms+=extraction;
        const bool extracted=r.success;r.success=false;r.stereo_method="LK";r.disparity_only=cfg.disparity_only;
        r.strategy_name=name;r.is_class1=region==1;r.pose_model_points.clear();
        r.model_frame=cfg.disparity_only?"unresolved":config_.template_geometry.enabled?"class0":group.is_dual?"board":region?"class1":"class0";
        r.left_color=lc;r.right_color=rc;r.left_roi_offset_x=roi.x;r.left_roi_offset_y=roi.y;
        r.timing.clear();
        for(auto& p:r.pts_left_match)p+=cv::Point2f(roi.x,roi.y);
        for(auto& p:r.kp_left)p.pt+=cv::Point2f(roi.x,roi.y);
        for(auto& m:r.good_matches)m.trainIdx+=model_offset;
        const bool tiny=ext.strategyType()==StrategyType::TinyTarget;
        if(!cfg.disparity_only&&group.is_dual&&!tiny&&
           ((&ext==robust_dual_c0_.get()&&dual_bc_tmpl_pts3d_.size()!=10)||
            (&ext==robust_dual_c1_.get()&&robust_c1_model_.size()!=10))) {
            r.failure_reason="dual_annotation_model_count_mismatch";return r;
        }
        if(!extracted||r.pts_left_match.size()<4) {
            if(r.failure_reason.empty())r.failure_reason=tiny?"tt_extraction_failed":"bc_extraction_failed";
            return r;
        }
        if(tiny&&policy.tiny_target) {
            const auto& q=robust_tt_->lastQuality();TinyTargetRoiEvidence evidence;
            evidence.extracted=true;evidence.finite=true;evidence.points=int(r.pts_left_match.size());
            evidence.legacy_iou=q.legacy_iou;evidence.direction_iou=q.direction_iou;evidence.direction_gap=q.direction_gap;
            evidence.span_x=q.span_x_px;evidence.span_y=q.span_y_px;evidence.geometry_valid=q.geometry_valid;
            evidence.emitted_direction_consistent=q.direction_reliable;
            const auto decision=evaluateTinyTargetRoi(evidence,config_.roi_selection,region==1,roi.width,roi.height);
            if(!decision.accepted){r.failure_reason=decision.reason;return r;}
        }
        // Validate model indices even when only a subset of the ten IDs remains.
        if(!cfg.disparity_only) {
            auto copy=r;copy.success=extracted;
            if(!monoDualMatchesValid(copy,model)){r.failure_reason="invalid_extraction_model_ids";return r;}
            r.pose_model_points=model;
        }
        const int minimum=tiny?4:std::max(cfg.min_matches,r.pose_min_inliers);
        auto f=context.match(r.pts_left_match,r.pts_left_match,minimum);
        if(visualize&&visualize_detailed_&&!output_dir_.empty()) {
            const std::string suffix="_f"+std::to_string(current_frame_)+".png";
            const std::string prefix=group.is_dual?"dual_roi_c"+std::to_string(region):tiny?"tiny_target":"binary_corner";
            cv::Mat status(int(f.diagnostics.size())*25+40,1120,CV_8UC3,cv::Scalar(25,25,25));
            for(size_t i=0;i<f.diagnostics.size();++i) {
                const auto& d=f.diagnostics[i];std::ostringstream line;
                const int id=i<r.good_matches.size()?r.good_matches[i].trainIdx:-1;
                line<<"ID="<<id<<" "<<d.reason<<" FB="<<d.fb_error<<" epi="<<d.epipolar_error<<" NCC="<<d.ncc<<" plane="<<d.planar_error_px;
                cv::putText(status,line.str(),{8,24+int(i)*25},cv::FONT_HERSHEY_SIMPLEX,.43,d.accepted?cv::Scalar(100,255,100):cv::Scalar(100,100,255),1);
            }
            utils::AsyncImageSaver::write(output_dir_+"/"+prefix+"_flow_status"+suffix,status);
            cv::Mat binary;
            if(tiny)binary=robust_tt_->lastLeftDebug().super_binary;
            else if(auto* bc=dynamic_cast<BinaryCornerExtractor*>(&ext))binary=bc->lastLeftBinary();
            if(!binary.empty())utils::AsyncImageSaver::write(output_dir_+"/"+prefix+"_binary"+suffix,binary);
            if(tiny) {
                const auto& q=robust_tt_->lastQuality();cv::Mat panel(85,1120,CV_8UC3,cv::Scalar(25,25,25));std::ostringstream line;
                line<<q.status<<" legacy="<<q.legacy_iou<<" direction="<<q.direction_iou<<" gap="<<q.direction_gap;
                cv::putText(panel,line.str(),{8,28},cv::FONT_HERSHEY_SIMPLEX,.5,{240,240,240},1);
                line.str("");line<<"physical quadrant="<<q.quadrant<<" refined="<<q.refined_count<<" max shift="<<q.max_refine_shift_px<<" original px";
                cv::putText(panel,line.str(),{8,60},cv::FONT_HERSHEY_SIMPLEX,.5,{240,240,240},1);
                utils::AsyncImageSaver::write(output_dir_+"/"+prefix+"_quality"+suffix,panel);
            }
        }
        flow_ms+=f.total_ms;builds+=f.pyramid_builds;coarse+=f.coarse_attempts;full_retry|=f.full_frame_fallback;
        if(f.search_roi.area()>0)flow_roi=f.search_roi;
        timing["flow_pyramid"]+=f.pyramid_ms;timing["flow_coarse"]+=f.coarse_ms;
        timing["flow_lk"]+=f.lk_ms;timing["flow_check"]+=f.check_ms;timing["flow_crop"]+=f.crop_ms;
        r.pts_left_good=f.left_points;r.pts_right_good=f.right_points;r.idx_from_filtered=f.left_indices;
        r.pts_left_used=f.left_points;r.pts_right_used=f.right_points;
        r.flow_planar_status=f.planar_status;r.flow_planar_rejected=f.planar_rejected;r.flow_planar_ms=f.planar_ms;
        r.disparity_success=f.success;r.failure_reason=f.failure_reason;r.n_matched=int(f.left_points.size());
        r.dual_c0_pairs=region==0?r.n_matched:0;r.dual_c1_pairs=region==1?r.n_matched:0;
        std::vector<int> ids(r.pts_left_match.size(),-1);
        for(auto m:r.good_matches)if(m.queryIdx>=0&&m.queryIdx<int(ids.size()))ids[m.queryIdx]=m.trainIdx;
        r.valid_mask.assign(r.pts_left_match.size(),false);
        for(size_t k=0;k<f.left_indices.size();++k) {
            const int q=f.left_indices[k];r.valid_mask[q]=true;
            const double d=f.left_points[k].x-f.right_points[k].x;r.disparity.push_back(d);r.dx_filtered.push_back(d);
            r.flow_regions.push_back(region);r.flow_original_indices.push_back(q);r.flow_template_indices.push_back(ids[q]);
        }
        return r;
    };

    auto chain=[&](const RoiRect& roi,int region,bool dual_fallback) {
        PipelineResult r;PoseEstimate p;
        const int tiny_threshold=region&&tiny_max_area_class1_>0?tiny_max_area_class1_:tiny_max_area_;
        const bool tiny=dual_fallback?roi.width*roi.height<=tiny_threshold:
            (!group.is_dual&&extractor_->strategyType()==StrategyType::TinyTarget);
        BinaryCornerExtractor* bc=dual_fallback?robust_dual_c1_.get():robust_bc_.get();
        if(!dual_fallback)bc->setUseClass1(region==1);
        if(!tiny) {
            const auto& model=dual_fallback?robust_c1_model_:bc->templateData().pts_3d;
            r=run(*bc,rect(roi,binary_roi_pad_),region,model,0,dual_fallback?"DualRoi_C1BC":"BinaryCorner");
            if(r.disparity_success||cfg.allow_left_pnp)p=solve(r,model);
            step(std::string(region?"C1BC:":"C0BC:")+(accepted(r,p)?"OK":r.failure_reason));
            if(accepted(r,p)||terminal(r))return std::make_pair(r,p);
            if(cfg.disparity_only&&!cfg.disparity_full_chain)return std::make_pair(r,p);
        } else step(region?"C1BC:skip_tiny":"C0BC:skip_tiny");
        if(cfg.disparity_only&&!cfg.disparity_full_chain) {
            r.failure_reason="tt_requires_disparity_full_chain";return std::make_pair(r,p);
        }
        const auto previous=r.failure_reason;
        robust_tt_->setUseClass1(region==1);
        const auto model=robust_tt_->templateData().pts_3d;
        r=run(*robust_tt_,rect(roi,tiny_roi_pad_),region,model,0,dual_fallback?"DualRoi_C1TT":"TinyTarget");
        r.fallback_used=!tiny||dual_fallback;
        if(r.disparity_success||cfg.allow_left_pnp)p=solve(r,model);
        step(std::string(region?"C1TT:":"C0TT:")+(accepted(r,p)?"OK":r.failure_reason));
        (void)previous;return std::make_pair(r,p);
    };

    if(!group.is_dual) {
        auto pair=chain(group.primary,group.is_class1?1:0,false);result=std::move(pair.first);final_pose=pair.second;
    } else {
        if(short_circuit)step("C0:short_circuit");
        else {
            result=run(*robust_dual_c0_,rect(group.primary,binary_roi_pad_),0,board,0,"DualRoi_BC");
            if(result.disparity_success||cfg.allow_left_pnp)final_pose=solve(result,board);
            step("C0:"+std::string(accepted(result,final_pose)?"OK":result.failure_reason));
            if(accepted(result,final_pose)&&!config_.dual_roi_tier1_only&&!robust_c1_model_.empty()) {
                const int limit=tiny_max_area_class1_>0?tiny_max_area_class1_:tiny_max_area_;
                if(secondary.area()<=limit)step("C1:skip_tiny_keep_primary");
                else {
                    auto c1=run(*robust_dual_c1_,rect(group.secondary,binary_roi_pad_+config_.dual_roi_secondary_expand),1,board,n0,"DualRoi_C1BC");
                    if(c1.disparity_success) {
                        bool duplicate=false;
                        for(auto a:result.pts_right_good)for(auto b:c1.pts_right_good)
                            if(cv::norm(a-b)<cfg.duplicate_distance_px)duplicate=true;
                        if(duplicate)step("C1:keep_primary[duplicate_right_point]");
                        if(!duplicate) {
                        auto joint=joined(result,c1);joint.strategy_name="DualRoi_BCJ";
                        if(cfg.disparity_only){result=std::move(joint);step("Joint:disparity_OK");}
                        else {
                            auto candidate=solve(joint,board,&final_pose);
                            auto reason=candidate.success&&candidate.source==PoseSource::StereoGpnp?
                                (policy.joint_quality_check?checkStereoDualJoint(final_pose,candidate,result,c1,board,camera_,policy.joint_limits):std::string{}):"joint_pose_failed";
                            if(reason.empty()){result=std::move(joint);final_pose=candidate;step("Joint:OK");}
                            else step("Joint:keep_primary["+reason+"]");
                        }
                        }
                    } else step("C1:keep_primary["+c1.failure_reason+"]");
                }
            }
        }
        if(!accepted(result,final_pose)&&!terminal(result)&&!config_.dual_roi_tier1_only&&config_.dual_roi_class1_fallback) {
            auto pair=chain(group.secondary,1,true);result=std::move(pair.first);final_pose=pair.second;result.fallback_used=true;
        }
    }
    if(!cfg.disparity_only&&final_pose.success&&!result.is_class1&&seedActive()) {
        PoseSeed seed;seed.R=state_.R_prev;seed.t=state_.t_prev;
        if(!motionGatePass(final_pose,seed,nullptr,false,"RobustStereo",true)) {
            final_pose=PoseEstimate{};result.failure_reason="motion_gate_rejected";
        }
    }
    const bool ok=accepted(result,final_pose);
    result.success=false;result.gpnp_success=false;
    if(!cfg.disparity_only)result.stereo_success=final_pose.success&&final_pose.source==PoseSource::StereoGpnp;
    result.disparity_only=cfg.disparity_only;result.stereo_method="LK";
    result.is_first_frame=!state_.has_cache;result.left_color=lc;result.right_color=rc;
    result.extract_ms=extraction_ms+flow_ms;result.stereo_match_ms=flow_ms;result.pnp_ms=pnp_ms;
    result.flow_pyramid_builds=builds;result.flow_coarse_attempts=coarse;result.flow_full_fallback=full_retry;result.flow_roi=flow_roi;
    timing["left_extract"]=extraction_ms;timing["flow"]=flow_ms;timing["stereo_pose"]=pnp_ms;result.timing=timing;
    if(ok)result.failure_reason.clear();
    step(ok?"FRAME:OK":"FRAME:FAIL");result.route_detail=route;
    if(ok)updateStickinessFromWinner(result.strategy_name.find("TT")!=std::string::npos||result.strategy_name=="TinyTarget"?"TinyTarget":"BinaryCorner");
    fillTemporalMeta(result);
    return result;
}
} // namespace gpnp
