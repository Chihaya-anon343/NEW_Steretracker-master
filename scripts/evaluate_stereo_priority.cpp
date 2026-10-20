// Only images and detector ROIs enter this executable; ground truth is evaluated offline.
#include "tracker/StereoTracker.hpp"
#include "common/Config.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/TinyTargetPolicy.hpp"
#ifdef STEREO_ROBUST_EVAL
#include "common/StereoRobustness.hpp"
#endif
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
std::string quoted(const std::string& s){std::string t="\"";for(auto c:s){t+=c;if(c=='\"')t+=c;}return t+'\"';}
int main(int argc,char** argv) {
    if(argc<5||argc>6)return 2;
    using namespace gpnp;cv::setNumThreads(1);g_verbose_console=false;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened())return 3;
    auto s=fs["strategies"],b=s["binary_corner"],t=s["tiny_target"],d=s["dual_roi"],cam=fs["camera"];
    Eigen::Matrix3d K;K<<double(cam["fx"]),0,double(cam["cx"]),0,double(cam["fy"]),double(cam["cy"]),0,0,1;
    TrackerConfig c;c.tiny_max_area=int(s["tiny_max_area"]);c.tiny_max_area_class1=int(s["close_range"]["tiny_max_area"]);
    c.template_real_width_mm=double(s["board_template"]["template_real_width_mm"]);
    c.template_real_height_mm=double(s["board_template"]["template_real_height_mm"]);
    c.dual_roi_class1_fallback=int(d["class1_fallback_enabled"])!=0;c.dual_roi_primary_span_ratio=double(d["primary_span_ratio"]);
    c.dual_roi_secondary_expand=int(d["secondary_expand_pixels"]);c.dual_roi_tier1_only=int(d["tier1_only"])!=0;
    c.dual_roi_class0_points_path=std::string(d["class0_points_path"]);c.dual_roi_class1_points_path=std::string(d["class1_points_path"]);
    c.dual_roi_joint_match_gate_px=double(d["joint_match_gate_px"]);c.template_geometry=readTemplateGeometryConfig(fs["template_geometry"]);
    c.stereo_flow=readStereoFlowConfig(fs["stereo_frontend"]);c.temporal.enabled=false;
    c.roi_selection.tt_verification=readTinyTargetVerificationConfig(fs["roi_selection"]["tt_verification"]);
#ifdef STEREO_ROBUST_EVAL
    c.stereo_robustness=readStereoRobustnessConfig(fs["stereo_frontend"]["robustness"]);
#endif
    BinaryCornerExtractor::Config bc;bc.corners=int(b["corners"]);bc.kernel_size=int(b["kernel_size"]);bc.scale=double(b["corner_scale"]);
    bc.target_size={int(b["target_width"]),int(b["target_height"])};bc.roi_pad_pixels=int(b["roi_pad_pixels"]);
    bc.pixel_to_meter_scale_class0=double(b["pixel_to_meter_scale_class0"]);bc.pixel_to_meter_scale_class1=double(b["pixel_to_meter_scale_class1"]);
    bc.mono_identity=readMonoCornerIdentityConfig(b["mono_identity"]);
    TinyTargetExtractor::Config tt;tt.target_size={int(t["target_width"]),int(t["target_height"])};
    tt.scale_factor=int(t["scale_factor"]);tt.roi_pad_pixels=int(t["roi_pad_pixels"]);
    tt.square_size_m_class0=double(t["square_size_m_class0"]);tt.square_size_m_class1=double(t["square_size_m_class1"]);
    tt.robustness=readTinyTargetRobustnessConfig(t["robustness"]);
    StereoTracker tracker(K,Eigen::Matrix3d::Identity(),{double(cam["baseline_mm"]),0,0},
        std::string(s["board_template"]["template_path"]),c,bc,std::string(b["template_dir"]),tt,std::string(t["template_dir"]));
    tracker.setVerboseConsole(false);
    std::ifstream jobs(argv[2]);std::ofstream out(argv[3]);if(!jobs||!out)return 4;
    const int repeats=std::stoi(argv[4]);const bool visual=argc==6;
    out<<"job,repeat,success,stereo,disparity_ok,source,strategy,class1,extract_ms,flow_ms,pnp_ms,total_ms,builds,tx,ty,tz";
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)out<<",R"<<i<<j;
    out<<",failure,route,raw,pairs\n"<<std::setprecision(17);
    int job,dual,class1;std::string lp,rp,last_l,last_r;cv::Mat left,right;RoiGroup g,h;
    while(jobs>>job>>lp>>rp>>dual>>class1>>g.primary.x>>g.primary.y>>g.primary.width>>g.primary.height
          >>g.secondary.x>>g.secondary.y>>g.secondary.width>>g.secondary.height
          >>h.primary.x>>h.primary.y>>h.primary.width>>h.primary.height
          >>h.secondary.x>>h.secondary.y>>h.secondary.width>>h.secondary.height) {
        g.is_dual=h.is_dual=dual!=0;g.is_class1=h.is_class1=class1!=0;
        if(lp!=last_l){left=cv::imread(lp);last_l=lp;}if(rp!=last_r){right=cv::imread(rp);last_r=rp;}
        if(left.empty()||right.empty())return 5;
        if(visual){auto dir=std::filesystem::path(argv[5])/std::to_string(job);std::filesystem::create_directories(dir);tracker.setOutputDir(dir.string());}
        for(int rep=repeats>1?-1:0;rep<repeats;++rep) {
            tracker.clearCache();tracker.setFrameNumber(job);cv::theRNG().state=20261020;
            const bool left_only=tracker.prefersLeftStereoFlow(g);
            auto start=std::chrono::steady_clock::now();auto r=tracker.process(left,right,visual,&g,left_only?nullptr:&h);
            double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();if(rep<0)continue;
            out<<job<<','<<rep<<','<<r.success<<','<<r.stereo_success<<','<<r.disparity_success<<','<<poseSourceName(r.pose_source)
                <<','<<r.strategy_name<<','<<r.is_class1<<','<<r.extract_ms<<','<<r.stereo_match_ms<<','<<r.pnp_ms<<','<<elapsed
                <<','<<r.flow_pyramid_builds<<','<<r.t.x()<<','<<r.t.y()<<','<<r.t.z();
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)out<<','<<r.R(i,j);
            std::vector<int> ids(r.pts_left_match.size(),-1);for(auto m:r.good_matches)if(m.queryIdx>=0&&m.queryIdx<int(ids.size()))ids[m.queryIdx]=m.trainIdx;
            std::ostringstream raw,pairs;raw<<std::setprecision(17);pairs<<std::setprecision(17);
            for(size_t i=0;i<r.pts_left_match.size();++i)raw<<ids[i]<<':'<<r.pts_left_match[i].x<<':'<<r.pts_left_match[i].y<<';';
            for(size_t k=0;k<r.pts_left_good.size()&&k<r.pts_right_good.size();++k) {
                int q=k<r.idx_from_filtered.size()?r.idx_from_filtered[k]:int(k);
                int id=q>=0&&q<int(ids.size())?ids[q]:-1;
                pairs<<id<<':'<<r.pts_left_good[k].x<<':'<<r.pts_left_good[k].y<<':'<<r.pts_right_good[k].x<<':'<<r.pts_right_good[k].y<<';';
            }
            out<<','<<quoted(r.failure_reason)<<','<<quoted(r.route_detail)<<','<<quoted(raw.str())<<','<<quoted(pairs.str())<<'\n';
        }
    }
    utils::AsyncImageSaver::flush();return jobs.eof()?0:6;
}
