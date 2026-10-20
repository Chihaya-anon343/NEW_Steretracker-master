// Fixed input ROI replay: the input file contains only image paths and boxes.
// GT is consumed later by the Python analysis, never by this executable.
#include "tracker/MonoTracker.hpp"
#include "common/MonoDualRoiPolicy.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/TinyTargetPolicy.hpp"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <sstream>

std::string csv(const std::string& s){std::string t="\"";for(char c:s){t+=c;if(c=='\"')t+=c;}return t+'\"';}
int main(int argc,char** argv) {
    if(argc<5||argc>6)return 2;
    cv::setNumThreads(1);gpnp::g_verbose_console=false;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened())return 3;
    using namespace gpnp;
    auto s=fs["strategies"],b=s["binary_corner"],t=s["tiny_target"],d=s["dual_roi"],cam=fs["camera"];
    Eigen::Matrix3d K;K<<double(cam["fx"]),0,double(cam["cx"]),0,double(cam["fy"]),double(cam["cy"]),0,0,1;
    TrackerConfig c;c.tiny_max_area=int(s["tiny_max_area"]);c.tiny_max_area_class1=int(s["close_range"]["tiny_max_area"]);
    c.template_real_width_mm=double(s["board_template"]["template_real_width_mm"]);
    c.template_real_height_mm=double(s["board_template"]["template_real_height_mm"]);
    c.dual_roi_class1_fallback=int(d["class1_fallback_enabled"])!=0;
    c.dual_roi_primary_span_ratio=double(d["primary_span_ratio"]);
    c.dual_roi_secondary_expand=int(d["secondary_expand_pixels"]);
    c.dual_roi_tier1_only=int(d["tier1_only"])!=0;
    c.dual_roi_class0_points_path=std::string(d["class0_points_path"]);
    c.dual_roi_class1_points_path=std::string(d["class1_points_path"]);
    c.dual_roi_joint_match_gate_px=double(d["joint_match_gate_px"]);
    c.mono_dual_roi=readMonoDualRoiConfig(d["mono_robustness"]);
    c.template_geometry=readTemplateGeometryConfig(fs["template_geometry"]);
    c.mono_pose_prior.enabled=int(fs["mono_pose_prior"]["enabled"])!=0;
    c.mono_pose_prior.max_tilt_deg=double(fs["mono_pose_prior"]["max_tilt_deg"]);
    c.temporal.enabled=false;
    c.roi_selection.tt_verification=readTinyTargetVerificationConfig(fs["roi_selection"]["tt_verification"]);
    BinaryCornerExtractor::Config bc;
    bc.corners=int(b["corners"]);bc.kernel_size=int(b["kernel_size"]);bc.scale=double(b["corner_scale"]);
    bc.target_size={int(b["target_width"]),int(b["target_height"])};
    bc.roi_pad_pixels=int(b["roi_pad_pixels"]);
    bc.pixel_to_meter_scale_class0=double(b["pixel_to_meter_scale_class0"]);
    bc.pixel_to_meter_scale_class1=double(b["pixel_to_meter_scale_class1"]);
    bc.mono_identity=readMonoCornerIdentityConfig(b["mono_identity"]);bc.template_geometry=c.template_geometry;
    TinyTargetExtractor::Config tt;tt.target_size={int(t["target_width"]),int(t["target_height"])};
    tt.scale_factor=int(t["scale_factor"]);tt.roi_pad_pixels=int(t["roi_pad_pixels"]);
    tt.square_size_m_class0=double(t["square_size_m_class0"]);tt.square_size_m_class1=double(t["square_size_m_class1"]);
    tt.template_geometry=c.template_geometry;tt.robustness=readTinyTargetRobustnessConfig(t["robustness"]);
    MonoTracker tracker(K,std::string(s["board_template"]["template_path"]),c,bc,std::string(b["template_dir"]),tt,std::string(t["template_dir"]));
    tracker.setVerboseConsole(false);
    const int repeats=std::stoi(argv[4]);if(repeats<1)return 4;
    std::ifstream input(argv[2]);std::ofstream out(argv[3]);if(!input||!out)return 5;
    out<<"job,repeat,success,strategy,is_class1,n,extract_ms,pnp_ms,total_ms,tx,ty,tz";
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)out<<",R"<<i<<j;
    out<<",failure,route,observations\n"<<std::setprecision(17);
    int job;std::string path;RoiGroup g;std::string previous;cv::Mat image;
    while(input>>job>>path>>g.primary.x>>g.primary.y>>g.primary.width>>g.primary.height
          >>g.secondary.x>>g.secondary.y>>g.secondary.width>>g.secondary.height) {
        g.is_dual=true;
        if(previous!=path){image=cv::imread(path);previous=path;}
        if(image.empty())return 6;
        const bool capture=argc==6;
        if(capture){auto dir=std::filesystem::path(argv[5])/std::to_string(job);std::filesystem::create_directories(dir);tracker.setOutputDir(dir.string());}
        for(int rep=repeats>1?-1:0;rep<repeats;++rep) {
            tracker.clearCache();tracker.setFrameNumber(job);cv::theRNG().state=20261020;
            const auto begin=std::chrono::steady_clock::now();
            const auto r=tracker.process(image,capture,&g);
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            if(rep<0)continue;
            out<<job<<','<<rep<<','<<r.success<<','<<r.strategy_name<<','<<r.is_class1<<','<<r.pts_left_match.size()
               <<','<<r.extract_ms<<','<<r.pnp_ms<<','<<elapsed<<','<<r.t.x()<<','<<r.t.y()<<','<<r.t.z();
            for(int i=0;i<3;++i)for(int j=0;j<3;++j)out<<','<<r.R(i,j);
            std::string points;
            for(size_t i=0;i<r.good_matches.size()&&i<r.pts_left_match.size();++i) {
                const auto id=r.good_matches[i].trainIdx;const auto p=r.pts_left_match[i];
                if(id<0||id>=int(r.pose_model_points.size()))return 7;
                auto m=r.pose_model_points[id];
                std::ostringstream a;a<<std::setprecision(17)<<id<<':'<<p.x<<':'<<p.y<<':'<<m.x()<<':'<<m.y()<<':'<<m.z()<<';';points+=a.str();
            }
            out<<','<<csv(r.failure_reason)<<','<<csv(r.route_detail)<<','<<csv(points)<<'\n';
        }
    }
    return input.eof()?0:8;
}
