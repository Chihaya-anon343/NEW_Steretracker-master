// Isolated real C++ BC/TT extraction on annotated ROIs; no PnP or YOLO.
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
int main(int argc, char** argv) {
    if(argc!=4) return 2;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);
    if(!fs.isOpened()) return 3;
    auto b=fs["strategies"]["binary_corner"], t=fs["strategies"]["tiny_target"];
    gpnp::BinaryCornerExtractor::Config bc;
    bc.corners=(int)b["corners"]; bc.kernel_size=(int)b["kernel_size"];
    bc.scale=(double)b["corner_scale"];
    bc.target_size={(int)b["target_width"],(int)b["target_height"]};
    bc.pixel_to_meter_scale_class0=(double)b["pixel_to_meter_scale_class0"];
    bc.pixel_to_meter_scale_class1=(double)b["pixel_to_meter_scale_class1"];
    bc.mono_identity=gpnp::readMonoCornerIdentityConfig(b["mono_identity"]);
    gpnp::TinyTargetExtractor::Config tt;
    tt.scale_factor=(int)t["scale_factor"];
    tt.target_size={(int)t["target_width"],(int)t["target_height"]};
    tt.square_size_m_class0=(double)t["square_size_m_class0"];
    tt.square_size_m_class1=(double)t["square_size_m_class1"];
    gpnp::g_verbose_console=false; cv::setNumThreads(1);
    gpnp::BinaryCornerExtractor binary(bc,(std::string)b["template_dir"]);
    gpnp::TinyTargetExtractor tiny(tt,(std::string)t["template_dir"]);
    binary.setMonoContext(fs["benchmark_stereo_context"].empty() || (int)fs["benchmark_stereo_context"]==0);
    binary.setDebugCapture(false); tiny.setDebugCapture(false);
    std::ifstream input(argv[2]); std::ofstream output(argv[3]);
    if(!input || !output) return 4;
    output << "job,success,n,x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,identity_status,identity_path,identity_ms,coarse_iou,identity_iou,fit_error_px,score_gap,extract_ms,failure_reason,refined_points,normalize_ms,corners_ms,refine_ms,validate_ms,hypotheses\n" << std::setprecision(12);
    std::string path, previous; cv::Mat image;
    int job,cls,method,x,y,w,h;
    while(input>>job>>path>>cls>>method>>x>>y>>w>>h) {
        if(path!=previous) { image=cv::imread(path); previous=path; }
        if(image.empty()) return 5;
        cv::Mat color=image(cv::Rect(x,y,w,h)),gray;
        cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);
        gpnp::FeatureExtractor* extractor=method==0 ? static_cast<gpnp::FeatureExtractor*>(&binary) : static_cast<gpnp::FeatureExtractor*>(&tiny);
        extractor->setUseClass1(cls==1);
        const auto begin=std::chrono::steady_clock::now();
        auto r=extractor->extractMono(gray,color);
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        output<<job<<','<<r.success<<','<<r.pts_left_match.size();
        for(int i=0;i<10;++i) {
            if(i<(int)r.pts_left_match.size()) output<<','<<r.pts_left_match[i].x+x<<','<<r.pts_left_match[i].y+y;
            else output<<",,";
        }
        auto d=method==0 ? binary.lastMonoIdentity() : gpnp::MonoCornerIdentityResult{};
        output<<','<<d.status<<','<<d.path<<','<<d.elapsed_ms<<','<<d.coarse_iou<<','<<d.iou
              <<','<<d.fit_error_px<<','<<d.score_gap<<','<<elapsed<<','<<r.failure_reason<<','<<d.refined_points
              <<','<<d.normalization_ms<<','<<d.corners_ms<<','<<d.refinement_ms<<','<<d.validation_ms<<','<<d.hypotheses<<'\n';
        if(job%40==0) output.flush();
    }
    return input.eof()?0:6;
}
