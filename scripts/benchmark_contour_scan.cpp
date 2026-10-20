// Compare frozen sources and current sources with identical inputs/configuration.
#include "feature/BinaryCornerExtractor.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

static uint64_t imageHash(const cv::Mat& image) {
    uint64_t hash=1469598103934665603ULL;
    for(int y=0;y<image.rows;++y) for(size_t x=0;x<image.cols*image.elemSize();++x)
        hash=(hash^image.ptr<unsigned char>(y)[x])*1099511628211ULL;
    return hash;
}
int main(int argc,char** argv) {
    if(argc!=5) return 2;
    cv::setNumThreads(1);gpnp::g_verbose_console=false;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);
    if(!fs.isOpened()) return 3;
    auto b=fs["strategies"]["binary_corner"],t=fs["strategies"]["tiny_target"];
    gpnp::BinaryCornerExtractor::Config bc;
    bc.corners=(int)b["corners"];bc.kernel_size=(int)b["kernel_size"];
    bc.scale=(double)b["corner_scale"];
    bc.target_size={(int)b["target_width"],(int)b["target_height"]};
    bc.pixel_to_meter_scale_class0=(double)b["pixel_to_meter_scale_class0"];
    bc.pixel_to_meter_scale_class1=(double)b["pixel_to_meter_scale_class1"];
    bc.mono_identity=gpnp::readMonoCornerIdentityConfig(b["mono_identity"]);
    gpnp::BinaryCornerExtractor binary(bc,(std::string)b["template_dir"]);
    binary.setMonoContext(true);
    gpnp::TinyTargetExtractor::Config tt;
    tt.target_size={(int)t["target_width"],(int)t["target_height"]};
    tt.scale_factor=(int)t["scale_factor"];
    tt.square_size_m_class0=(double)t["square_size_m_class0"];
    tt.square_size_m_class1=(double)t["square_size_m_class1"];
    gpnp::TinyTargetExtractor tiny(tt,(std::string)t["template_dir"]);
    std::ifstream jobs(argv[2]);std::ofstream out(argv[3]);
    const int repeats=std::stoi(argv[4]);if(!jobs||!out||repeats<1)return 4;
    out<<"job,cls,area,extractor,capture,repeat,success,n,angle,overlap,points,ids,status,hypotheses,detected,rejected,debug_hash,extract_ms,refine_ms\n"<<std::setprecision(17);
    int job,cls,unused,x,y,w,h;std::string path,previous;cv::Mat image;
    while(jobs>>job>>path>>cls>>unused>>x>>y>>w>>h) {
        if(path!=previous){image=cv::imread(path);previous=path;}
        if(image.empty())return 5;
        cv::Mat gray;cv::cvtColor(image(cv::Rect(x,y,w,h)),gray,cv::COLOR_BGR2GRAY);
        binary.setUseClass1(cls==1);tiny.setUseClass1(cls==1);
        for(int capture=0;capture<2;++capture)for(int kind=0;kind<2;++kind) {
            if(kind==1 && w*h>4096)continue; // TT's intended small-target range.
            binary.setDebugCapture(capture);tiny.setDebugCapture(capture);
            gpnp::FeatureExtractor& extractor=kind==0?static_cast<gpnp::FeatureExtractor&>(binary):static_cast<gpnp::FeatureExtractor&>(tiny);
            if(repeats>1)extractor.extractMono(gray,{});
            for(int repeat=0;repeat<repeats;++repeat) {
                const auto started=std::chrono::steady_clock::now();
                auto r=extractor.extractMono(gray,{});
                const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                int angle=kind==0?-1:tiny.lastMatchedAngle();
                const auto& identity=binary.lastMonoIdentity();
                uint64_t debug_hash=0;
                if(capture) {
                    if(kind==0 && !binary.lastContourBinary().empty()) {
                        cv::Mat rendered=cv::Mat::zeros(binary.lastContourBinary().size(),CV_8U);
                        if(!binary.lastContour().empty())cv::drawContours(rendered,std::vector<std::vector<cv::Point>>{binary.lastContour()},-1,255,1);
                        debug_hash=imageHash(rendered);
                    } else if(kind==1)debug_hash=imageHash(tiny.lastLeftDebug().super_binary);
                }
                out<<job<<','<<cls<<','<<w*h<<','<<(kind==0?"BC":"TT")<<','<<capture<<','<<repeat<<','<<r.success<<','<<r.good_matches.size()<<','<<angle<<','<<(kind==0?binary.lastMatchOverlap():tiny.lastMatchOverlap())<<',';
                for(auto p:r.pts_left_match)out<<p.x<<':'<<p.y<<';';out<<',';
                for(auto m:r.good_matches)out<<m.queryIdx<<':'<<m.trainIdx<<';';
                out<<','<<(kind==0?identity.status:"")<<','<<(kind==0?identity.hypotheses:0)<<','<<(kind==0?identity.detected_count:0)<<','<<r.bc_rejected_count<<','<<debug_hash<<','<<ms<<','<<(kind==0?identity.refinement_ms:0)<<'\n';
            }
        }
    }
    return jobs.eof()?0:6;
}
