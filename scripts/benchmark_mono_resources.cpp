// Linux-only process resource probe. Same decoded, common-success ROIs for each
// mode. CPU/wall interval excludes loading and warmup; ru_maxrss includes both.
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/resource.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
double cpu(const rusage& r) {return 1000.*(r.ru_utime.tv_sec+r.ru_stime.tv_sec)+.001*(r.ru_utime.tv_usec+r.ru_stime.tv_usec);}
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    std::string mode=argv[3];if(mode!="current"&&mode!="normalized"&&mode!="refined")return 2;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened())return 3;
    auto b=fs["strategies"]["binary_corner"];gpnp::BinaryCornerExtractor::Config c;
    c.corners=(int)b["corners"];c.kernel_size=(int)b["kernel_size"];c.scale=(double)b["corner_scale"];
    c.target_size={(int)b["target_width"],(int)b["target_height"]};
    c.pixel_to_meter_scale_class0=(double)b["pixel_to_meter_scale_class0"];
    c.pixel_to_meter_scale_class1=(double)b["pixel_to_meter_scale_class1"];
    c.mono_identity=gpnp::readMonoCornerIdentityConfig(b["mono_identity"]);
    c.mono_identity.enabled=true;c.mono_identity.class1_enabled=false;
    c.mono_identity.frontend=mode=="current"?"legacy":"normalized";
    c.mono_identity.refine_original=mode=="refined";
    gpnp::g_verbose_console=false;cv::setNumThreads(1);
    gpnp::BinaryCornerExtractor extractor(c,(std::string)b["template_dir"]);
    extractor.setMonoContext(true);extractor.setDebugCapture(false);
    std::ifstream in(argv[2]);int job,cls,method,x,y,w,h;std::string path;
    std::vector<cv::Mat> images;
    while(in>>job>>path>>cls>>method>>x>>y>>w>>h) {
        if(cls!=0)return 4;cv::Mat color=cv::imread(path),gray;if(color.empty())return 5;
        cv::cvtColor(color(cv::Rect(x,y,w,h)),gray,cv::COLOR_BGR2GRAY);images.push_back(gray);
    }
    if(!in.eof()||images.empty())return 6;
    for(const auto& image:images)if(!extractor.extractMono(image,{}).success)return 7;
    rusage before{},after{};getrusage(RUSAGE_SELF,&before);
    const auto start=std::chrono::steady_clock::now();int success=0;
    for(int repeat=0;repeat<50;++repeat)for(const auto& image:images)success+=extractor.extractMono(image,{}).success;
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    getrusage(RUSAGE_SELF,&after);
    std::cout<<std::setprecision(12)<<"RESOURCE {\"mode\":\""<<mode<<"\",\"roi_count\":"<<images.size()
        <<",\"calls\":"<<50*images.size()<<",\"success\":"<<success<<",\"cpu_ms\":"<<cpu(after)-cpu(before)
        <<",\"wall_ms\":"<<wall<<",\"single_core_utilization_pct\":"<<100*(cpu(after)-cpu(before))/wall
        <<",\"peak_rss_mib_including_load_warmup\":"<<after.ru_maxrss/1024.
        <<",\"peak_rss_mib_before_interval\":"<<before.ru_maxrss/1024.<<"}\n";
}
