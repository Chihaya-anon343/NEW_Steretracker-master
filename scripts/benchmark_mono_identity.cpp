// Paired timing: both extractors use the same decoded image/ROI, alternate
// execution order, warm once, and use one OpenCV thread. Run without other jobs.
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened())return 3;
    auto node=fs["strategies"]["binary_corner"];gpnp::BinaryCornerExtractor::Config c;
    c.corners=(int)node["corners"];c.kernel_size=(int)node["kernel_size"];c.scale=(double)node["corner_scale"];
    c.target_size={(int)node["target_width"],(int)node["target_height"]};
    c.pixel_to_meter_scale_class0=(double)node["pixel_to_meter_scale_class0"];
    c.pixel_to_meter_scale_class1=(double)node["pixel_to_meter_scale_class1"];
    c.mono_identity=gpnp::readMonoCornerIdentityConfig(node["mono_identity"]);
    c.mono_identity.enabled=true;c.mono_identity.method="hybrid";
    auto legacy=c;legacy.mono_identity.enabled=false;
    gpnp::g_verbose_console=false;cv::setNumThreads(1);
    gpnp::BinaryCornerExtractor old(legacy,(std::string)node["template_dir"]),updated(c,(std::string)node["template_dir"]);
    old.setMonoContext(true);updated.setMonoContext(true);old.setDebugCapture(false);updated.setDebugCapture(false);
    std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)return 4;
    out<<"job,repeat,mode,success,extract_ms,identity_ms\n"<<std::setprecision(12);
    int job,cls,method,x,y,w,h;std::string path;
    while(in>>job>>path>>cls>>method>>x>>y>>w>>h) {
        cv::Mat color=cv::imread(path),gray;if(color.empty())return 5;
        cv::cvtColor(color(cv::Rect(x,y,w,h)),gray,cv::COLOR_BGR2GRAY);
        old.setUseClass1(cls==1);updated.setUseClass1(cls==1);
        old.extractMono(gray,{});updated.extractMono(gray,{});
        for(int repeat=0;repeat<8;++repeat)for(int leg=0;leg<2;++leg) {
            const bool new_mode=(leg+repeat)%2;auto& extractor=new_mode?updated:old;
            auto begin=std::chrono::steady_clock::now();auto result=extractor.extractMono(gray,{});
            double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            out<<job<<','<<repeat<<','<<(new_mode?"hybrid":"legacy")<<','<<result.success<<','<<elapsed<<','<<extractor.lastMonoIdentity().elapsed_ms<<'\n';
        }
    }
    return in.eof()?0:6;
}
