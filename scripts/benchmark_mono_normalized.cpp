// Paired A/B/C, alternating order, decoded inputs excluded, one OpenCV thread.
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened()) return 3;
    auto node=fs["strategies"]["binary_corner"];gpnp::BinaryCornerExtractor::Config c;
    c.corners=(int)node["corners"];c.kernel_size=(int)node["kernel_size"];c.scale=(double)node["corner_scale"];
    c.target_size={(int)node["target_width"],(int)node["target_height"]};
    c.pixel_to_meter_scale_class0=(double)node["pixel_to_meter_scale_class0"];
    c.pixel_to_meter_scale_class1=(double)node["pixel_to_meter_scale_class1"];
    c.mono_identity=gpnp::readMonoCornerIdentityConfig(node["mono_identity"]);
    c.mono_identity.enabled=true;c.mono_identity.class1_enabled=false;c.mono_identity.method="hybrid";
    c.mono_identity.frontend="legacy";auto b=c;b.mono_identity.frontend="normalized";b.mono_identity.refine_original=false;
    auto r=b;r.mono_identity.refine_original=true;
    gpnp::g_verbose_console=false;cv::setNumThreads(1);
    gpnp::BinaryCornerExtractor a(c,(std::string)node["template_dir"]),normalized(b,(std::string)node["template_dir"]),refined(r,(std::string)node["template_dir"]);
    gpnp::BinaryCornerExtractor* extractors[]={&a,&normalized,&refined};
    const char* modes[]={"current","normalized","refined"};
    for(auto e:extractors) {e->setMonoContext(true);e->setDebugCapture(false);}
    std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)return 4;
    out<<"job,cls,repeat,mode,success,extract_ms,normalize_ms,corners_ms,refine_ms,validate_ms,refined_points,hypotheses\n"<<std::setprecision(12);
    int job,cls,method,x,y,w,h;std::string path;
    while(in>>job>>path>>cls>>method>>x>>y>>w>>h) {
        cv::Mat color=cv::imread(path),gray;if(color.empty())return 5;
        cv::cvtColor(color(cv::Rect(x,y,w,h)),gray,cv::COLOR_BGR2GRAY);
        for(auto e:extractors) {e->setUseClass1(cls==1);e->extractMono(gray,{});}
        for(int repeat=0;repeat<12;++repeat)for(int leg=0;leg<3;++leg) {
            int mode=(repeat+((repeat/3)%2 ? 2-leg:leg))%3;auto& e=*extractors[mode];
            auto begin=std::chrono::steady_clock::now();auto result=e.extractMono(gray,{});
            double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            const auto& d=e.lastMonoIdentity();
            out<<job<<','<<cls<<','<<repeat<<','<<modes[mode]<<','<<result.success<<','<<elapsed<<','
               <<d.normalization_ms<<','<<d.corners_ms<<','<<d.refinement_ms<<','<<d.validation_ms<<','<<d.refined_points<<','<<d.hypotheses<<'\n';
        }
    }
    return in.eof()?0:6;
}
