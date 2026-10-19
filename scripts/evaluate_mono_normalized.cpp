// A/B/C quality evaluation shares decoded image and ROI, never reads GT.
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <filesystem>
int main(int argc,char** argv) {
    if(argc!=4 && argc!=5) return 2;
    const bool capture=argc==5 && std::string(argv[4])=="--capture";
    const std::string root=argv[1];
    const char* modes[]={"current","normalized","refined"};
    std::vector<std::unique_ptr<gpnp::BinaryCornerExtractor>> extractors;
    std::vector<std::ofstream> outputs;
    gpnp::g_verbose_console=false;cv::setNumThreads(1);
    for(auto mode:modes) {
        const std::string dir=root+"/corners/"+mode;
        cv::FileStorage fs(dir+"/config_snapshot.json",cv::FileStorage::READ);if(!fs.isOpened())return 3;
        auto b=fs["strategies"]["binary_corner"];gpnp::BinaryCornerExtractor::Config c;
        c.corners=(int)b["corners"];c.kernel_size=(int)b["kernel_size"];c.scale=(double)b["corner_scale"];
        c.target_size={(int)b["target_width"],(int)b["target_height"]};
        c.pixel_to_meter_scale_class0=(double)b["pixel_to_meter_scale_class0"];
        c.pixel_to_meter_scale_class1=(double)b["pixel_to_meter_scale_class1"];
        c.mono_identity=gpnp::readMonoCornerIdentityConfig(b["mono_identity"]);
        auto e=std::make_unique<gpnp::BinaryCornerExtractor>(c,(std::string)b["template_dir"]);
        e->setMonoContext(true);e->setDebugCapture(capture);extractors.push_back(std::move(e));
        outputs.emplace_back(dir+"/detections.csv");if(!outputs.back())return 4;
        outputs.back()<<"job,success,n,x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,identity_status,identity_path,identity_ms,coarse_iou,identity_iou,fit_error_px,score_gap,extract_ms,failure_reason,refined_points,normalize_ms,corners_ms,refine_ms,validate_ms,hypotheses\n"<<std::setprecision(12);
    }
    std::ifstream in(argv[2]);if(!in)return 4;
    int job,cls,method,x,y,w,h,count=0;std::string path,previous;cv::Mat image;
    const int limit=std::stoi(argv[3]); // 0 = complete suite; positive = diagnostic pilot
    while(in>>job>>path>>cls>>method>>x>>y>>w>>h) {
        if(path!=previous) {image=cv::imread(path);previous=path;}
        if(image.empty())return 5;
        cv::Mat color=image(cv::Rect(x,y,w,h)),gray;cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);
        for(int mode=0;mode<3;++mode) {
            auto& e=*extractors[mode];e.setUseClass1(cls==1);
            auto start=std::chrono::steady_clock::now();auto r=e.extractMono(gray,color);
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            auto& out=outputs[mode];out<<job<<','<<r.success<<','<<r.pts_left_match.size();
            for(int i=0;i<10;++i) {
                if(i<(int)r.pts_left_match.size())out<<','<<r.pts_left_match[i].x+x<<','<<r.pts_left_match[i].y+y;
                else out<<",,";
            }
            const auto& d=e.lastMonoIdentity();out<<','<<d.status<<','<<d.path<<','<<d.elapsed_ms<<','<<d.coarse_iou<<','<<d.iou
                <<','<<d.fit_error_px<<','<<d.score_gap<<','<<elapsed<<','<<r.failure_reason<<','<<d.refined_points
                <<','<<d.normalization_ms<<','<<d.corners_ms<<','<<d.refinement_ms<<','<<d.validation_ms<<','<<d.hypotheses<<'\n';
            if(capture) {
                const std::string prefix=root+"/diagnostics/job"+std::to_string(job)+"_"+modes[mode];
                std::filesystem::create_directories(root+"/diagnostics");
                for(const auto& item:std::vector<std::pair<std::string,cv::Mat>>{
                    {"foreground",d.foreground},{"coarse",d.coarse_normalized},
                    {"normalized",d.normalized},{"reference",d.reference}})
                    if(!item.second.empty())cv::imwrite(prefix+"_"+item.first+".png",item.second);
                cv::FileStorage meta(prefix+".json",cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
                meta<<"status"<<d.status<<"path"<<d.path<<"iou"<<d.iou<<"fit_error_px"<<d.fit_error_px;
                if(!d.image_to_template.empty())meta<<"image_to_template"<<d.image_to_template;
                if(!d.before_refinement.empty())meta<<"before_refinement"<<cv::Mat(d.before_refinement);
                if(!d.points.empty())meta<<"points"<<cv::Mat(d.points);
            }
            if(count%20==0)out.flush();
        }
        if(limit>0 && ++count>=limit)break;
        if(limit==0)++count;
    }
    return in.eof() || (limit>0 && count==limit)?0:6;
}
