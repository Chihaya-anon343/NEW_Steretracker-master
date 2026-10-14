#include "common/Config.hpp"
#include "common/LogConfig.hpp"
#include "detection/YoloRoiProvider.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "matching/StereoFlowMatcher.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>

using namespace gpnp;
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct ObservedPoint { int region,index;cv::Point2f left,right;bool ok; };
struct Sample { double yolo=0,features=0,total=0,match=0;int pairs=0,builds=0,retries=0;bool sufficient=false;std::vector<double> errors;std::vector<ObservedPoint> points; };
double mean(const std::vector<double>& v){return v.empty()?0:std::accumulate(v.begin(),v.end(),0.0)/v.size();}
double p95(std::vector<double> v){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[static_cast<size_t>(std::ceil(v.size()*.95))-1];}
int main(int argc,char** argv){try{
    if(argc<4)throw std::runtime_error("Usage: benchmark_dual_roi_frontend config.json iterations output [prepare]");
    const int iterations=std::stoi(argv[2]);if(iterations<10)throw std::runtime_error("At least 10 iterations required");
    const std::string output=argv[3];std::filesystem::create_directories(output);g_verbose_console=false;
    cv::FileStorage fs(argv[1],cv::FileStorage::READ);if(!fs.isOpened())throw std::runtime_error("Cannot open config");
    const std::string left_path=fs["input"]["left"],right_path=fs["input"]["right"];
    cv::Mat left=cv::imread(left_path),right;
    if(left.empty())throw std::runtime_error("Cannot load left image");
    const double truth_d=80,truth_dy=1.774;
    if(argc>4&&std::string(argv[4])=="prepare"){
        const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-truth_d,0,1,truth_dy);
        cv::warpAffine(left,right,transform,left.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
        std::filesystem::create_directories(std::filesystem::path(right_path).parent_path());cv::imwrite(right_path,right);
    }else right=cv::imread(right_path);
    if(right.empty())throw std::runtime_error("Cannot load right image");
    auto yn=fs["yolo"],sn=fs["strategies"],bn=sn["binary_corner"],cn=fs["camera"];
    auto yc=makeYoloConfig(static_cast<std::string>(yn["model_path"]),DeviceType::CPU,static_cast<double>(yn["conf_threshold"]));
    yc.target_class_id=static_cast<int>(yn["target_class_id"]);yc.roi_expand_ratio=static_cast<double>(yn["roi_expand_ratio"]);
    RoiGenerator::Config roi_cfg{yc.target_class_id,static_cast<float>(yc.roi_expand_ratio),static_cast<int>(yn["roi_min_size"]),static_cast<int>(sn["dual_trigger_area"])};
    YoloRoiProvider yolo;if(!yolo.initialize(yc,roi_cfg))throw std::runtime_error("YOLO failed to initialize");
    BinaryCornerExtractor::Config bc;bc.corners=static_cast<int>(bn["corners"]);bc.kernel_size=static_cast<int>(bn["kernel_size"]);
    bc.scale=static_cast<double>(bn["corner_scale"]);bc.target_size={static_cast<int>(bn["target_width"]),static_cast<int>(bn["target_height"])};
    bc.pixel_to_meter_scale_class0=static_cast<double>(bn["pixel_to_meter_scale_class0"]);
    bc.pixel_to_meter_scale_class1=static_cast<double>(bn["pixel_to_meter_scale_class1"]);
    BinaryCornerExtractor extractor(bc,static_cast<std::string>(bn["template_dir"]));extractor.setDebugCapture(false);
    Eigen::Matrix3d k;k<<static_cast<double>(cn["fx"]),0,static_cast<double>(cn["cx"]),0,static_cast<double>(cn["fy"]),static_cast<double>(cn["cy"]),0,0,1;
    const auto camera=makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),{static_cast<double>(cn["baseline_mm"]),0,0});
    auto flow_cfg=readStereoFlowConfig(fs["stereo_frontend"]);
    const int pad=static_cast<int>(sn["dual_roi"]["secondary_expand_pixels"]);
    auto rect=[&](RoiRect r,int p){return cv::Rect(r.x-p,r.y-p,r.width+2*p,r.height+2*p)&cv::Rect(0,0,left.cols,left.rows);};
    std::ofstream points_csv(output+"/points.csv");
    points_csv<<"iteration,path,region,index,left_x,left_y,right_x,right_y,disparity_px,error_px,group_ok\n";
    auto run=[&](bool flow,int iteration){
        Sample sample;const auto start=Clock::now();RoiGroup l,r;
        if(flow)l=yolo.detectLeft(left).group;else std::tie(l,r)=yolo.detect(left,right);
        sample.yolo=milliseconds(start);
        if(!l.valid()||!l.is_dual||(!flow&&(!r.valid()||!r.is_dual)))throw std::runtime_error("YOLO must detect class0 and class1 in both baseline images");
        const auto feature_start=Clock::now();cv::Mat gl,gr;cv::cvtColor(left,gl,cv::COLOR_BGR2GRAY);cv::cvtColor(right,gr,cv::COLOR_BGR2GRAY);
        const cv::Rect lroi[2]={rect(l.primary,0),rect(l.secondary,pad)};
        std::unique_ptr<StereoFlowFrame> context;
        if(flow)context=std::make_unique<StereoFlowFrame>(camera,flow_cfg,gl,gr,lroi[0]|lroi[1]);
        for(int region=0;region<2;++region){
            std::vector<cv::Point2f> pl,pr;std::vector<int> ids;bool ok=false;
            if(flow){
                auto extracted=extractor.extractMono(gl(lroi[region]),cv::Mat());
                for(auto& p:extracted.pts_left_match)p+=cv::Point2f(lroi[region].x,lroi[region].y);
                auto matches=context->match(extracted.pts_left_match,extracted.pts_left_match);
                sample.match+=matches.total_ms;sample.builds+=matches.pyramid_builds;sample.retries+=matches.full_frame_fallback;
                pl=std::move(matches.left_points);pr=std::move(matches.right_points);ids=std::move(matches.left_indices);ok=matches.success;
            }else{
                const auto rr=rect(region?r.secondary:r.primary,region?pad:0);
                auto extracted=extractor.extract(gl(lroi[region]),gr(rr),cv::Mat(),cv::Mat());
                pl=extracted.pts_left_good;pr=extracted.pts_right_good;
                for(auto& p:pl)p+=cv::Point2f(lroi[region].x,lroi[region].y);
                for(auto& p:pr)p+=cv::Point2f(rr.x,rr.y);
                for(int i=0;i<static_cast<int>(pl.size());++i)ids.push_back(i);
                ok=pl.size()==pr.size()&&pl.size()>=static_cast<size_t>(flow_cfg.min_matches);
            }
            sample.sufficient|=ok;sample.pairs+=static_cast<int>(std::min(pl.size(),pr.size()));
            for(size_t i=0;i<pl.size()&&i<pr.size();++i){
                const double d=pl[i].x-pr[i].x,error=std::abs(d-truth_d);sample.errors.push_back(error);
                sample.points.push_back({region,ids[i],pl[i],pr[i],ok});
            }
        }
        sample.features=milliseconds(feature_start);sample.total=milliseconds(start);return sample;
    };
    for(int i=0;i<10;++i){run(i%2,-1);run(!(i%2),-1);}
    const auto before=yolo.inferenceCounts();std::vector<Sample> old_samples,new_samples;
    std::ofstream csv(output+"/samples.csv");csv<<"iteration,path,yolo_ms,features_ms,total_ms,match_ms,pairs,sufficient,pyramid_builds,full_retries\n";
    for(int i=0;i<iterations;++i){for(int j=0;j<2;++j){const bool flow=(i+j)%2;auto s=run(flow,i);(flow?new_samples:old_samples).push_back(s);
        csv<<i<<','<<(flow?"flow":"dual_bc")<<','<<s.yolo<<','<<s.features<<','<<s.total<<','<<s.match<<','<<s.pairs<<','<<s.sufficient<<','<<s.builds<<','<<s.retries<<'\n';}
        if((i+1)%10==0)std::cout<<"Measured "<<i+1<<'/'<<iterations<<'\n'<<std::flush;
    }
    const auto after=yolo.inferenceCounts();
    // Write observations after all measured work, outside the timed frontend.
    for(auto entry:std::vector<std::pair<std::string,const std::vector<Sample>*>>{{"dual_bc",&old_samples},{"flow",&new_samples}})
        for(size_t iteration=0;iteration<entry.second->size();++iteration)
            for(const auto& p:(*entry.second)[iteration].points) {
                const double d=p.left.x-p.right.x;
                points_csv<<iteration<<','<<entry.first<<','<<p.region<<','<<p.index<<','<<p.left.x<<','<<p.left.y<<','
                    <<p.right.x<<','<<p.right.y<<','<<d<<','<<std::abs(d-truth_d)<<','<<p.ok<<'\n';
            }
    std::ofstream json(output+"/summary.json");json<<"{\"opencv_threads\":"<<cv::getNumThreads()<<",\"iterations\":"<<iterations
        <<",\"yolo_left_calls\":"<<after.left-before.left<<",\"yolo_right_calls\":"<<after.right-before.right;
    for(auto entry:std::vector<std::pair<std::string,const std::vector<Sample>*>>{{"dual_bc",&old_samples},{"flow",&new_samples}}){
        std::vector<double> yolo_values,feature_values,total_values,errors;int sufficient=0,pairs=0,builds=0,retries=0;
        for(const auto& s:*entry.second){yolo_values.push_back(s.yolo);feature_values.push_back(s.features);total_values.push_back(s.total);
            errors.insert(errors.end(),s.errors.begin(),s.errors.end());sufficient+=s.sufficient;pairs+=s.pairs;builds+=s.builds;retries+=s.retries;}
        json<<",\""<<entry.first<<"\":{\"yolo_mean_ms\":"<<mean(yolo_values)<<",\"features_mean_ms\":"<<mean(feature_values)
            <<",\"total_mean_ms\":"<<mean(total_values)<<",\"total_p95_ms\":"<<p95(total_values)<<",\"sufficient_runs\":"<<sufficient
            <<",\"pairs\":"<<pairs<<",\"pyramid_builds\":"<<builds<<",\"full_retries\":"<<retries
            <<",\"disparity_mae_px\":"<<mean(errors)<<",\"disparity_max_error_px\":"<<(errors.empty()?0:*std::max_element(errors.begin(),errors.end()))<<'}';
    }
    json<<"}\n";std::cout<<"Summary: "<<output<<"/summary.json\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
