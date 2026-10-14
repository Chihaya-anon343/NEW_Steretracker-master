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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
using namespace gpnp;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
struct Sample {
    double yolo=0,features=0,total=0,disparity=0,match=0,pyramid=0,lk=0,crop=0;
    int pairs=0,roi_area=0; bool ok=false,full_retry=false;
};
double quantile(std::vector<double> v,double p) { std::sort(v.begin(),v.end()); return v[static_cast<size_t>((v.size()-1)*p)]; }
RoiRect padded(RoiRect r,int pad,cv::Size size) {
    const int x=std::max(0,r.x-pad),y=std::max(0,r.y-pad);
    return {x,y,std::min(size.width-x,r.width+2*pad),std::min(size.height-y,r.height+2*pad)};
}
int main(int argc,char** argv) {
  try {
    const std::string config=argc>1?argv[1]:"config/tracker_config.json";
    const int iterations=argc>2?std::stoi(argv[2]):50;
    const std::string output=argc>3?argv[3]:"output/bc_flow_benchmark";
    const bool roi_compare=argc>4 && std::string(argv[4])=="roi";
    const std::string old_name=roi_compare?"全图光流":"原路径",new_name=roi_compare?"ROI 光流":"光流路径";
    if(iterations<10) throw std::runtime_error("Use at least 10 measured iterations");
    g_verbose_console=false;
    cv::FileStorage fs(config,cv::FileStorage::READ);
    if(!fs.isOpened()) throw std::runtime_error("Cannot open config");
    const std::string lp=fs["input"]["left"],rp=fs["input"]["right"];
    const cv::Mat left=cv::imread(lp),right=cv::imread(rp);
    if(left.empty()||right.empty()) throw std::runtime_error("Cannot load images");
    auto yn=fs["yolo"],sn=fs["strategies"],bn=sn["binary_corner"],cn=fs["camera"];
    auto yc=makeYoloConfig(static_cast<std::string>(yn["model_path"]),DeviceType::CPU,static_cast<double>(yn["conf_threshold"]));
    yc.target_class_id=static_cast<int>(yn["target_class_id"]);yc.roi_expand_ratio=static_cast<double>(yn["roi_expand_ratio"]);
    RoiGenerator::Config rc{yc.target_class_id,static_cast<float>(yc.roi_expand_ratio),static_cast<int>(yn["roi_min_size"]),static_cast<int>(sn["dual_trigger_area"])};
    YoloRoiProvider yolo;if(!yolo.initialize(yc,rc)) throw std::runtime_error("YOLO unavailable");
    BinaryCornerExtractor::Config bc;
    bc.corners=static_cast<int>(bn["corners"]);bc.kernel_size=static_cast<int>(bn["kernel_size"]);bc.scale=static_cast<double>(bn["corner_scale"]);
    bc.target_size={static_cast<int>(bn["target_width"]),static_cast<int>(bn["target_height"])};
    bc.pixel_to_meter_scale_class0=static_cast<double>(bn["pixel_to_meter_scale_class0"]);
    bc.pixel_to_meter_scale_class1=static_cast<double>(bn["pixel_to_meter_scale_class1"]);bc.roi_pad_pixels=static_cast<int>(bn["roi_pad_pixels"]);
    BinaryCornerExtractor extractor(bc,static_cast<std::string>(bn["template_dir"]));extractor.setDebugCapture(false);
    Eigen::Matrix3d k;k << static_cast<double>(cn["fx"]),0,static_cast<double>(cn["cx"]),0,static_cast<double>(cn["fy"]),static_cast<double>(cn["cy"]),0,0,1;
    auto flow_config=readStereoFlowConfig(fs["stereo_frontend"]);flow_config.enabled=true;flow_config.disparity_only=true;
    StereoFlowMatcher matcher(makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),Eigen::Vector3d(static_cast<double>(cn["baseline_mm"]),0,0)),flow_config);
    auto full_cfg=flow_config;full_cfg.roi_pyramid=false;
    auto roi_cfg=flow_config;roi_cfg.roi_pyramid=true;
    const auto camera=makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),Eigen::Vector3d(static_cast<double>(cn["baseline_mm"]),0,0));
    StereoFlowMatcher full_matcher(camera,full_cfg),roi_matcher(camera,roi_cfg);
    auto run=[&](bool flow) {
      Sample s;const auto start=Clock::now();RoiGroup lg,rg;
      const bool use_flow=flow || roi_compare;
      if(use_flow) lg=yolo.detectLeft(left).group;else std::tie(lg,rg)=yolo.detect(left,right);
      s.yolo=ms(start);
      if(!lg.valid()||lg.is_dual||lg.is_class1||(!use_flow&&!rg.valid())||lg.primary.width*lg.primary.height<=static_cast<int>(sn["tiny_max_area"]))
        throw std::runtime_error("Requires ordinary class0 BC detections");
      const auto features_start=Clock::now();cv::Mat gl,gr;
      cv::cvtColor(left,gl,cv::COLOR_BGR2GRAY);cv::cvtColor(right,gr,cv::COLOR_BGR2GRAY);
      const auto lr=padded(lg.primary,bc.roi_pad_pixels,left.size());
      cv::Mat lc=gl(cv::Rect(lr.x,lr.y,lr.width,lr.height)).clone();std::vector<double> disparities;
      if(use_flow) {
        auto result=extractor.extractMono(lc,cv::Mat());
        for(auto& p:result.pts_left_match) p+=cv::Point2f(lr.x,lr.y);
        const auto& selected=roi_compare?(flow?roi_matcher:full_matcher):matcher;
        auto matched=selected.match(gl,gr,result.pts_left_match,result.pts_left_match,{lr.x,lr.y,lr.width,lr.height});
        s.match=matched.total_ms;s.pyramid=matched.pyramid_ms;s.lk=matched.lk_ms;s.crop=matched.crop_ms;
        s.roi_area=matched.roi_used?matched.search_roi.area():left.rows*left.cols;s.full_retry=matched.full_frame_fallback;
        s.ok=matched.success;s.pairs=static_cast<int>(matched.left_points.size());
        for(size_t i=0;i<matched.left_points.size();++i) disparities.push_back(matched.left_points[i].x-matched.right_points[i].x);
      } else {
        const auto rr=padded(rg.primary,bc.roi_pad_pixels,right.size());
        auto result=extractor.extract(lc,gr(cv::Rect(rr.x,rr.y,rr.width,rr.height)).clone(),cv::Mat(),cv::Mat());
        s.pairs=static_cast<int>(result.pts_left_good.size());
        // BC stereo extraction leaves the overall pose-success flag false.
        s.ok=s.pairs>=flow_config.min_matches && result.n_kp_left>=3 &&
             result.pts_right_good.size()==result.pts_left_good.size();
        for(size_t i=0;i<result.pts_left_good.size();++i) disparities.push_back(result.pts_left_good[i].x+lr.x-result.pts_right_good[i].x-rr.x);
      }
      if(!disparities.empty()) s.disparity=quantile(disparities,.5);
      s.features=ms(features_start);s.total=ms(start);return s;
    };
    for(int i=0;i<10;++i) { run(i%2);run(!(i%2)); }
    const auto before=yolo.inferenceCounts();std::vector<Sample> old_samples,new_samples;
    std::filesystem::create_directories(output);std::ofstream csv(output+"/samples.csv");
    csv<<"iteration,path,yolo_ms,features_ms,total_ms,pairs,matching_ok,median_disparity_px,match_ms,pyramid_ms,lk_ms,crop_ms,roi_area,full_retry\n";
    for(int i=0;i<iterations;++i) {
      for(int j=0;j<2;++j) {
        const bool flow=(i+j)%2;auto s=run(flow);(flow?new_samples:old_samples).push_back(s);
        const std::string path=roi_compare?(flow?"roi_lk":"full_lk"):(flow?"left_bc_lk":"dual_bc");
        csv<<i<<','<<path<<','<<s.yolo<<','<<s.features<<','<<s.total<<','<<s.pairs<<','<<s.ok<<','<<s.disparity
           <<','<<s.match<<','<<s.pyramid<<','<<s.lk<<','<<s.crop<<','<<s.roi_area<<','<<s.full_retry<<'\n';
      }
      if((i+1)%10==0) std::cout<<"Measured "<<i+1<<'/'<<iterations<<" pairs of runs\n"<<std::flush;
    }
    auto values=[](const std::vector<Sample>& samples,double Sample::*field) { std::vector<double> v;for(const auto& s:samples)v.push_back(s.*field);return v; };
    auto mean=[](const std::vector<double>& v) { return std::accumulate(v.begin(),v.end(),0.0)/v.size(); };
    std::ofstream report(output+"/report.md");report<<std::fixed<<std::setprecision(3);
    report<<"# 普通 BC 双目前端性能对比\n\n对比："<<old_name<<" / "<<new_name<<"。\n\n输入：`"<<lp<<"` / `"<<rp<<"`，"<<left.cols<<"×"<<left.rows
      <<"。CPU ONNX，模型 `"<<static_cast<std::string>(yn["model_path"])<<"`。OpenCV "<<CV_VERSION
      <<"，OpenCV 线程数 "<<cv::getNumThreads()<<"，YOLO intra-op 线程数 "<<yc.intra_op_threads
      <<"，硬件并发数 "<<std::thread::hardware_concurrency()<<"。Release 优化构建。\n\n"
      <<"每条路径预热 10 次，随后各测量 "<<iterations<<" 次，每轮交替先后顺序。图像常驻内存，包含 YOLO、灰度转换、ROI 裁剪和特征/匹配；不包含模型加载、磁盘读写、可视化、PnP、融合。所有光流路径保留匹配检查和按需粗搜索，使用同坐标初值，无物理位姿初值。";
    report<<(roi_compare?"两条路径都仅左图 YOLO+BC，唯一变量为全图/ROI 金字塔。":"旧路径保留左右 YOLO 并行和双侧 BC。")<<"\n\n"
      <<"| 阶段 | 原路径均值 ms | 光流路径均值 ms | 原路径中位数 ms | 光流路径中位数 ms | 原路径 P95 ms | 光流路径 P95 ms |\n|---|---:|---:|---:|---:|---:|---:|\n";
    for(auto entry:std::vector<std::pair<std::string,double Sample::*>>{{"YOLO",&Sample::yolo},{"灰度/裁剪/BC/匹配",&Sample::features},
        {"光流匹配（特征阶段子项）",&Sample::match},{"金字塔（匹配子项）",&Sample::pyramid},{"LK（匹配子项）",&Sample::lk},
        {"裁剪（匹配子项）",&Sample::crop},{"总计",&Sample::total}}) {
      const auto a=values(old_samples,entry.second),b=values(new_samples,entry.second);
      report<<'|'<<entry.first<<'|'<<mean(a)<<'|'<<mean(b)<<'|'<<quantile(a,.5)<<'|'<<quantile(b,.5)<<'|'<<quantile(a,.95)<<'|'<<quantile(b,.95)<<"|\n";
    }
    const double a=mean(values(old_samples,&Sample::total)),b=mean(values(new_samples,&Sample::total));
    report<<"\n总耗时均值减少 "<<(1-b/a)*100<<"%，加速比 "<<a/b<<"；对应串行前端处理能力 "<<1000/a<<" → "<<1000/b<<" 帧/秒。\n\n";
    for(auto entry:std::vector<std::pair<std::string,const std::vector<Sample>*>>{{old_name,&old_samples},{new_name,&new_samples}}) {
      int ok=0,lo=100000,hi=0;std::vector<double> ds;
      for(const auto& s:*entry.second) { ok+=s.ok;lo=std::min(lo,s.pairs);hi=std::max(hi,s.pairs);ds.push_back(s.disparity); }
      report<<entry.first<<"：匹配成功 "<<ok<<'/'<<iterations<<"，点数 "<<lo<<"–"<<hi<<"，视差中位数 "<<quantile(ds,.5)<<" px。\n\n";
    }
    int retries=0;for(const auto& s:new_samples) retries+=s.full_retry;
    report<<"\nROI 路径全图回退次数："<<retries<<"。搜索像素面积："<<new_samples.front().roi_area<<" / "<<left.rows*left.cols<<"。\n\n";
    const auto after=yolo.inferenceCounts();report<<"测量阶段 YOLO 总调用：左 "<<after.left-before.left<<"、右 "<<after.right-before.right
      <<"。\n\n单张人工图对反复处理的前端微基准，不能代表真实连续序列、GPU 环境或完整系统 FPS，不评价物理位姿精度。原始结果见 samples.csv。\n";
    std::cout<<"Report: "<<output<<"/report.md\n";return 0;
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
