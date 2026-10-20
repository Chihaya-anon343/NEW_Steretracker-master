// Reuse projection/drawing helpers; the reference experiment is not executed.
#define main reference_experiment_main
#include "experiment_class_centers.cpp"
#undef main
#include "detection/YoloDetector.hpp"

void writeRun(cv::FileStorage& file,const PipelineResult& r) {
    file<<"{"<<"success"<<int(r.success)<<"strategy"<<r.strategy_name<<"model_frame"<<r.model_frame
        <<"failure_reason"<<r.failure_reason<<"points"<<int(r.good_matches.size());
    if(r.success) {
        file<<"t_mm"<<"["<<r.t.x()<<r.t.y()<<r.t.z()<<"]"<<"R"<<"[";
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)file<<r.R(i,j);
        file<<"]";
        auto origin=project({{0,0,0}},r.R,r.t);file<<"origin_px"<<"["<<origin[0].x<<origin[0].y<<"]";
        double err=0;int count=0;
        for(size_t i=0;i<r.good_matches.size()&&i<r.pts_left_match.size();++i) {
            int id=r.good_matches[i].trainIdx;
            if(id<0||id>=int(r.pose_model_points.size()))continue;
            err+=cv::norm(project({r.pose_model_points[id]},r.R,r.t)[0]-r.pts_left_match[i]);++count;
        }
        file<<"mean_reprojection_px"<<(count?err/count:0.0);
    }
    file<<"}";
}
cv::Mat panel(cv::Mat im,const std::string& title,const std::string& line2,const std::string& line3) {
    double scale=std::min(850.0/im.rows,800.0/im.cols);
    cv::resize(im,im,{},scale,scale,cv::INTER_AREA);
    int width=std::max(660,im.cols);cv::copyMakeBorder(im,im,110,0,0,width-im.cols,cv::BORDER_CONSTANT,{24,24,24});
    label(im,title,{12,26},{255,255,255},.53);label(im,line2,{12,53},{230,230,230},.49);label(im,line3,{12,81},{230,230,230},.48);
    return im;
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    out=argv[2];std::filesystem::create_directories(out);
    cv::FileStorage fs("config/tracker_config.json",cv::FileStorage::READ);
    auto camera=fs["camera"];K<<double(camera["fx"]),0,double(camera["cx"]),0,double(camera["fy"]),double(camera["cy"]),0,0,1;
    Kcv=cv::Mat(3,3,CV_64F);for(int i=0;i<3;++i)for(int j=0;j<3;++j)Kcv.at<double>(i,j)=K(i,j);
    auto st=fs["strategies"],board=st["board_template"],bn=st["binary_corner"],tn=st["tiny_target"];
    TrackerConfig cfg;cfg.template_real_width_mm=double(board["template_real_width_mm"]);cfg.template_real_height_mm=double(board["template_real_height_mm"]);
    cfg.template_geometry=readTemplateGeometryConfig(fs["template_geometry"]);
    cfg.tiny_max_area=int(st["tiny_max_area"]);cfg.tiny_max_area_class1=int(st["close_range"]["tiny_max_area"]);
    cfg.temporal.enabled=false;cfg.mono_pose_prior.enabled=int(fs["mono_pose_prior"]["enabled"]);
    cfg.mono_pose_prior.max_tilt_deg=double(fs["mono_pose_prior"]["max_tilt_deg"]);
    BinaryCornerExtractor::Config bc;bc.corners=int(bn["corners"]);bc.kernel_size=int(bn["kernel_size"]);bc.scale=double(bn["corner_scale"]);
    bc.target_size={int(bn["target_width"]),int(bn["target_height"])};bc.roi_pad_pixels=int(bn["roi_pad_pixels"]);
    bc.pixel_to_meter_scale_class0=double(bn["pixel_to_meter_scale_class0"]);bc.pixel_to_meter_scale_class1=double(bn["pixel_to_meter_scale_class1"]);
    bc.mono_identity=readMonoCornerIdentityConfig(bn["mono_identity"]);
    TinyTargetExtractor::Config tt;tt.target_size={int(tn["target_width"]),int(tn["target_height"])};tt.scale_factor=int(tn["scale_factor"]);
    tt.square_size_m_class0=double(tn["square_size_m_class0"]);tt.square_size_m_class1=double(tn["square_size_m_class1"]);tt.roi_pad_pixels=int(tn["roi_pad_pixels"]);
    tt.robustness=readTinyTargetRobustnessConfig(tn["robustness"]);
    cfg.roi_selection.tt_verification=readTinyTargetVerificationConfig(fs["roi_selection"]["tt_verification"]);
    YoloConfig yc;yc.model_path=std::string(fs["yolo"]["model_path"]);yc.device=DeviceType::CPU;yc.conf_threshold=float(fs["yolo"]["conf_threshold"]);
    YoloDetector yolo(yc);
    std::ifstream paths(argv[1]);std::string path;int selected=0,scanned=0;
    cv::FileStorage results(out+"/results.json",cv::FileStorage::WRITE|cv::FileStorage::FORMAT_JSON);
    results<<"samples"<<"[";
    std::ofstream audit(out+"/selection_audit.tsv");audit<<"path\tstatus\n";
    while(std::getline(paths,path)&&selected<6) {
        ++scanned;cv::Mat image=cv::imread(path);if(image.empty()){audit<<path<<"\tunreadable\n";continue;}
        std::vector<Detection> detections;if(yolo.detect(image,detections)!=Status::Success){audit<<path<<"\tinference_failed\n";continue;}
        Detection d[2];bool found=false;double score=-1;
        for(const auto& a:detections)if(a.class_id==0)for(const auto& b:detections)if(b.class_id==1) {
            double contained=(a.bbox&b.bbox).area()/std::max(1.f,b.bbox.area());
            if(contained<.95||a.bbox.x<2||a.bbox.y<2||a.bbox.br().x>image.cols-2||a.bbox.br().y>image.rows-2)continue;
            if(std::min(b.bbox.width,b.bbox.height)<4)continue;
            if(a.confidence*b.confidence>score){score=a.confidence*b.confidence;d[0]=a;d[1]=b;found=true;}
        }
        if(!found){audit<<path<<"\tno_complete_contained_pair\n";continue;}
        audit<<path<<"\tselected\n";++selected;
        std::string id="sample_"+std::to_string(selected),folder=out+"/"+id;std::filesystem::create_directories(folder);
        PipelineResult r[2];cv::Rect boxes[2];cv::Mat visual[2];
        results<<"{"<<"id"<<id<<"path"<<path<<"width"<<image.cols<<"height"<<image.rows;
        results<<"detections"<<"[";
        for(int cls=0;cls<2;++cls) {
            // Preserve the detected class-specific box; pad selection remains
            // in MonoTracker, exactly as in the production independent route.
            auto b=d[cls].bbox;int x=std::max(0,int(std::floor(b.x))),y=std::max(0,int(std::floor(b.y)));
            int right=std::min(image.cols,int(std::ceil(b.br().x))),bottom=std::min(image.rows,int(std::ceil(b.br().y)));
            boxes[cls]={x,y,right-x,bottom-y};
            results<<"{"<<"class_id"<<cls<<"confidence"<<d[cls].confidence<<"roi"<<"["<<x<<y<<right-x<<bottom-y<<"]"<<"}";
        }
        results<<"]";
        for(int cls=0;cls<2;++cls) {
            auto b=boxes[cls];RoiGroup roi{{b.x,b.y,b.width,b.height},{},false};roi.is_class1=cls;
            MonoTracker tracker(K,"data/clean_target2.png",cfg,bc,std::string(bn["template_dir"]),tt,std::string(tn["template_dir"]));
            tracker.setVerboseConsole(false);tracker.setVisualizeDetailed(false);
            std::string arm=folder+"/class"+std::to_string(cls);std::filesystem::create_directories(arm);
            tracker.setOutputDir(arm);tracker.setFrameNumber(1);cv::setRNGSeed(20261020);
            r[cls]=tracker.process(image,true,&roi);
            results<<(cls?"class1":"class0");writeRun(results,r[cls]);
            cv::Mat im=image.clone();cv::rectangle(im,b,{0,255,255},2);
            for(auto p:r[cls].pts_left_match)cv::circle(im,p,3,{255,255,0},2,cv::LINE_AA);
            if(r[cls].success)drawAxes(im,r[cls],3);
            std::string t=r[cls].success?"t mm = ("+fmt(r[cls].t.x())+", "+fmt(r[cls].t.y())+", "+fmt(r[cls].t.z())+")":"No accepted pose";
            visual[cls]=panel(im,id+" | class"+std::to_string(cls)+" | "+(r[cls].success?"SUCCESS ":"FAILED ")+r[cls].strategy_name,t,
                r[cls].success?"Origin: class0 | red X / green Y / blue -Z":r[cls].failure_reason);
            cv::imwrite(arm+"/axes.png",visual[cls]);
        }
        cv::Mat both;cv::hconcat(visual[0],visual[1],both);cv::imwrite(folder+"/comparison.png",both);
        cv::Mat overlay=image.clone();cv::rectangle(overlay,boxes[0],{0,255,255},2);cv::rectangle(overlay,boxes[1],{255,255,0},2);
        drawAxes(overlay,r[0],4);drawAxes(overlay,r[1],3,true);
        std::string subtitle="Independent cold starts | at least one pose failed";
        if(r[0].success&&r[1].success) {
            auto a=project(axes,r[0].R,r[0].t),b=project(axes,r[1].R,r[1].t);
            cv::line(overlay,a[0],b[0],{255,0,255},2,cv::LINE_AA);
            double gap=(r[1].t-r[0].t).norm(),px=cv::norm(a[0]-b[0]);
            results<<"distance_mm"<<gap<<"image_gap_px"<<px<<"delta_z_mm"<<(r[1].t.z()-r[0].t.z());
            subtitle="Origin gap: "+fmt(gap)+" mm | "+fmt(px)+" px";
        }
        results<<"}";
        cv::imwrite(folder+"/overlay.png",panel(overlay,id+" | solid: class0 / dashed: class1",subtitle,"Same image and K; +X red, +Y green, -Z blue"));
        // Target-detail crop, especially useful when the full frame is tall.
        cv::Rect crop=boxes[0];int pad=std::max(20,crop.width/8);crop=(cv::Rect(crop.x-pad,crop.y-pad,crop.width+2*pad,crop.height+2*pad)&cv::Rect(0,0,image.cols,image.rows));
        cv::imwrite(folder+"/target_detail.png",panel(overlay(crop).clone(),id+" | target detail",subtitle,"Detected boxes: class0 yellow / class1 cyan"));
        std::cout<<id<<" "<<path<<" c0="<<r[0].success<<" c1="<<r[1].success<<std::endl;
    }
    results<<"]"<<"scanned"<<scanned<<"selected"<<selected;
    utils::AsyncImageSaver::flush();return selected?0:3;
}
