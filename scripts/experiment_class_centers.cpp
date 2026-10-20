// Independent class0/class1 runs on the identical reference image.
// Run through scripts/run_class_center_experiment.py inside the Docker toolchain.
#include "tracker/MonoTracker.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/TinyTargetPolicy.hpp"
#include "pose/MonoPnPSolver.hpp"
#include "utils/AsyncImageSaver.hpp"
#include "utils/PoseUtils.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace gpnp;
namespace {
std::string out="output/class_center_alignment_20261020";
cv::Mat Kcv;
Eigen::Matrix3d K;
std::string fmt(double x,int digits=2) { std::ostringstream s;s<<std::fixed<<std::setprecision(digits)<<x;return s.str(); }
std::vector<cv::Point2f> project(const std::vector<Eigen::Vector3d>& p,const Eigen::Matrix3d& R,const Eigen::Vector3d& t) {
    std::vector<cv::Point3f> v;for(auto a:p)v.emplace_back(a.x(),a.y(),a.z());
    cv::Mat rc(3,3,CV_64F),rv;for(int i=0;i<3;++i)for(int j=0;j<3;++j)rc.at<double>(i,j)=R(i,j);
    cv::Rodrigues(rc,rv);std::vector<cv::Point2f> q;
    cv::projectPoints(v,rv,cv::Vec3d(t.x(),t.y(),t.z()),Kcv,cv::noArray(),q);return q;
}
void label(cv::Mat& im,const std::string& s,cv::Point p,cv::Scalar color={245,245,245},double scale=.55) {
    cv::putText(im,s,p,cv::FONT_HERSHEY_SIMPLEX,scale,{0,0,0},3,cv::LINE_AA);
    cv::putText(im,s,p,cv::FONT_HERSHEY_SIMPLEX,scale,color,1,cv::LINE_AA);
}
std::vector<Eigen::Vector3d> axes={{0,0,0},{55,0,0},{0,55,0},{0,0,-55}};
void drawAxes(cv::Mat& im,const PipelineResult& r,int thick=2,bool dashed=false) {
    if(!r.success)return;
    auto q=project(axes,r.R,r.t);
    cv::Scalar colors[]={{0,0,255},{0,255,0},{255,100,20}};
    for(int i=1;i<4;++i) {
        if(dashed) {for(int k=0;k<10;k+=2)cv::line(im,q[0]+(q[i]-q[0])*(k/10.f),q[0]+(q[i]-q[0])*((k+1)/10.f),colors[i-1],thick,cv::LINE_AA);}
        else cv::arrowedLine(im,q[0],q[i],colors[i-1],thick,cv::LINE_AA,0,.12);
        label(im,i==1?"X":i==2?"Y":"-Z",q[i]+cv::Point2f(5,-5),colors[i-1],.45);
    }
    cv::circle(im,q[0],5,{0,255,255},thick,cv::LINE_AA);
}
void metrics(std::ofstream& csv,const std::string& stage,int cls,const PipelineResult& r) {
    auto q=project({{0,0,0}},r.R,r.t);
    double err=0;int count=0;
    for(size_t i=0;i<r.good_matches.size()&&i<r.pts_left_match.size();++i) {
        int id=r.good_matches[i].trainIdx;
        if(id<0||id>=int(r.pose_model_points.size()))continue;
        auto pred=project({r.pose_model_points[id]},r.R,r.t);
        err+=cv::norm(pred[0]-r.pts_left_match[i]);++count;
    }
    csv<<stage<<','<<cls<<','<<r.success<<','<<r.strategy_name<<','<<r.model_frame<<','<<count<<','
       <<r.t.x()<<','<<r.t.y()<<','<<r.t.z()<<','<<q[0].x<<','<<q[0].y<<','<<(count?err/count:0);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)csv<<','<<r.R(i,j);
    csv<<'\n';
}
}
int main() {
    std::filesystem::create_directories(out);
    cv::FileStorage fs("config/tracker_config.json",cv::FileStorage::READ);
    cv::Mat image=cv::imread("data/clean_target2.png");if(image.empty())return 2;
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
    // Primary experiment should take BC for both full-size patterns; preserve TT fallback.
    std::string bdir=std::string(bn["template_dir"]),tdir=std::string(tn["template_dir"]);
    std::ofstream csv(out+"/poses.csv");csv<<std::setprecision(12);
    csv<<"stage,class_id,success,strategy,model_frame,n_points,tx_mm,ty_mm,tz_mm,origin_u_px,origin_v_px,mean_reprojection_px";
    for(int i=0;i<9;++i)csv<<",r"<<i/3<<i%3;
    csv<<'\n';
    PipelineResult runs[2];std::vector<cv::Point2f> annotations[2];
    for(int cls=0;cls<2;++cls) {
        annotations[cls]=readCorners("scripts/class"+std::to_string(cls)+"_points.txt");
        auto box=cv::boundingRect(annotations[cls]);int pad=cls?5:10;
        box=(cv::Rect(box.x-pad,box.y-pad,box.width+2*pad,box.height+2*pad)&cv::Rect(0,0,image.cols,image.rows));
        RoiGroup roi{{box.x,box.y,box.width,box.height},{},false};roi.is_class1=cls;
        // Separate trackers: independent cold starts; no shared pose or seed.
        MonoTracker tracker(K,"data/clean_target2.png",cfg,bc,bdir,tt,tdir);
        tracker.setVerboseConsole(false);tracker.setVisualizeDetailed(true);
        const std::string arm=out+"/class"+std::to_string(cls);std::filesystem::create_directories(arm);
        tracker.setOutputDir(arm);tracker.setFrameNumber(1);cv::setRNGSeed(20261020);
        auto r=tracker.process(image,true,&roi);runs[cls]=r;
        metrics(csv,"tracker",cls,r);
        std::ofstream points(arm+"/observations.csv");points<<"template_id,u_px,v_px,x_mm,y_mm,z_mm\n"<<std::setprecision(12);
        cv::Mat visual=image.clone();cv::rectangle(visual,box,{0,255,255},1);
        for(size_t i=0;i<r.good_matches.size()&&i<r.pts_left_match.size();++i){
            int id=r.good_matches[i].trainIdx;if(id<0||id>=int(r.pose_model_points.size()))continue;
            auto p=r.pose_model_points[id];auto q=r.pts_left_match[i];
            points<<id<<','<<q.x<<','<<q.y<<','<<p.x()<<','<<p.y()<<','<<p.z()<<'\n';
            cv::circle(visual,q,3,{255,255,0},1,cv::LINE_AA);label(visual,std::to_string(id),q+cv::Point2f(4,-4),{255,255,0},.35);
        }
        drawAxes(visual,r);
        cv::copyMakeBorder(visual,visual,100,0,0,0,cv::BORDER_CONSTANT,{25,25,25});
        label(visual,"Class "+std::to_string(cls)+" only | "+r.strategy_name+" | "+(r.success?"SUCCESS":"FAILED"),{15,25});
        label(visual,"t [mm] = ("+fmt(r.t.x())+", "+fmt(r.t.y())+", "+fmt(r.t.z())+")",{15,51});
        label(visual,"Origin = class0 bbox center | red X / green Y / blue -Z",{15,78},{210,210,210},.48);
        cv::imwrite(arm+"/axes.png",visual);
        std::cout<<"CLASS "<<cls<<" success="<<r.success<<" strategy="<<r.strategy_name<<" t="<<r.t.transpose()<<"\n";
    }
    utils::AsyncImageSaver::flush();
    // Diagnostic only: exact physical projections verify the common-origin math,
    // independently of the real-image frontend and its separate model scales.
    Eigen::Matrix3d truthR=(Eigen::AngleAxisd(.14,Eigen::Vector3d::UnitY())*Eigen::AngleAxisd(-.08,Eigen::Vector3d::UnitX())).toRotationMatrix();
    Eigen::Vector3d truthT(25,-35,900);
    for(int cls=0;cls<2;++cls) {
        std::vector<Eigen::Vector3d> model;
        for(auto p:annotations[cls])model.emplace_back(p.x/image.cols*cfg.template_real_width_mm,p.y/image.rows*cfg.template_real_height_mm,0);
        placeTemplateInClass0(model,cls,cfg.template_geometry);
        auto q=project(model,truthR,truthT);MonoPnPSolver solver(cfg.mono_pose_prior);auto p=solver.solve(q,model,K);
        PipelineResult r;r.success=p.success;r.R=p.R;r.t=p.t;r.strategy_name="ideal_projection_control";r.model_frame="class0";r.pose_model_points=model;r.pts_left_match=q;
        for(int i=0;i<int(q.size());++i)r.good_matches.emplace_back(i,i,0);
        metrics(csv,"ideal_control",cls,r);
    }
    if(!runs[0].success||!runs[1].success)return 3;
    auto q0=project(axes,runs[0].R,runs[0].t),q1=project(axes,runs[1].R,runs[1].t);
    cv::Mat overlay=image.clone();drawAxes(overlay,runs[0],3);drawAxes(overlay,runs[1],2,true);
    cv::line(overlay,q0[0],q1[0],{255,0,255},1,cv::LINE_AA);
    label(overlay,"C0",q0[0]+cv::Point2f(-35,-12),{255,255,0});label(overlay,"C1",q1[0]+cv::Point2f(8,20),{255,255,0});
    cv::Mat zoom;cv::resize(overlay(cv::Rect(280,300,190,190)),zoom,{570,570},0,0,cv::INTER_NEAREST);
    cv::imwrite(out+"/origin_zoom.png",zoom);
    cv::copyMakeBorder(overlay,overlay,110,0,0,0,cv::BORDER_CONSTANT,{25,25,25});
    label(overlay,"Same image: solid = class0, dashed = class1",{15,25});
    label(overlay,"3D origin gap = "+fmt((runs[1].t-runs[0].t).norm())+" mm",{15,52});
    label(overlay,"Image origin gap = "+fmt(cv::norm(q1[0]-q0[0]))+" px",{15,79});
    cv::imwrite(out+"/overlay.png",overlay);
    cv::Mat a=cv::imread(out+"/class0/axes.png"),b=cv::imread(out+"/class1/axes.png"),both;
    cv::hconcat(a,b,both);cv::imwrite(out+"/comparison.png",both);
    // Orthographic camera-coordinate views expose the depth difference hidden
    // by the perspective image. Both panels use the same mm/pixel scale.
    cv::Mat world(600,1100,CV_8UC3,cv::Scalar(28,28,28));
    for(int panel=0;panel<2;++panel) {
        int dim=panel==0?0:1;double mid=.5*(runs[0].t[dim]+runs[1].t[dim]);
        auto map=[&](const Eigen::Vector3d& p){return cv::Point(panel*550+275+int((p[dim]-mid)*1.6),520-int((p.z()-650)*1.6));};
        label(world,panel==0?"Camera X-Z plane [mm]":"Camera Y-Z plane [mm]",{panel*550+20,30});
        for(int z=650;z<=950;z+=50){
            int y=520-int((z-650)*1.6);cv::line(world,{panel*550+65,y},{panel*550+520,y},{65,65,65},1);
            label(world,std::to_string(z),{panel*550+10,y+5},{170,170,170},.4);
        }
        label(world,"Z",{panel*550+10,50},{200,200,200},.45);
        for(int tick=-2;tick<=2;++tick) {
            double v=mid+tick*50;auto p=map(Eigen::Vector3d(dim==0?v:0,dim==1?v:0,650));
            cv::line(world,{p.x,50},{p.x,520},{65,65,65},1);
            label(world,fmt(v,0),{p.x-15,545},{170,170,170},.4);
        }
        cv::line(world,map(runs[0].t),map(runs[1].t),{255,0,255},2,cv::LINE_AA);
        for(int cls=0;cls<2;++cls) {
            auto origin=map(runs[cls].t);cv::circle(world,origin,6,cls?cv::Scalar(0,200,255):cv::Scalar(255,220,0),-1,cv::LINE_AA);
            cv::Scalar colors[]={{0,0,255},{0,255,0},{255,100,20}};
            for(int axis=0;axis<3;++axis)cv::arrowedLine(world,origin,map(runs[cls].t+runs[cls].R*axes[axis+1]),colors[axis],2,cv::LINE_AA,0,.2);
            label(world,"class"+std::to_string(cls),origin+cv::Point(12,17));
        }
        label(world,panel==0?"X ->":"Y ->",{panel*550+470,570},{200,200,200},.45);
    }
    cv::imwrite(out+"/camera_space.png",world);
    std::cout<<"ORIGIN GAP mm="<<(runs[1].t-runs[0].t).norm()<<" px="<<cv::norm(q1[0]-q0[0])<<"\n";
    return 0;
}
