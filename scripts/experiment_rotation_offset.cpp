#define main reference_experiment_main
#include "experiment_class_centers.cpp"
#undef main

int main(){
    out="output/rotation_offset_20261020";std::filesystem::create_directories(out);
    cv::FileStorage fs("config/tracker_config.json",cv::FileStorage::READ);
    auto cam=fs["camera"],st=fs["strategies"],bn=st["binary_corner"],tn=st["tiny_target"];
    K<<double(cam["fx"]),0,double(cam["cx"]),0,double(cam["fy"]),double(cam["cy"]),0,0,1;
    Kcv=cv::Mat(3,3,CV_64F);for(int i=0;i<3;++i)for(int j=0;j<3;++j)Kcv.at<double>(i,j)=K(i,j);
    TrackerConfig cfg;cfg.template_real_width_mm=double(st["board_template"]["template_real_width_mm"]);cfg.template_real_height_mm=double(st["board_template"]["template_real_height_mm"]);
    cfg.template_geometry=readTemplateGeometryConfig(fs["template_geometry"]);cfg.temporal.enabled=false;
    cfg.tiny_max_area=int(st["tiny_max_area"]);cfg.tiny_max_area_class1=int(st["close_range"]["tiny_max_area"]);
    cfg.mono_pose_prior.enabled=int(fs["mono_pose_prior"]["enabled"]);cfg.mono_pose_prior.max_tilt_deg=double(fs["mono_pose_prior"]["max_tilt_deg"]);
    BinaryCornerExtractor::Config bc;bc.corners=int(bn["corners"]);bc.kernel_size=int(bn["kernel_size"]);bc.scale=double(bn["corner_scale"]);
    bc.target_size={int(bn["target_width"]),int(bn["target_height"])};bc.roi_pad_pixels=int(bn["roi_pad_pixels"]);
    bc.pixel_to_meter_scale_class0=double(bn["pixel_to_meter_scale_class0"]);bc.pixel_to_meter_scale_class1=double(bn["pixel_to_meter_scale_class1"]);bc.mono_identity=readMonoCornerIdentityConfig(bn["mono_identity"]);
    TinyTargetExtractor::Config tt;tt.target_size={int(tn["target_width"]),int(tn["target_height"])};tt.scale_factor=int(tn["scale_factor"]);
    tt.square_size_m_class0=double(tn["square_size_m_class0"]);tt.square_size_m_class1=double(tn["square_size_m_class1"]);tt.roi_pad_pixels=int(tn["roi_pad_pixels"]);tt.robustness=readTinyTargetRobustnessConfig(tn["robustness"]);
    cfg.roi_selection.tt_verification=readTinyTargetVerificationConfig(fs["roi_selection"]["tt_verification"]);
    auto source=cv::imread("data/clean_target2.png");std::vector<cv::Point2f> anno[2]={readCorners("scripts/class0_points.txt"),readCorners("scripts/class1_points.txt")};
    cv::Point2f center(371.98f,377.35f);double sx=cfg.template_real_width_mm/source.cols,sy=cfg.template_real_height_mm/source.rows;
    std::vector<Eigen::Vector3d> physical[2];for(int cls=0;cls<2;++cls)for(auto p:anno[cls])physical[cls].emplace_back((p.x-center.x)*sx,(p.y-center.y)*sy,0);
    std::ofstream csv(out+"/rotation.csv");csv<<std::setprecision(12)<<"angle,c0_success,c1_success,raw_t_x,raw_t_y,raw_t_z,placed_t_x,placed_t_y,placed_t_z,actual_shift_x,actual_shift_y,actual_shift_z,expected_shift_x,expected_shift_y,expected_shift_z,shift_residual_mm,pair_gap_mm,c0_rotation_error_deg,c1_rotation_error_deg,ideal_origin_error_mm\n";
    cv::Mat gallery(1200,2400,CV_8UC3,cv::Scalar(24,24,24));int count=0;
    for(int angle:{0,45,90,135,180,225,270,315}){
        Eigen::Matrix3d truthR=Eigen::AngleAxisd(angle*CV_PI/180,Eigen::Vector3d::UnitZ()).toRotationMatrix();Eigen::Vector3d truthT(100,0,650);
        std::vector<cv::Point2f> corners={{0,0},{float(source.cols-1),0},{float(source.cols-1),float(source.rows-1)},{0,float(source.rows-1)}};
        std::vector<Eigen::Vector3d> cornerModel;for(auto p:corners)cornerModel.emplace_back((p.x-center.x)*sx,(p.y-center.y)*sy,0);
        auto dest=project(cornerModel,truthR,truthT);auto H=cv::getPerspectiveTransform(corners,dest);cv::Mat image;
        cv::warpPerspective(source,image,H,{1744,1744},cv::INTER_LINEAR,cv::BORDER_CONSTANT,{20,20,20});
        std::string folder=out+"/angle_"+std::to_string(angle);std::filesystem::create_directories(folder);
        PipelineResult r[2],raw;
        for(int cls=0;cls<2;++cls){
            auto q=project(physical[cls],truthR,truthT);auto box=cv::boundingRect(q);int pad=cls?5:10;box=cv::Rect(box.x-pad,box.y-pad,box.width+2*pad,box.height+2*pad)&cv::Rect(0,0,image.cols,image.rows);
            RoiGroup roi{{box.x,box.y,box.width,box.height},{},false};roi.is_class1=cls;
            for(int enabled=(cls?0:1);enabled<=1;++enabled){
                auto c=cfg;if(!enabled)c.template_geometry.class1_center_in_class0_mm.setZero();
                MonoTracker tr(K,"data/clean_target2.png",c,bc,std::string(bn["template_dir"]),tt,std::string(tn["template_dir"]));tr.setVerboseConsole(false);cv::setRNGSeed(20261020);
                auto p=tr.process(image,false,&roi);if(enabled)r[cls]=p;else raw=p;
            }
        }
        Eigen::Vector3d delta(cfg.template_geometry.class1_center_in_class0_mm.x(),cfg.template_geometry.class1_center_in_class0_mm.y(),0);
        auto shift=Eigen::Vector3d(r[1].t-raw.t),expected=Eigen::Vector3d(-r[1].R*delta);
        auto rotationError=[&](const Eigen::Matrix3d& R){return std::acos(std::clamp(((R.transpose()*truthR).trace()-1)*.5,-1.,1.))*180/CV_PI;};
        // Exact correspondences, same configured origin and physical board scale.
        auto idealModel=physical[1];placeTemplateInClass0(idealModel,true,cfg.template_geometry);MonoPnPSolver solver(cfg.mono_pose_prior);
        auto ideal=solver.solve(project(idealModel,truthR,truthT),idealModel,K);
        csv<<angle<<','<<r[0].success<<','<<r[1].success;
        for(auto p:{raw.t,r[1].t,shift,expected})for(int i=0;i<3;++i)csv<<','<<p[i];
        csv<<','<<(shift-expected).norm()<<','<<(r[1].t-r[0].t).norm()<<','<<rotationError(r[0].R)<<','<<rotationError(r[1].R)<<','<<(ideal.t-truthT).norm()<<'\n';
        drawAxes(image,r[0],4);drawAxes(image,r[1],3,true);
        cv::Mat im=image(cv::Rect(270,350,940,940)).clone();cv::resize(im,im,{600,600});
        cv::rectangle(im,{0,0,600,75},{24,24,24},-1);
        label(im,std::to_string(angle)+" deg | solid C0 / dashed C1",{10,25});
        label(im,"Pair gap "+fmt((r[1].t-r[0].t).norm())+" mm",{10,52});
        cv::imwrite(folder+"/overlay.png",im);im.copyTo(gallery(cv::Rect(count%4*600,count/4*600,600,600)));++count;
        std::cout<<angle<<" residual="<<(shift-expected).norm()<<" rotation errors="<<rotationError(r[0].R)<<","<<rotationError(r[1].R)<<std::endl;
    }
    cv::imwrite(out+"/gallery.png",gallery);return 0;
}
