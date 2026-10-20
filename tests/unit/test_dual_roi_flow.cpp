#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/MonoDualRoiPolicy.hpp"
#include "common/StereoRobustness.hpp"
#include "tracker/MonoTracker.hpp"
#include "pose/GPnPSolver.hpp"
#include "pose/InitialPnPSolver.hpp"
#include "common/LogConfig.hpp"
#include "tracker/StereoTracker.hpp"
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>

using namespace gpnp;
namespace {
StereoCameraParams calibration() {
    Eigen::Matrix3d k;k<<900,0,320,0,900,240,0,0,1;
    return makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),{60,0,0});
}
struct DualFixture {
    std::filesystem::path root;
    cv::Mat left,right;
    RoiGroup left_roi{{180,100,280,280},{295,215,50,50},true};
    RoiGroup right_roi{{120,100,280,280},{235,215,50,50},true};
    TrackerConfig config=makeTrackerConfig(4,true,400,400,800,0,0);
    BinaryCornerExtractor::Config bc;
    TinyTargetExtractor::Config tt;
    DualFixture(double dy=0) {
        root=std::filesystem::path("tests/output")/("dual_lk_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const std::vector<cv::Point> polygon{{40,5},{60,5},{60,35},{95,35},{95,65},{60,65},{60,95},{40,95},{40,65},{5,50}};
        cv::Mat mask(100,100,CV_8UC1,cv::Scalar(0));
        cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{polygon},255);
        cv::Mat template_mask;cv::resize(mask,template_mask,{50,50},0,0,cv::INTER_NEAREST);
        cv::imwrite((root/"0_degrees.png").string(),template_mask);
        std::vector<cv::Point2f> corners;
        cv::Point2f center(0,0);
        for(auto p:polygon){corners.emplace_back(p);center+=cv::Point2f(p);}
        center*=1.f/corners.size();std::reverse(corners.begin(),corners.end());
        const auto order=BinaryCornerExtractor::reorderByGeometry(corners,center,0,-1);
        std::ofstream tmpl(root/"0_degrees.txt"),c0(root/"class0.txt"),c1(root/"class1.txt");
        for(size_t i=0;i<order.size();++i) {
            const auto p=corners[order[i]];
            tmpl<<"Corner_"<<i<<": "<<p.x*.5<<", "<<p.y*.5<<'\n';
            c0<<"Corner_"<<i<<": "<<50+p.x*3<<", "<<50+p.y*3<<'\n';
            c1<<"Corner_"<<i<<": "<<180+p.x*.4<<", "<<180+p.y*.4<<'\n';
        }
        cv::Mat board(400,400,CV_8UC1,cv::Scalar(0)),large,small;
        cv::resize(mask,large,{300,300},0,0,cv::INTER_NEAREST);large.copyTo(board(cv::Rect(50,50,300,300)));
        board(cv::Rect(175,175,50,50)).setTo(0);
        cv::resize(mask,small,{40,40},0,0,cv::INTER_NEAREST);small.copyTo(board(cv::Rect(180,180,40,40)));
        cv::imwrite((root/"board.png").string(),board);
        cv::Mat scene(480,640,CV_8UC1,cv::Scalar(0));board.copyTo(scene(cv::Rect(120,40,400,400)));
        left=cv::Mat(scene.size(),CV_8UC1);cv::RNG rng(87131);
        // Distinct patches without changing the binary contour or template geometry.
        for(int y=0;y<scene.rows;++y)for(int x=0;x<scene.cols;++x)
            left.at<uchar>(y,x)=static_cast<uchar>((scene.at<uchar>(y,x)?175:0)+rng.uniform(0,80));
        cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-60,0,1,dy);
        cv::warpAffine(left,right,transform,left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
        config.temporal.enabled=false;config.stereo_flow.enabled=true;config.stereo_flow.dual_roi_flow=true;
        config.stereo_flow.dual_roi_legacy_fallback=false;config.stereo_flow.roi_pyramid=true;
        config.stereo_flow.max_epipolar_error_px=3;config.dual_roi_class0_points_path=(root/"class0.txt").string();
        config.dual_roi_class1_points_path=(root/"class1.txt").string();
        bc.corners=10;bc.scale=1;bc.roi_pad_pixels=0;
        bc.pixel_to_meter_scale_class0=.013; // Deliberately wrong for the board: must not enter Tier1/1b.
        bc.pixel_to_meter_scale_class1=.0008;
        tt.square_size_m_class0=.4;tt.square_size_m_class1=.038;tt.roi_pad_pixels=0;
    }
    ~DualFixture(){utils::AsyncImageSaver::flush();std::filesystem::remove_all(root);}
    std::unique_ptr<StereoTracker> tracker() {
        auto c=calibration();auto tracker=std::make_unique<StereoTracker>(c.K,c.R_rl,c.t_rl,
            (root/"board.png").string(),config,bc,root.string(),tt,root.string());
        tracker->setOutputDir(root.string());tracker->setVerboseConsole(false);return tracker;
    }
    void exportPreview() {
        utils::AsyncImageSaver::flush();const std::filesystem::path destination("output/dual_roi_flow_experiment/synthetic");
        std::filesystem::create_directories(destination);
        for(const auto& p:std::filesystem::directory_iterator(root))if(p.path().filename().string().find("_f")!=std::string::npos)
            std::filesystem::copy_file(p.path(),destination/p.path().filename(),std::filesystem::copy_options::overwrite_existing);
    }
};
void report(const PipelineResult& r) {
    if(!r.success&&!r.disparity_success)std::cerr<<"Dual result: "<<r.failure_reason<<" route="<<r.route_detail
        <<" c0="<<r.dual_c0_pairs<<" c1="<<r.dual_c1_pairs<<'\n';
}
void test_mono_dual_policy_is_inert_in_stereo() {
    DualFixture f;f.config.stereo_flow.disparity_only=true;
    auto a=f.tracker();auto before=a->process(f.left,f.right,false,&f.left_roi,nullptr);
    f.config.mono_dual_roi.enabled=true;
    auto b=f.tracker();auto after=b->process(f.left,f.right,false,&f.left_roi,nullptr);
    TEST_ASSERT(before.disparity_success&&after.disparity_success);
    TEST_ASSERT(before.pts_left_good==after.pts_left_good);
    TEST_ASSERT(before.pts_right_good==after.pts_right_good);
    TEST_ASSERT(before.disparity==after.disparity);
    TEST_ASSERT(before.strategy_name==after.strategy_name);
}
REGISTER_TEST(test_mono_dual_policy_is_inert_in_stereo);
void test_mono_dual_rejects_joint_that_degrades_primary() {
    DualFixture f;f.config.mono_dual_roi.enabled=true;
    f.bc.mono_identity.enabled=true;f.bc.mono_identity.frontend="normalized";
    f.bc.mono_identity.quality_filter=true;
    MonoTracker tracker(calibration().K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    tracker.setVerboseConsole(false);auto r=tracker.process(f.left,false,&f.left_roi);
    TEST_ASSERT(r.success);TEST_ASSERT(r.strategy_name=="DualRoi_BC");
    TEST_ASSERT(r.route_detail.find("primary_reprojection_worse")!=std::string::npos);
    // Rasterized small-target corners have different sampling errors. The
    // geometric board is consistent, but the joint measurement fit is worse.
    // With the new gate explicitly ablated the same candidate is accepted.
    f.config.mono_dual_roi.joint_quality_check=false;
    MonoTracker unguarded(calibration().K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    unguarded.setVerboseConsole(false);auto q=unguarded.process(f.left,false,&f.left_roi);
    TEST_ASSERT(q.success&&q.strategy_name=="DualRoi_BCJ");
    TEST_ASSERT(q.pts_left_match.size()>10);
    std::vector<Eigen::Vector3d> primary_model;
    for(auto m:r.good_matches)primary_model.push_back(r.pose_model_points[m.trainIdx]);
    PoseEstimate a,b;a.success=b.success=true;a.R=r.R;a.t=r.t;b.R=q.R;b.t=q.t;
    TEST_ASSERT(monoGroupReprojection(b,r.pts_left_match,primary_model,calibration().K)>
        monoGroupReprojection(a,r.pts_left_match,primary_model,calibration().K)+.25);
    TEST_ASSERT(r.n_template_match==(int)r.good_matches.size());
    std::vector<bool> used(20,false);
    for(auto m:q.good_matches){TEST_ASSERT(m.trainIdx>=0&&m.trainIdx<20);TEST_ASSERT(!used[m.trainIdx]);used[m.trainIdx]=true;}
    TEST_ASSERT(std::abs(r.t.z()-900)<25);
}
REGISTER_TEST(test_mono_dual_rejects_joint_that_degrades_primary);
void test_mono_dual_class1_bc_uses_annotation_model_without_scale_leak() {
    DualFixture f;f.config.mono_dual_roi.enabled=true;
    auto group=f.left_roi;group.primary={0,0,640,480};
    MonoTracker tracker(calibration().K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    tracker.setVerboseConsole(false);
    auto r=tracker.process(f.left,false,&group);
    TEST_ASSERT(r.success);TEST_ASSERT(r.strategy_name=="DualRoi_C1BC");
    TEST_ASSERT(r.is_class1&&r.pose_model_points.size()==10);
    TEST_ASSERT(r.n_template_match==(int)r.good_matches.size());
    // Annotation span is 36mm; BC scale setting would instead give 36px here.
    double lo=1e9,hi=-1e9;for(auto p:r.pose_model_points){lo=std::min(lo,p.x());hi=std::max(hi,p.x());}
    TEST_ASSERT_NEAR(hi-lo,36,1e-4);
    const auto first=r.pts_left_match;
    tracker.clearCache();auto primary=tracker.process(f.left,false,&f.left_roi);
    TEST_ASSERT(primary.success&&!primary.is_class1);
    tracker.clearCache();auto again=tracker.process(f.left,false,&group);
    TEST_ASSERT(again.success&&again.strategy_name==r.strategy_name);
    TEST_ASSERT(again.pts_left_match==first);
}
REGISTER_TEST(test_mono_dual_class1_bc_uses_annotation_model_without_scale_leak);
void test_mono_dual_tiny_class1_goes_directly_to_guarded_tt() {
    DualFixture f;f.config.mono_dual_roi.enabled=true;f.config.tiny_max_area_class1=3000;
    f.tt.robustness.enabled=true;
    auto group=f.left_roi;group.primary={0,0,640,480};
    MonoTracker tracker(calibration().K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    tracker.setVerboseConsole(false);auto r=tracker.process(f.left,false,&group);
    TEST_ASSERT(r.success&&r.strategy_name=="DualRoi_C1TT");
    TEST_ASSERT(r.pts_left_match.size()==4&&r.pose_model_points.size()==4);
    TEST_ASSERT(r.route_detail.find("BC:skip_tiny_roi")!=std::string::npos);
    TEST_ASSERT(r.tt_status=="success");
}
REGISTER_TEST(test_mono_dual_tiny_class1_goes_directly_to_guarded_tt);
void test_dual_pose_uses_board_model_and_augments_class1() {
    DualFixture f;auto tracker=f.tracker();tracker->setFrameNumber(11);
    auto r=tracker->process(f.left,f.right,true,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.pose_source==PoseSource::StereoGpnp);
    TEST_ASSERT_EQ(r.strategy_name,std::string("DualRoi_BCJ"));
    TEST_ASSERT(r.dual_c0_pairs>=6&&r.dual_c1_pairs>=4);
    TEST_ASSERT(std::abs(r.t.z()-900)<25);
    TEST_ASSERT_EQ(r.model_frame,std::string("board"));
    TEST_ASSERT_EQ(r.flow_pyramid_builds,1);
    for(size_t i=0;i<r.flow_regions.size();++i) {
        const int raw=r.idx_from_filtered[i];
        TEST_ASSERT(cv::norm(r.pts_left_good[i]-r.pts_left_match[raw])<1e-5);
        if(r.flow_regions[i]==0)TEST_ASSERT_EQ(r.flow_template_indices[i],r.flow_original_indices[i]);
        else TEST_ASSERT(r.flow_template_indices[i]>=10&&r.flow_template_indices[i]<20);
    }
    utils::AsyncImageSaver::flush();
    for(const auto* name:{"stereo_f11.png","dual_roi_left_f11.png","dual_roi_right_f11.png",
        "dual_roi_binary_f11.png","dual_roi_correspondence_f11.png","dual_roi_reproj_f11.png",
        "dual_roi_bc_process_f11.png","dual_roi_c0_flow_status_f11.png","dual_roi_c1_flow_status_f11.png"})
        TEST_ASSERT(!cv::imread((f.root/name).string()).empty());
    TEST_ASSERT_EQ(tracker->frameCount(),1);TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    f.exportPreview();
}
void test_dual_disparity_only_has_no_pnp_and_exports_region_ids() {
    DualFixture f(1.774);f.config.stereo_flow.disparity_only=true;auto tracker=f.tracker();
    tracker->setFrameNumber(12);auto r=tracker->process(f.left,f.right,true,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.disparity_success&&!r.success&&!r.gpnp_success&&!r.stereo_success);
    TEST_ASSERT(r.pnp_ms==0&&r.timing["initial_pnp"]==0&&r.timing["gpnp"]==0);
    TEST_ASSERT_EQ(r.flow_pyramid_builds,1);
    TEST_ASSERT(r.dual_c0_pairs>=6&&r.dual_c1_pairs>=6);
    for(size_t i=0;i<r.disparity.size();++i) {
        TEST_ASSERT(std::abs(r.disparity[i]-60)<.2);
        TEST_ASSERT(std::abs(r.pts_right_good[i].y-r.pts_left_good[i].y-1.774)<.2);
        if(r.flow_regions[i]==1)TEST_ASSERT_EQ(r.flow_template_indices[i],-1);
    }
    std::ifstream csv(f.root/"disparity_f12.csv");std::string header;std::getline(csv,header);
    TEST_ASSERT_EQ(header,std::string("left_index,left_x,left_y,right_x,right_y,disparity_px,region,original_index,template_index,model_frame"));
    TEST_ASSERT(tracker->getLogs()[0].disparity_success&&tracker->getLogs()[0].pnp_ms==0);
    f.exportPreview();
}
void test_dual_occlusion_preserves_original_corner_and_model_ids() {
    DualFixture f;f.config.stereo_flow.disparity_only=true;
    f.right(cv::Rect(104,221,43,43)).setTo(30);auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.disparity_success);
    if(r.dual_c0_pairs==0||r.dual_c0_pairs==10)std::cerr<<"Occlusion c0="<<r.dual_c0_pairs<<" c1="<<r.dual_c1_pairs<<" route="<<r.route_detail<<'\n';
    TEST_ASSERT(r.dual_c0_pairs>0&&r.dual_c0_pairs<10);
    bool nonprefix=false;
    for(size_t i=0;i<r.flow_regions.size();++i)if(r.flow_regions[i]==0) {
        TEST_ASSERT_EQ(r.flow_template_indices[i],r.flow_original_indices[i]);
        if(r.flow_original_indices[i]!=static_cast<int>(i))nonprefix=true;
        TEST_ASSERT(cv::norm(r.pts_left_match[r.idx_from_filtered[i]]-r.pts_left_good[i])<1e-5);
    }
    TEST_ASSERT(nonprefix);
}
void test_dual_bad_class1_keeps_class0_pose() {
    DualFixture f;f.right(cv::Rect(235,215,50,50)).setTo(30);auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success);
    TEST_ASSERT_EQ(r.strategy_name,std::string("DualRoi_BC"));TEST_ASSERT_EQ(r.dual_c1_pairs,0);
    TEST_ASSERT(std::abs(r.t.z()-900)<25);
}
void test_dual_primary_short_circuit_uses_class1_frame() {
    DualFixture f;f.left_roi.primary={0,0,640,480};auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.is_class1&&r.fallback_used);
    TEST_ASSERT_EQ(r.strategy_name,std::string("DualRoi_C1BC"));TEST_ASSERT_EQ(r.model_frame,std::string("class1"));
    TEST_ASSERT_EQ(r.dual_c0_pairs,0);TEST_ASSERT(r.dual_c1_pairs>=6);
    TEST_ASSERT(std::abs(r.t.z()-900)<40);
}
void test_dual_tier1_only_and_missing_annotation() {
    DualFixture f;f.config.dual_roi_tier1_only=true;auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.dual_c1_pairs==0);
    f.config.dual_roi_tier1_only=false;f.config.dual_roi_class1_points_path.clear();
    tracker=f.tracker();r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.dual_c1_pairs==0);
}
void test_dual_model_count_mismatch_does_not_pair_prefix() {
    DualFixture f;f.config.dual_roi_class1_fallback=false;
    {std::ofstream bad(f.root/"bad_class0.txt");for(int i=0;i<9;++i)bad<<"Corner_"<<i<<": "<<70+10*i<<", "<<100+8*i<<'\n';}
    f.config.dual_roi_class0_points_path=(f.root/"bad_class0.txt").string();auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);
    TEST_ASSERT(!r.success&&!r.stereo_success&&r.pts_left_good.empty());
}
void test_dual_lazy_legacy_fallback_counts_one_frame_and_log() {
    DualFixture f;f.config.stereo_flow.dual_roi_legacy_fallback=true;f.config.dual_roi_class1_fallback=false;
    f.right.setTo(30);auto tracker=f.tracker();int calls=0;
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    TEST_ASSERT_EQ(calls,1);TEST_ASSERT(!r.success&&r.legacy_frontend_fallback);
    TEST_ASSERT_EQ(tracker->frameCount(),1);TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    f.config.stereo_flow.dual_roi_legacy_fallback=false;tracker=f.tracker();calls=0;
    tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    TEST_ASSERT_EQ(calls,0);
}
void test_dual_disparity_never_calls_legacy_pose_route() {
    DualFixture f;f.config.stereo_flow.dual_roi_legacy_fallback=true;f.config.stereo_flow.disparity_only=true;
    f.right.setTo(30);auto tracker=f.tracker();int calls=0;
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    TEST_ASSERT_EQ(calls,1);TEST_ASSERT(!r.disparity_success&&r.pnp_ms==0&&!r.success);
    TEST_ASSERT(r.legacy_frontend_fallback&&r.route_detail.find("C1TT:")!=std::string::npos);
}
void test_dual_tt_disparity_fallback_has_no_pose_solver() {
    DualFixture f;f.left_roi.primary={0,0,640,480};f.config.stereo_flow.disparity_only=true;
    f.config.stereo_flow.dual_roi_legacy_fallback=true;f.bc.kernel_size=51;
    auto tracker=f.tracker();int calls=0;tracker->setFrameNumber(13);
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});report(r);
    TEST_ASSERT_EQ(calls,1);TEST_ASSERT(r.disparity_success&&!r.success&&r.pnp_ms==0);
    TEST_ASSERT_EQ(r.strategy_name,std::string("DualRoi_C1TT"));TEST_ASSERT(r.legacy_frontend_fallback);
    TEST_ASSERT_EQ(r.dual_c1_pairs,4);TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    for(double d:r.disparity)TEST_ASSERT(std::abs(d-60)<.2);
    tracker=f.tracker();
    r=tracker->process(f.left,f.right,false,&f.left_roi,&f.right_roi);
    TEST_ASSERT(r.disparity_success&&r.strategy_name=="DualRoi_C1TT"&&r.pnp_ms==0);
}
void test_dual_legacy_tt_requires_explicit_left_pose_permission() {
    DualFixture f;f.left_roi.primary={0,0,640,480};f.bc.kernel_size=51;
    f.config.stereo_flow.dual_roi_legacy_fallback=true;f.config.stereo_flow.allow_left_pnp=true;
    auto tracker=f.tracker();
    auto provider=[&](){return std::make_pair(f.left_roi,f.right_roi);};
    tracker->setFrameNumber(15);
    auto r=tracker->process(f.left,f.right,true,&f.left_roi,nullptr,provider);report(r);
    TEST_ASSERT(r.success&&!r.stereo_success&&r.legacy_frontend_fallback);
    TEST_ASSERT(r.pose_source==PoseSource::LeftPnpFallback);TEST_ASSERT_EQ(r.strategy_name,std::string("DualRoi_C1TT"));
    f.exportPreview();
    tracker=f.tracker();
    r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){return std::make_pair(f.left_roi,RoiGroup{f.right_roi.secondary,{},false,true});});
    TEST_ASSERT(r.success&&r.pose_source==PoseSource::LeftPnpFallback&&!r.stereo_success);
    f.config.stereo_flow.allow_left_pnp=false;tracker=f.tracker();
    r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,provider);
    TEST_ASSERT(!r.success&&!r.stereo_success);TEST_ASSERT_EQ(tracker->frameCount(),1);TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
}
void test_dual_config_and_frontend_selection() {
    cv::FileStorage fs("{\"method\":\"left_bc_lk\",\"dual_roi_flow\":true,\"dual_roi_legacy_fallback\":false}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    auto cfg=readStereoFlowConfig(fs.root());TEST_ASSERT(cfg.dual_roi_flow&&!cfg.dual_roi_legacy_fallback);
    cfg.enabled=false;TEST_ASSERT_THROWS(validateStereoFlowConfig(cfg),std::invalid_argument);
    DualFixture f;auto tracker=f.tracker();TEST_ASSERT(tracker->prefersLeftStereoFlow(f.left_roi));
    TEST_ASSERT(!tracker->prefersLeftBcFlow(f.left_roi));
    f.config.stereo_flow.dual_roi_flow=false;tracker=f.tracker();TEST_ASSERT(!tracker->prefersLeftStereoFlow(f.left_roi));
}
void test_frame_context_reuses_pyramids_and_resets_between_frames() {
    DualFixture f;auto c=calibration();StereoFlowConfig cfg;cfg.roi_pyramid=true;cfg.max_epipolar_error_px=3;
    std::vector<cv::Point2f> points{{240,110},{300,110},{450,200},{300,320}},seeds;
    for(auto p:points)seeds.push_back(p-cv::Point2f(60,0));
    StereoFlowFrame frame(c,cfg,f.left,f.right,{180,100,280,280});
    const auto first=frame.match(points,seeds),second=frame.match(points,seeds);
    TEST_ASSERT(first.success&&second.success);TEST_ASSERT_EQ(first.pyramid_builds,1);TEST_ASSERT_EQ(second.pyramid_builds,0);
    TEST_ASSERT(first.left_indices==second.left_indices);
    for(size_t i=0;i<first.right_points.size();++i)TEST_ASSERT(cv::norm(first.right_points[i]-second.right_points[i])<1e-5);
    f.right.setTo(30);StereoFlowFrame new_frame(c,cfg,f.left,f.right,{180,100,280,280});
    TEST_ASSERT(!new_frame.match(points,seeds).success);
}
void test_full_chain_dual_left_tt_precedes_right_detection() {
    DualFixture f;f.left_roi.primary={0,0,640,480};f.bc.kernel_size=51;
    f.config.stereo_flow.disparity_only=true;f.config.stereo_flow.disparity_full_chain=true;
    f.config.stereo_flow.dual_roi_legacy_fallback=true;auto tracker=f.tracker();int calls=0;
    tracker->setFrameNumber(22);
    auto r=tracker->process(f.left,f.right,true,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    report(r);TEST_ASSERT_EQ(calls,0);TEST_ASSERT(r.disparity_success&&r.pnp_ms==0&&!r.success);
    TEST_ASSERT(r.strategy_name=="DualRoi_C1TT"&&!r.legacy_frontend_fallback&&r.fallback_used);
    TEST_ASSERT_EQ(r.dual_c1_pairs,4);TEST_ASSERT(r.route_detail.find("C1TT_LK:OK")!=std::string::npos);
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    for(double d:r.disparity)TEST_ASSERT_NEAR(d,60,.2);
}
void test_full_chain_dual_switches_respect_fallback_and_tier1() {
    DualFixture f;f.bc.kernel_size=51;f.config.stereo_flow.disparity_only=true;
    f.right.setTo(30); // Force failure; a large kernel alone can still leave valid class0 corners.
    f.config.stereo_flow.disparity_full_chain=true;f.config.stereo_flow.dual_roi_legacy_fallback=true;
    f.config.dual_roi_tier1_only=true;auto tracker=f.tracker();int calls=0;
    auto provider=[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);};
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,provider);
    TEST_ASSERT(!r.disparity_success&&calls==0&&r.route_detail.find("C1TT")==std::string::npos);
    f.config.dual_roi_tier1_only=false;f.config.dual_roi_class1_fallback=false;tracker=f.tracker();
    r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,provider);
    TEST_ASSERT(!r.disparity_success&&calls==0&&r.route_detail.find("C1TT")==std::string::npos);
}
void test_full_chain_dual_still_supports_lazy_right_tt_attempt() {
    DualFixture f;f.config.stereo_flow.disparity_only=true;f.config.stereo_flow.disparity_full_chain=true;
    f.config.stereo_flow.dual_roi_legacy_fallback=true;f.right.setTo(30);auto tracker=f.tracker();int calls=0;
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    TEST_ASSERT_EQ(calls,1);TEST_ASSERT(!r.disparity_success&&r.pnp_ms==0&&!r.success);
    TEST_ASSERT(r.legacy_frontend_fallback&&r.route_detail.find("C1TT_LK:")!=std::string::npos);
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
}
void test_full_chain_dual_negative_disparity_does_not_retry_tt() {
    DualFixture f;std::swap(f.left,f.right);std::swap(f.left_roi,f.right_roi);
    f.config.stereo_flow.disparity_only=true;f.config.stereo_flow.disparity_full_chain=true;
    f.config.stereo_flow.dual_roi_legacy_fallback=true;auto tracker=f.tracker();int calls=0;
    auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr,[&](){++calls;return std::make_pair(f.left_roi,f.right_roi);});
    TEST_ASSERT_EQ(calls,0);TEST_ASSERT(!r.disparity_success);
    TEST_ASSERT_EQ(r.failure_reason,std::string("negative_disparity_check_input_order"));
    TEST_ASSERT(r.route_detail.find("C1TT")==std::string::npos);
}
}
namespace {
Eigen::Vector2d modelCenter(const std::vector<Eigen::Vector3d>& points) {
    TEST_ASSERT(!points.empty());
    Eigen::Vector2d lo=points.front().head<2>(),hi=lo;
    for(const auto& p:points){lo=lo.cwiseMin(p.head<2>());hi=hi.cwiseMax(p.head<2>());}
    return (lo+hi)*.5;
}
void test_template_geometry_config_validation_and_legacy_default() {
    cv::FileStorage old("{}",cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
    TEST_ASSERT(!readTemplateGeometryConfig(old["template_geometry"]).enabled);
    cv::FileStorage good("{\"template_geometry\":{\"enabled\":true,\"class1_center_in_class0_mm\":[12.5,-7]}}",
        cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
    const auto c=readTemplateGeometryConfig(good["template_geometry"]);
    TEST_ASSERT(c.enabled);TEST_ASSERT((c.class1_center_in_class0_mm-Eigen::Vector2d(12.5,-7)).norm()<1e-12);
    for(const std::string body:{"{\"enabled\":true}","{\"enabled\":true,\"class1_center_in_class0_mm\":[1,2,3]}",
        "{\"enabled\":true,\"class1_center_in_class0_mm\":[\"bad\",2]}","{\"enabled\":2}"}) {
        cv::FileStorage fs("{\"template_geometry\":"+body+"}",cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
        TEST_ASSERT_THROWS(readTemplateGeometryConfig(fs["template_geometry"]),std::invalid_argument);
    }
    auto bad=c;bad.class1_center_in_class0_mm.x()=std::numeric_limits<double>::infinity();
    TEST_ASSERT_THROWS(validateTemplateGeometry(bad),std::invalid_argument);
}
void test_template_geometry_extractor_centers_and_repeated_switches() {
    DualFixture f;
    TemplateGeometryConfig g;g.enabled=true;g.class1_center_in_class0_mm={12,-7};
    f.bc.template_geometry=g;f.tt.template_geometry=g;
    BinaryCornerExtractor bc(f.bc,f.root.string());TinyTargetExtractor tt(f.tt,f.root.string());
    const auto base=bc.templateData().pts_3d;
    TEST_ASSERT(modelCenter(base).norm()<1e-10);
    for(int i=0;i<4;++i) {
        bc.setUseClass1(true);tt.setUseClass1(true);
        TEST_ASSERT((modelCenter(bc.templateData().pts_3d)-g.class1_center_in_class0_mm).norm()<1e-10);
        TEST_ASSERT((modelCenter(tt.templateData().pts_3d)-g.class1_center_in_class0_mm).norm()<1e-10);
        // Offset must preserve class-specific physical dimensions and point IDs.
        const double ratio=f.bc.pixel_to_meter_scale_class1/f.bc.pixel_to_meter_scale_class0;
        for(size_t j=1;j<base.size();++j)
            TEST_ASSERT(std::abs((bc.templateData().pts_3d[j]-bc.templateData().pts_3d[0]).norm()-ratio*(base[j]-base[0]).norm())<1e-10);
        bc.setUseClass1(false);tt.setUseClass1(false);
        TEST_ASSERT(modelCenter(bc.templateData().pts_3d).norm()<1e-10);
        TEST_ASSERT(modelCenter(tt.templateData().pts_3d).norm()<1e-10);
        for(size_t j=0;j<base.size();++j)TEST_ASSERT((bc.templateData().pts_3d[j]-base[j]).norm()<1e-10);
    }
    // Bbox center differs from vertex mean for an asymmetric polygon.
    std::vector<Eigen::Vector3d> pts{{0,0,0},{10,0,0},{10,10,0},{0,10,0},{1,2,0}};
    auto legacy=pts;placeTemplateInClass0(legacy,true,TemplateGeometryConfig{});
    for(size_t j=0;j<pts.size();++j)TEST_ASSERT((legacy[j]-pts[j]).norm()==0);
    placeTemplateInClass0(pts,true,g);
    TEST_ASSERT((pts[0]-Eigen::Vector3d(7,-12,0)).norm()<1e-12);
}
PipelineResult geometryObservations(const StereoCameraParams& c,const std::vector<Eigen::Vector3d>& model,
                                    const Eigen::Matrix3d& r,const Eigen::Vector3d& t) {
    PipelineResult result;
    for(int i=0;i<static_cast<int>(model.size());++i) {
        const Eigen::Vector3d pl=r*model[i]+t,pr=c.R_rl.transpose()*(pl-c.t_rl);
        const Eigen::Vector3d l=c.K*pl,rr=c.K*pr;
        result.pts_left_match.emplace_back(l.x()/l.z(),l.y()/l.z());
        result.pts_left_good.push_back(result.pts_left_match.back());
        result.pts_right_good.emplace_back(rr.x()/rr.z(),rr.y()/rr.z());
        result.idx_from_filtered.push_back(i);result.good_matches.emplace_back(i,i,0);
        result.disparity.push_back(result.pts_left_good.back().x-result.pts_right_good.back().x);
    }
    return result;
}
void test_template_geometry_gpnp_switches_and_seed_use_one_origin() {
    const auto c=calibration();
    TemplateGeometryConfig geometry;geometry.enabled=true;geometry.class1_center_in_class0_mm={32,-19};
    std::vector<Eigen::Vector3d> c0{{-140,-110,0},{140,-110,0},{140,110,0},{-140,110,0},{-100,-30,0},{60,70,0}};
    std::vector<Eigen::Vector3d> c1{{-20,-15,0},{20,-15,0},{20,15,0},{-20,15,0},{-10,-6,0},{12,7,0}};
    std::vector<Eigen::Vector3d> tt{{-18,-18,0},{18,-18,0},{18,18,0},{-18,18,0}};
    placeTemplateInClass0(c0,false,geometry);placeTemplateInClass0(c1,true,geometry);placeTemplateInClass0(tt,true,geometry);
    auto merged=c0;merged.insert(merged.end(),c1.begin(),c1.end());
    const Eigen::Matrix3d r=(Eigen::AngleAxisd(.31,Eigen::Vector3d::UnitY())*
        Eigen::AngleAxisd(-.19,Eigen::Vector3d::UnitX())).toRotationMatrix();
    const Eigen::Vector3d t(90,-45,1300);
    Eigen::Matrix3d seed_r=r;Eigen::Vector3d seed_t=t;
    std::filesystem::create_directories("output/template_geometry_optimization");
    std::ofstream metrics("output/template_geometry_optimization/synthetic_pose.csv");
    metrics<<"point_count,solver,translation_error_mm,rotation_matrix_error\n";
    for(const auto& model:{c0,c1,merged,tt,c0}) {
        auto data=geometryObservations(c,model,r,t);
        if(model.size()==4) {
            // The production TT path uses left PnP, not InitialPnP or stereo GPnP.
            MonoPnPSolver solver;PoseSeed seed;seed.R=seed_r;seed.t=seed_t;
            for(const PoseSeed* previous:{static_cast<const PoseSeed*>(nullptr),static_cast<const PoseSeed*>(&seed)}) {
                const auto pose=solver.solve(data.pts_left_match,model,c.K,previous,.3);
                TEST_ASSERT(pose.success);TEST_ASSERT((pose.t-t).norm()<.2);
                metrics<<model.size()<<",TT_left_PnP,"<<(pose.t-t).norm()<<','<<(pose.R-r).norm()<<'\n';
                seed_r=pose.R;seed_t=pose.t;
            }
            continue;
        }
        MatchResult matches;matches.good_matches=data.good_matches;matches.pts_left_match=data.pts_left_match;
        InitialPnPSolver initial;const auto cold=initial.solve(matches,model,c.K);
        TEST_ASSERT(cold.success);TEST_ASSERT((cold.t-t).norm()<.2);
        GPnPSolver solver(c);double elapsed=0;
        auto pose=solver.solve(data,model,&seed_r,&seed_t,elapsed,{PoseInitializationSource::PreviousPose});
        TEST_ASSERT(pose.success&&pose.source==PoseSource::StereoGpnp);
        TEST_ASSERT((pose.t-t).norm()<.2);TEST_ASSERT((pose.R-r).norm()<.01);
        metrics<<model.size()<<",StereoGpnp,"<<(pose.t-t).norm()<<','<<(pose.R-r).norm()<<'\n';
        seed_r=pose.R;seed_t=pose.t;
    }
}
void shiftCenterPatch(DualFixture& f) {
    const auto patch=f.left(cv::Rect(300,220,40,40)).clone();
    f.left(cv::Rect(300,220,40,40)).setTo(30);
    patch.copyTo(f.left(cv::Rect(305,217,40,40)));
    const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-60,0,1,0);
    cv::warpAffine(f.left,f.right,transform,f.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
    f.left_roi.secondary.x+=5;f.left_roi.secondary.y-=3;
    f.right_roi.secondary.x+=5;f.right_roi.secondary.y-=3;
    f.config.template_geometry.enabled=true;f.config.template_geometry.class1_center_in_class0_mm={5,-3};
    f.bc.pixel_to_meter_scale_class0=.006;
}
void test_template_geometry_dual_joint_class1_recovery_and_visuals() {
    DualFixture f;shiftCenterPatch(f);auto tracker=f.tracker();
    tracker->setFrameNumber(21);
    auto joint=tracker->process(f.left,f.right,true,&f.left_roi,nullptr);report(joint);
    TEST_ASSERT(joint.success&&joint.dual_c1_pairs>=4);TEST_ASSERT_EQ(joint.model_frame,std::string("class0"));
    auto close=f.left_roi;close.primary={0,0,640,480};tracker->setFrameNumber(22);
    auto fallback=tracker->process(f.left,f.right,true,&close,nullptr);report(fallback);
    TEST_ASSERT(fallback.success&&fallback.is_class1);TEST_ASSERT_EQ(fallback.model_frame,std::string("class0"));
    TEST_ASSERT((modelCenter(fallback.pose_model_points)-Eigen::Vector2d(5,-3)).norm()<1e-9);
    TEST_ASSERT((fallback.t-joint.t).norm()<30);
    tracker->setFrameNumber(23);auto recovered=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);
    TEST_ASSERT(recovered.success);TEST_ASSERT((recovered.t-joint.t).norm()<1);
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(3));
    std::filesystem::create_directories("output/template_geometry_optimization");
    std::ofstream metrics("output/template_geometry_optimization/route_switch.csv");
    metrics<<"strategy,frame,tx_mm,ty_mm,tz_mm,model_frame\n";
    for(const auto* r:{&joint,&fallback,&recovered})
        metrics<<r->strategy_name<<','<<(&joint==r?21:&fallback==r?22:23)<<','
               <<r->t.x()<<','<<r->t.y()<<','<<r->t.z()<<','<<r->model_frame<<'\n';
    utils::AsyncImageSaver::flush();
    for(const auto* name:{"dual_roi_reproj_f21.png","dual_roi_reproj_f22.png","stereo_f22.png"})
        TEST_ASSERT(!cv::imread((f.root/name).string()).empty());
    const auto dest=std::filesystem::path("output/template_geometry_optimization/visuals");
    std::filesystem::create_directories(dest);
    for(const auto& p:std::filesystem::directory_iterator(f.root))if(p.path().filename().string().find("_f2")!=std::string::npos)
        std::filesystem::copy_file(p.path(),dest/p.path().filename(),std::filesystem::copy_options::overwrite_existing);
}
void test_template_geometry_single_class1_mono_and_stereo() {
    DualFixture f;shiftCenterPatch(f);
    cv::Mat bgr;cv::cvtColor(f.left,bgr,cv::COLOR_GRAY2BGR);
    auto c=calibration();
    MonoTracker mono(c.K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    mono.setVerboseConsole(false);
    RoiGroup c1{f.left_roi.secondary,{},false};c1.is_class1=true;
    auto mr=mono.process(bgr,false,&c1);
    TEST_ASSERT(mr.success);TEST_ASSERT_EQ(mr.model_frame,std::string("class0"));
    TEST_ASSERT((modelCenter(mr.pose_model_points)-Eigen::Vector2d(5,-3)).norm()<1e-9);
    auto stereo=f.tracker();RoiGroup right{f.right_roi.secondary,{},false};right.is_class1=true;
    auto sr=stereo->process(f.left,f.right,false,&c1,&right);report(sr);
    TEST_ASSERT(sr.success);TEST_ASSERT_EQ(sr.model_frame,std::string("class0"));
    TEST_ASSERT((modelCenter(sr.pose_model_points)-Eigen::Vector2d(5,-3)).norm()<1e-9);
    // Force the existing class1 TT routes without changing their solver choice.
    f.config.tiny_max_area_class1=100000;
    MonoTracker mono_tt(c.K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    mono_tt.setVerboseConsole(false);
    const auto mt=mono_tt.process(bgr,false,&c1);
    TEST_ASSERT(mt.success&&mt.strategy_name=="TinyTarget");
    TEST_ASSERT_EQ(mt.model_frame,std::string("class0"));
    TEST_ASSERT((modelCenter(mt.pose_model_points)-Eigen::Vector2d(5,-3)).norm()<1e-9);
    const auto st=f.tracker()->process(f.left,f.right,false,&c1,&right);
    TEST_ASSERT(st.success&&st.strategy_name=="TinyTarget");
    TEST_ASSERT_EQ(st.model_frame,std::string("class0"));
    TEST_ASSERT((modelCenter(st.pose_model_points)-Eigen::Vector2d(5,-3)).norm()<1e-9);
}
void test_normalized_mono_dual_secondary_keeps_class1_context() {
    DualFixture f;
    f.bc.mono_identity.enabled=true;f.bc.mono_identity.frontend="normalized";
    f.bc.mono_identity.class1_enabled=false;
    cv::Mat bgr;cv::cvtColor(f.left,bgr,cv::COLOR_GRAY2BGR);auto c=calibration();
    MonoTracker mono(c.K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    mono.setVerboseConsole(false);mono.setOutputDir(f.root.string());mono.setFrameNumber(31);
    auto r=mono.process(bgr,true,&f.left_roi);
    TEST_ASSERT(r.success);TEST_ASSERT(r.strategy_name=="DualRoi_BCJ");
    TEST_ASSERT(r.n_matched>10);utils::AsyncImageSaver::flush();
    TEST_ASSERT(std::filesystem::exists(f.root/"mono_identity_c0_f31.png"));
    // No class0 identity panel must be produced for the inner class1 ROI.
    TEST_ASSERT(!std::filesystem::exists(f.root/"mono_identity_joint_c1_f31.png"));
    auto again=mono.process(bgr,false,&f.left_roi);
    TEST_ASSERT(again.success);TEST_ASSERT((r.t-again.t).norm()<1e-5);
}
void test_template_geometry_disparity_is_independent_of_offset() {
    DualFixture f;f.config.stereo_flow.disparity_only=true;f.config.stereo_flow.disparity_full_chain=true;
    auto baseline=f.tracker()->process(f.left,f.right,false,&f.left_roi,nullptr);
    f.config.template_geometry.enabled=true;f.config.template_geometry.class1_center_in_class0_mm={130,-85};
    auto changed=f.tracker()->process(f.left,f.right,false,&f.left_roi,nullptr);
    TEST_ASSERT(baseline.disparity_success&&changed.disparity_success&&!changed.success);
    TEST_ASSERT_EQ(changed.model_frame,std::string("unresolved"));
    TEST_ASSERT_EQ(baseline.disparity.size(),changed.disparity.size());
    for(size_t i=0;i<changed.disparity.size();++i) {
        TEST_ASSERT(baseline.disparity[i]==changed.disparity[i]);
        TEST_ASSERT(cv::norm(baseline.pts_right_good[i]-changed.pts_right_good[i])==0);
        TEST_ASSERT_EQ(baseline.flow_template_indices[i],changed.flow_template_indices[i]);
    }
    TEST_ASSERT(changed.pnp_ms==0&&changed.pose_model_points.empty());
}
}
void test_mono_roi_verification_rejects_background_and_reuses_valid_candidate() {
    DualFixture f;f.config.roi_selection.enabled=true;f.config.roi_selection.verify_mono=true;
    f.bc.mono_identity.enabled=true;f.bc.mono_identity.frontend="normalized";
    f.bc.mono_identity.quality_filter=true;f.bc.mono_identity.partial_matching=true;
    cv::Mat color;cv::cvtColor(f.left,color,cv::COLOR_GRAY2BGR);auto c=calibration();
    MonoTracker mono(c.K,(f.root/"board.png").string(),f.config,f.bc,f.root.string(),f.tt,f.root.string());
    mono.setVerboseConsole(false);mono.setOutputDir(f.root.string());mono.setFrameNumber(41);
    RoiGroup background{{2,2,28,25},{},false};
    auto wrong=mono.processCandidates(color,false,{{background,0,-1,.99,0}});
    TEST_ASSERT(!wrong.success);TEST_ASSERT(wrong.failure_reason=="roi_appearance_rejected");
    auto result=mono.processCandidates(color,true,{{background,0,-1,.99,0},{f.left_roi,1,2,.8,1}});
    TEST_ASSERT(result.success);TEST_ASSERT(result.roi_selected==1);TEST_ASSERT(result.roi_verified==2);
    TEST_ASSERT(result.bc_detected_count==10);TEST_ASSERT(result.good_matches.size()==result.pts_left_match.size());
    TEST_ASSERT(mono.getLogs().size()==2);
    TEST_ASSERT(std::abs(mono.getLogs().back().extract_ms-result.extract_ms)<1e-9);
    for(auto match:result.good_matches)TEST_ASSERT(match.trainIdx>=0&&match.trainIdx<(int)result.pose_model_points.size());
    utils::AsyncImageSaver::flush();TEST_ASSERT(std::filesystem::exists(f.root/"mono_roi_selection_f41.png"));
}
REGISTER_TEST(test_mono_roi_verification_rejects_background_and_reuses_valid_candidate);
REGISTER_TEST(test_template_geometry_config_validation_and_legacy_default);
REGISTER_TEST(test_template_geometry_extractor_centers_and_repeated_switches);
REGISTER_TEST(test_template_geometry_gpnp_switches_and_seed_use_one_origin);
REGISTER_TEST(test_template_geometry_dual_joint_class1_recovery_and_visuals);
REGISTER_TEST(test_template_geometry_single_class1_mono_and_stereo);
REGISTER_TEST(test_normalized_mono_dual_secondary_keeps_class1_context);
REGISTER_TEST(test_template_geometry_disparity_is_independent_of_offset);
REGISTER_TEST(test_dual_pose_uses_board_model_and_augments_class1);
REGISTER_TEST(test_dual_disparity_only_has_no_pnp_and_exports_region_ids);
REGISTER_TEST(test_dual_occlusion_preserves_original_corner_and_model_ids);
REGISTER_TEST(test_dual_bad_class1_keeps_class0_pose);
REGISTER_TEST(test_dual_primary_short_circuit_uses_class1_frame);
REGISTER_TEST(test_dual_tier1_only_and_missing_annotation);
REGISTER_TEST(test_dual_model_count_mismatch_does_not_pair_prefix);
REGISTER_TEST(test_dual_lazy_legacy_fallback_counts_one_frame_and_log);
REGISTER_TEST(test_dual_disparity_never_calls_legacy_pose_route);
REGISTER_TEST(test_dual_tt_disparity_fallback_has_no_pose_solver);
REGISTER_TEST(test_dual_legacy_tt_requires_explicit_left_pose_permission);
REGISTER_TEST(test_dual_config_and_frontend_selection);
REGISTER_TEST(test_frame_context_reuses_pyramids_and_resets_between_frames);
REGISTER_TEST(test_full_chain_dual_left_tt_precedes_right_detection);
REGISTER_TEST(test_full_chain_dual_switches_respect_fallback_and_tier1);
REGISTER_TEST(test_full_chain_dual_still_supports_lazy_right_tt_attempt);
REGISTER_TEST(test_full_chain_dual_negative_disparity_does_not_retry_tt);
void test_stereo_robust_bc_uses_right_observations() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.bc.pixel_to_meter_scale_class0=.006;
    auto g=f.left_roi;g.is_dual=false;auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&g,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.pose_source==PoseSource::StereoGpnp);
    TEST_ASSERT(r.pts_left_good.size()>=6&&r.good_matches.size()>=6);
    TEST_ASSERT(std::abs(r.t.z()-900)<35);
    auto bad=tracker->process(f.left,cv::Mat(f.right.size(),f.right.type(),cv::Scalar(100)),false,&g,nullptr);
    TEST_ASSERT(!bad.success&&!bad.stereo_success);
}
REGISTER_TEST(test_stereo_robust_bc_uses_right_observations);
void test_stereo_robust_tt_is_stereo_in_pose_mode() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.tiny_max_area=1000000;
    // This fixture's oblique polygon has an ~81.35-source-pixel minimum
    // enclosing square, at 3 mm/source pixel (not its 90px axis-aligned box).
    f.tt.square_size_m_class0=.244;
    auto g=f.left_roi;g.is_dual=false;auto tracker=f.tracker();
    TEST_ASSERT(tracker->prefersLeftStereoFlow(g));
    auto r=tracker->process(f.left,f.right,false,&g,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.pose_source==PoseSource::StereoGpnp);
    TEST_ASSERT(r.strategy_name=="TinyTarget"&&r.pts_right_good.size()==4);
    TEST_ASSERT(r.tt_status=="success");
    TEST_ASSERT(std::abs(r.t.z()-900)<40);
}
REGISTER_TEST(test_stereo_robust_tt_is_stereo_in_pose_mode);
void test_stereo_robust_disparity_skips_every_pose_candidate() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.stereo_flow.disparity_only=true;
    auto tracker=f.tracker();auto r=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);report(r);
    TEST_ASSERT(r.disparity_success&&!r.success&&!r.gpnp_success&&!r.stereo_success);
    TEST_ASSERT(r.pnp_ms==0&&r.pose_model_points.empty()&&r.pose_source==PoseSource::Unspecified);
    TEST_ASSERT(r.flow_pyramid_builds==1);
    for(size_t k=0;k<r.idx_from_filtered.size();++k) {
        auto q=r.idx_from_filtered[k];
        TEST_ASSERT(r.pts_left_good[k]==r.pts_left_match[q]);
        bool found=false;for(auto m:r.good_matches)if(m.queryIdx==q){found=true;TEST_ASSERT(m.trainIdx==r.flow_template_indices[k]);}
        TEST_ASSERT(found);
    }
}
REGISTER_TEST(test_stereo_robust_disparity_skips_every_pose_candidate);
void test_stereo_robust_class1_model_and_state_restore() {
    DualFixture f;f.config.stereo_robustness.enabled=true;
    auto tracker=f.tracker();auto a=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);
    auto g=f.left_roi;g.primary={0,0,640,480};auto b=tracker->process(f.left,f.right,false,&g,nullptr);report(b);
    TEST_ASSERT(b.success&&b.stereo_success&&b.strategy_name=="DualRoi_C1BC");
    double lo=1e9,hi=-1e9;for(auto p:b.pose_model_points){lo=std::min(lo,p.x());hi=std::max(hi,p.x());}
    TEST_ASSERT_NEAR(hi-lo,36,1e-4);
    tracker->clearCache();auto c=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);
    TEST_ASSERT(a.success&&c.success);TEST_ASSERT(a.pts_left_match==c.pts_left_match);TEST_ASSERT((a.t-c.t).norm()<1e-6);
}
REGISTER_TEST(test_stereo_robust_class1_model_and_state_restore);
void test_stereo_robust_config_boundaries() {
    auto read=[](const char* text){cv::FileStorage fs(text,cv::FileStorage::READ|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);return readStereoRobustnessConfig(fs["r"]);};
    TEST_ASSERT(!read("{}").enabled);
    TEST_ASSERT(read("{\"r\":{\"enabled\":true,\"min_bc_points\":6}}").enabled);
    TEST_ASSERT_THROWS(read("{\"r\":{\"enabled\":2}}"),std::invalid_argument);
    TEST_ASSERT_THROWS(read("{\"r\":{\"min_bc_points\":3}}"),std::invalid_argument);
    TEST_ASSERT_THROWS(read("{\"r\":{\"joint_limits\":{\"max_rotation_delta_deg\":181}}}"),std::invalid_argument);
}
REGISTER_TEST(test_stereo_robust_config_boundaries);
void test_stereo_joint_gate_uses_both_cameras_and_same_pairs() {
    const auto camera=calibration();PoseEstimate p;p.success=true;p.t={0,0,900};
    std::vector<Eigen::Vector3d> model{{-100,-100,0},{100,-100,0},{100,100,0},{-100,100,0}};
    PipelineResult r;
    for(int i=0;i<4;++i) {
        auto a=camera.K*(model[i]+p.t);auto b=camera.K*(model[i]+p.t-camera.t_rl);
        r.pts_left_match.emplace_back(a.x()/a.z(),a.y()/a.z());r.pts_left_good.push_back(r.pts_left_match.back());
        r.pts_right_good.emplace_back(b.x()/b.z(),b.y()/b.z());r.idx_from_filtered.push_back(i);r.good_matches.emplace_back(i,i,0);
    }
    MonoDualRoiConfig limits;
    TEST_ASSERT(checkStereoDualJoint(p,p,r,r,model,camera,limits).empty());
    auto changed=p;changed.t.x()+=2;
    TEST_ASSERT(checkStereoDualJoint(p,changed,r,r,model,camera,limits)=="primary_reprojection_worse");
    auto bad_right=r;for(auto& pixel:bad_right.pts_right_good)pixel.x+=10;
    TEST_ASSERT(checkStereoDualJoint(p,p,r,bad_right,model,camera,limits)=="group_reprojection");
    auto nonpositive=p;nonpositive.t.z()=-100;
    TEST_ASSERT(checkStereoDualJoint(p,nonpositive,r,r,model,camera,limits)=="nonfinite_or_depth");
}
REGISTER_TEST(test_stereo_joint_gate_uses_both_cameras_and_same_pairs);
void test_stereo_robust_left_fallback_requires_permission() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.bc.pixel_to_meter_scale_class0=.006;
    auto g=f.left_roi;g.is_dual=false;
    cv::Mat blank(f.right.size(),f.right.type(),cv::Scalar(100));
    auto a=f.tracker();auto rejected=a->process(f.left,blank,false,&g,nullptr);
    TEST_ASSERT(!rejected.success&&!rejected.stereo_success);
    f.config.stereo_flow.allow_left_pnp=true;auto b=f.tracker();auto accepted=b->process(f.left,blank,false,&g,nullptr);
    TEST_ASSERT(accepted.success&&!accepted.stereo_success&&accepted.pose_source==PoseSource::LeftPnpFallback);
}
REGISTER_TEST(test_stereo_robust_left_fallback_requires_permission);
void test_stereo_robust_negative_disparity_does_not_run_tt() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.stereo_flow.disparity_only=true;
    auto g=f.right_roi;g.is_dual=false;auto tracker=f.tracker();
    auto r=tracker->process(f.right,f.left,false,&g,nullptr);
    TEST_ASSERT(!r.disparity_success);
    TEST_ASSERT(r.failure_reason=="negative_disparity_check_input_order");
    TEST_ASSERT(r.route_detail.find("TT:")==std::string::npos);
}
REGISTER_TEST(test_stereo_robust_negative_disparity_does_not_run_tt);
void test_stereo_robust_visualization_is_inert_and_has_both_camera_axes() {
    DualFixture f;f.config.stereo_robustness.enabled=true;auto tracker=f.tracker();tracker->setFrameNumber(91);
    cv::theRNG().state=20261020;auto a=tracker->process(f.left,f.right,false,&f.left_roi,nullptr);
    tracker->clearCache();cv::theRNG().state=20261020;auto b=tracker->process(f.left,f.right,true,&f.left_roi,nullptr);
    TEST_ASSERT(a.success&&b.success);TEST_ASSERT(a.pts_left_good==b.pts_left_good);TEST_ASSERT(a.pts_right_good==b.pts_right_good);
    TEST_ASSERT((a.t-b.t).norm()<1e-7&&(a.R-b.R).norm()<1e-7);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(std::filesystem::exists(f.root/"stereo_axes_lr_f91.png"));
    TEST_ASSERT(std::filesystem::exists(f.root/"stereo_f91.png"));
}
REGISTER_TEST(test_stereo_robust_visualization_is_inert_and_has_both_camera_axes);
void test_stereo_robust_dual_class1_tt_uses_metric_stereo() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.tiny_max_area_class1=3000;
    f.tt.square_size_m_class1=.0325;
    auto g=f.left_roi;g.primary={0,0,640,480};auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&g,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success&&r.is_class1&&r.strategy_name=="DualRoi_C1TT");
    TEST_ASSERT(r.pts_right_good.size()==4&&r.pose_model_points.size()==4);
    TEST_ASSERT(r.route_detail.find("C1BC:skip_tiny")!=std::string::npos);
    TEST_ASSERT(std::abs(r.t.z()-900)<45);
}
REGISTER_TEST(test_stereo_robust_dual_class1_tt_uses_metric_stereo);
void test_stereo_robust_initial_pnp_off_uses_stereo_seed() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.use_initial_pnp=false;
    f.bc.pixel_to_meter_scale_class0=.006;auto g=f.left_roi;g.is_dual=false;auto tracker=f.tracker();
    auto r=tracker->process(f.left,f.right,false,&g,nullptr);report(r);
    TEST_ASSERT(r.success&&r.stereo_success);
    TEST_ASSERT(r.last_gpnp_monitor.initial_source=="StereoTriangulation");
}
REGISTER_TEST(test_stereo_robust_initial_pnp_off_uses_stereo_seed);
void test_stereo_robust_respects_disparity_full_chain_switch() {
    DualFixture f;f.config.stereo_robustness.enabled=true;f.config.stereo_flow.disparity_only=true;
    f.config.stereo_flow.disparity_full_chain=false;f.config.tiny_max_area=1000000;
    auto g=f.left_roi;g.is_dual=false;auto tracker=f.tracker();
    TEST_ASSERT(!tracker->prefersLeftStereoFlow(g));
    auto r=tracker->process(f.left,f.right,false,&g,nullptr);
    TEST_ASSERT(!r.disparity_success&&r.failure_reason=="disparity_only_requires_supported_flow_strategy");
}
REGISTER_TEST(test_stereo_robust_respects_disparity_full_chain_switch);
int main(){g_verbose_console=false;return gpnp_test::TestRegistry::instance().runAll();}
