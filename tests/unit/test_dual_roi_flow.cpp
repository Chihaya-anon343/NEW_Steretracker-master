#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
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
int main(){g_verbose_console=false;return gpnp_test::TestRegistry::instance().runAll();}
