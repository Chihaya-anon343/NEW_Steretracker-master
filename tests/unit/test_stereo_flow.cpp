#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
#include "matching/StereoFlowMatcher.hpp"
#include "tracker/StereoTracker.hpp"
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

using namespace gpnp;
namespace {
StereoCameraParams camera() {
    Eigen::Matrix3d k;
    k << 900,0,320, 0,900,240, 0,0,1;
    return makeStereoCameraParams(k, Eigen::Matrix3d::Identity(), Eigen::Vector3d(60,0,0));
}
struct ImagePair {
    cv::Mat left, right;
    std::vector<cv::Point2f> points, predicted;
};
ImagePair texturedPair(int disparity=80, double dy=0) {
    ImagePair pair;
    pair.left = cv::Mat(480,640,CV_8UC1,cv::Scalar(90));
    pair.points = {{180,110},{310,100},{470,130},{510,300},{350,360},{180,330}};
    cv::RNG rng(76219);
    for (const auto& p : pair.points) {
        cv::Mat patch(49,49,CV_8UC1);
        rng.fill(patch, cv::RNG::UNIFORM, 20, 240);
        cv::GaussianBlur(patch, patch, cv::Size(3,3), .6);
        patch.copyTo(pair.left(cv::Rect(static_cast<int>(p.x)-24,static_cast<int>(p.y)-24,49,49)));
        pair.predicted.emplace_back(p.x-disparity+.2f,p.y+dy-.15f);
    }
    const cv::Mat transform = (cv::Mat_<double>(2,3)<<1,0,-disparity,0,1,dy);
    cv::warpAffine(pair.left, pair.right, transform,
                   pair.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(90));
    return pair;
}
void checkShift(const FlowMatchResult& result, int disparity, double tolerance=.5) {
    TEST_ASSERT(result.success);
    for (size_t i=0;i<result.left_points.size();++i) {
        TEST_ASSERT(std::abs(result.left_points[i].x-result.right_points[i].x-disparity)<tolerance);
        TEST_ASSERT(std::abs(result.left_points[i].y-result.right_points[i].y)<tolerance);
    }
}
void test_seeded_large_disparity() {
    auto pair=texturedPair(140);
    const auto result=StereoFlowMatcher(camera(),StereoFlowConfig{}).match(pair.left,pair.right,pair.points,pair.predicted);
    checkShift(result,140);
    TEST_ASSERT_EQ(result.left_indices.size(),pair.points.size());
    TEST_ASSERT_EQ(result.coarse_attempts,0);
}
void test_cold_epipolar_search() {
    auto pair=texturedPair(80);
    const auto result=StereoFlowMatcher(camera(),StereoFlowConfig{}).match(pair.left,pair.right,pair.points);
    checkShift(result,80);
    TEST_ASSERT_EQ(result.coarse_attempts,static_cast<int>(pair.points.size()));
}
void test_cold_epipolar_band_recovers_subpixel_vertical_offsets() {
    for (double dy : {-2.25, 1.774, 2.25}) {
        const auto pair=texturedPair(140,dy);
        StereoFlowConfig config;config.max_epipolar_error_px=3;
        const auto result=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points);
        TEST_ASSERT(result.success);
        TEST_ASSERT_EQ(result.left_indices.size(),pair.points.size());
        for (size_t i=0;i<result.right_points.size();++i) {
            TEST_ASSERT(std::abs(result.left_points[i].x-result.right_points[i].x-140)<.15);
            TEST_ASSERT(std::abs(result.right_points[i].y-result.left_points[i].y-dy)<.15);
        }
        for (const auto& d:result.diagnostics) {
            TEST_ASSERT(d.coarse_attempted && d.accepted);
            TEST_ASSERT(d.coarse_candidates>0 && d.coarse_candidates<=9*config.max_search_candidates);
        }
    }
}
void test_epipolar_band_keeps_quality_and_uniqueness_gates() {
    auto pair=texturedPair(80,6);
    StereoFlowConfig config;config.max_epipolar_error_px=3;
    auto outside=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points);
    TEST_ASSERT(!outside.success && outside.left_points.empty());
    pair=texturedPair(80,2.25);
    pair.right(cv::Rect(76,88,49,49)).copyTo(pair.right(cv::Rect(16,88,49,49)));
    const auto ambiguous=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points);
    TEST_ASSERT(ambiguous.success);
    TEST_ASSERT(!ambiguous.diagnostics[0].accepted);
    TEST_ASSERT_EQ(ambiguous.diagnostics[0].reason,std::string("coarse_nonunique_peak"));
    TEST_ASSERT(ambiguous.left_indices==std::vector<int>({1,2,3,4,5}));
}
void test_epipolar_band_works_with_oblique_epipolar_lines() {
    auto pair=texturedPair(60);
    auto c=camera();c.R_rl=Eigen::AngleAxisd(.04,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d h=(c.K*c.R_rl.transpose()*
        (Eigen::Matrix3d::Identity()-c.t_rl*Eigen::RowVector3d(0,0,1)/900)*c.K_inv).eval();
    cv::Mat transform(3,3,CV_64F);
    for(int r=0;r<3;++r) for(int s=0;s<3;++s) transform.at<double>(r,s)=h(r,s);
    cv::warpPerspective(pair.left,pair.right,transform,pair.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,90);
    StereoFlowConfig config;config.max_epipolar_error_px=3;
    const auto result=StereoFlowMatcher(c,config).match(pair.left,pair.right,pair.points);
    if (!result.success) for (size_t i=0;i<result.diagnostics.size();++i) {
        const auto& d=result.diagnostics[i];
        std::cerr << "Oblique point " << i << " reason=" << d.reason
                  << " coarse=" << d.coarse_best_ncc << "/" << d.coarse_second_ncc
                  << " seed=" << d.initial_right << " fb=" << d.fb_error
                  << " epi=" << d.epipolar_error << " ncc=" << d.ncc << '\n';
    }
    TEST_ASSERT(result.success);
    for(size_t i=0;i<result.right_points.size();++i) {
        auto p=pair.points[result.left_indices[i]];
        const Eigen::Vector3d q=h*Eigen::Vector3d(p.x,p.y,1);
        TEST_ASSERT(cv::norm(result.right_points[i]-cv::Point2f(q.x()/q.z(),q.y()/q.z()))<.3);
    }
}
void test_partial_occlusion_preserves_ids() {
    auto pair=texturedPair();
    pair.right(cv::Rect(75,85,50,50)).setTo(90);
    auto config=StereoFlowConfig{}; config.coarse_retry=false;
    const auto result=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(result.success);
    TEST_ASSERT(!result.diagnostics[0].accepted);
    TEST_ASSERT(result.left_indices==std::vector<int>({1,2,3,4,5}));
    checkShift(result,80);
}
void test_invalid_points_are_not_sent_to_lk() {
    auto pair=texturedPair();
    pair.points.insert(pair.points.begin(),cv::Point2f(std::numeric_limits<float>::quiet_NaN(),120));
    pair.predicted.insert(pair.predicted.begin(),cv::Point2f(100,120));
    auto result=StereoFlowMatcher(camera(),StereoFlowConfig{}).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(result.success);
    TEST_ASSERT(result.left_indices==std::vector<int>({1,2,3,4,5,6}));
    TEST_ASSERT_EQ(result.diagnostics[0].reason,std::string("left_point_or_border"));
}
void test_epipolar_violation_is_rejected() {
    auto pair=texturedPair(80,6);
    auto config=StereoFlowConfig{}; config.coarse_retry=false;
    auto result=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(!result.success);
    TEST_ASSERT(result.left_points.empty());
}
void test_swapped_pair_negative_disparity_is_rejected() {
    auto pair=texturedPair(-40);
    auto config=StereoFlowConfig{}; config.coarse_retry=false;
    auto result=StereoFlowMatcher(camera(),config).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(!result.success && result.left_points.empty());
    TEST_ASSERT_EQ(result.failure_reason,std::string("negative_disparity_check_input_order"));
    for (const auto& d:result.diagnostics)
        TEST_ASSERT_EQ(d.reason,std::string("disparity_range"));
}
void test_blank_right_and_invalid_input() {
    auto pair=texturedPair();
    pair.right.setTo(90);
    auto matcher=StereoFlowMatcher(camera(),StereoFlowConfig{});
    TEST_ASSERT(!matcher.match(pair.left,pair.right,pair.points,pair.predicted).success);
    auto bad=matcher.match(pair.left,cv::Mat(),pair.points,pair.predicted);
    TEST_ASSERT(!bad.success);
    TEST_ASSERT_EQ(bad.failure_reason,std::string("invalid_stereo_images"));
    TEST_ASSERT(!matcher.match(pair.left,pair.right(cv::Rect(0,0,100,100)),pair.points).success);
}
void test_config_defaults_and_validation() {
    cv::FileStorage old("{\"camera\":{}}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(!readStereoFlowConfig(old["stereo_frontend"]).enabled);
    cv::FileStorage valid("{\"method\":\"left_bc_lk\",\"on_failure\":\"left_pnp\",\"window_size\":31}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    auto c=readStereoFlowConfig(valid.root());
    TEST_ASSERT(c.enabled && c.allow_left_pnp && c.window_size==31);
    TEST_ASSERT(c.coarse_epipolar_band);
    cv::FileStorage line_only("{\"method\":\"left_bc_lk\",\"coarse_epipolar_band\":false}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(!readStereoFlowConfig(line_only.root()).coarse_epipolar_band);
    cv::FileStorage invalid("{\"method\":\"typo\"}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT_THROWS(readStereoFlowConfig(invalid.root()),std::invalid_argument);
    c.min_ncc=std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT_THROWS(validateStereoFlowConfig(c),std::invalid_argument);
}
ImagePair compactRoiPair(int disparity=45,int dy=0,int x=700) {
    ImagePair pair;
    pair.left=cv::Mat(900,1200,CV_8UC1,cv::Scalar(90));
    pair.points={{float(x),260},{float(x+100),270},{float(x+140),330},
                 {float(x+100),390},{float(x+20),380},{float(x-20),320}};
    cv::RNG rng(5086);
    for (auto p:pair.points) {
        cv::Mat patch(49,49,CV_8UC1);rng.fill(patch,cv::RNG::UNIFORM,20,240);
        cv::GaussianBlur(patch,patch,{3,3},.6);
        patch.copyTo(pair.left(cv::Rect(int(p.x)-24,int(p.y)-24,49,49)));
        pair.predicted.emplace_back(p.x-disparity+.2f,p.y+dy-.15f);
    }
    const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-disparity,0,1,dy);
    cv::warpAffine(pair.left,pair.right,transform,pair.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(90));
    return pair;
}
void compareRoiToFull(ImagePair pair,bool cold=false) {
    StereoFlowConfig full;full.max_epipolar_error_px=3;
    auto roi=full;roi.roi_pyramid=true;
    const auto seeds=cold?std::vector<cv::Point2f>{}:pair.predicted;
    const auto a=StereoFlowMatcher(camera(),full).match(pair.left,pair.right,pair.points,seeds);
    const auto b=StereoFlowMatcher(camera(),roi).match(pair.left,pair.right,pair.points,seeds);
    TEST_ASSERT(a.success && b.success && b.roi_used && !b.full_frame_fallback);
    TEST_ASSERT(b.search_roi.area()<pair.left.rows*pair.left.cols*.85);
    TEST_ASSERT(a.left_indices==b.left_indices);
    for(size_t i=0;i<a.right_points.size();++i) {
        TEST_ASSERT(cv::norm(a.right_points[i]-b.right_points[i])<.1);
        TEST_ASSERT(cv::norm(b.left_points[i]-pair.points[b.left_indices[i]])<1e-5);
        const auto truth=pair.predicted[b.left_indices[i]]+cv::Point2f(-.2f,.15f);
        TEST_ASSERT(cv::norm(b.right_points[i]-truth)<.5);
    }
}
void test_roi_equivalence_and_coordinate_restore() { compareRoiToFull(compactRoiPair()); }
void test_roi_large_disparity_and_cold_search() {
    compareRoiToFull(compactRoiPair(300));compareRoiToFull(compactRoiPair(120),true);
    compareRoiToFull(compactRoiPair(120,2),true);
}
void test_roi_vertical_offset_and_image_edges() {
    compareRoiToFull(compactRoiPair(45,2));
    compareRoiToFull(compactRoiPair(80,0,150));
    compareRoiToFull(compactRoiPair(45,0,1000));
}
void test_roi_occlusion_and_full_frame_retry() {
    auto pair=compactRoiPair();
    pair.right(cv::Rect(655-24,260-24,49,49)).setTo(90);
    compareRoiToFull(pair);
    pair.right.setTo(90);
    StereoFlowConfig cfg;cfg.roi_pyramid=true;
    const auto rejected=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(!rejected.success && rejected.roi_used && rejected.full_frame_fallback);
    TEST_ASSERT(!rejected.roi_failure_reason.empty());
    TEST_ASSERT(rejected.total_ms>=rejected.roi_attempt_ms);
    cfg.roi_fallback_full=false;
    const auto no_retry=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(!no_retry.success && no_retry.roi_used && !no_retry.full_frame_fallback);
}
void test_roi_bypasses_large_region_and_general_extrinsics() {
    auto pair=compactRoiPair();StereoFlowConfig cfg;cfg.roi_pyramid=true;
    auto full=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted,{0,0,1200,900});
    TEST_ASSERT(full.success && !full.roi_used);
    auto c=camera();c.R_rl=Eigen::AngleAxisd(.01,Eigen::Vector3d::UnitY()).toRotationMatrix();
    auto general=StereoFlowMatcher(c,cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(!general.roi_used);
}
void test_pose_validation_uses_original_indices() {
    const auto c=camera();
    std::vector<Eigen::Vector3d> model{{-70,-40,0},{-10,-60,0},{70,-30,0},{65,50,0},{-20,60,0},{-65,20,0}};
    PoseEstimate p; p.success=true; p.t=Eigen::Vector3d(0,0,900);
    PipelineResult result;
    for (size_t i=0;i<model.size();++i) {
        const auto u=(c.K*(model[i]+p.t)).eval();
        result.pts_left_match.emplace_back(u.x()/u.z(),u.y()/u.z());
        result.good_matches.emplace_back(static_cast<int>(i),static_cast<int>(i),0);
    }
    result.idx_from_filtered={1,3,4,5};
    for (int i:result.idx_from_filtered) {
        result.pts_left_good.push_back(result.pts_left_match[i]);
        const auto u=(c.K*(model[i]+p.t-c.t_rl)).eval();
        result.pts_right_good.emplace_back(u.x()/u.z(),u.y()/u.z());
    }
    TEST_ASSERT(checkFlowPose(p,result,model,c,true,1,2).success);
    result.idx_from_filtered={0,1,2,3};
    TEST_ASSERT(!checkFlowPose(p,result,model,c,true,1,2).success);
    p.t.z()=-900;
    TEST_ASSERT(!checkFlowPose(p,result,model,c,true,1,2).success);
}

struct Fixture {
    std::filesystem::path root;
    cv::Mat left,right;
    RoiGroup roi{{270,130,120,120},{},false};
    Fixture() {
        root=std::filesystem::path("tests/output")/ ("bc_flow_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        std::vector<cv::Point> polygon{{40,5},{60,5},{60,35},{95,35},{95,65},{60,65},{60,95},{40,95},{40,65},{5,50}};
        std::vector<cv::Point2f> points;
        cv::Point2f mean(0,0);
        for (auto p:polygon) { points.emplace_back(p); mean+=cv::Point2f(p); }
        mean*=1.0f/points.size();
        // External white contours use the reverse direction of this polygon.
        std::reverse(points.begin(),points.end());
        auto order=BinaryCornerExtractor::reorderByGeometry(points,mean,0,-1);
        {
            std::ofstream txt(root/"0_degrees.txt");
            for (size_t i=0;i<order.size();++i)
                txt<<"Corner_"<<i<<": "<<points[order[i]].x/2<<", "<<points[order[i]].y/2<<"\n";
        }
        cv::Mat mask(100,100,CV_8UC1,cv::Scalar(0));
        cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{polygon},cv::Scalar(255));
        cv::Mat template_mask;
        cv::resize(mask, template_mask, cv::Size(50,50), 0, 0, cv::INTER_NEAREST);
        cv::imwrite((root/"0_degrees.png").string(),template_mask);
        left=cv::Mat(480,640,CV_8UC1,cv::Scalar(0));
        mask.copyTo(left(cv::Rect(280,140,100,100)));
        const cv::Mat transform = (cv::Mat_<double>(2,3)<<1,0,-60,0,1,0);
        cv::warpAffine(left,right,transform,left.size());
    }
    ~Fixture() { utils::AsyncImageSaver::flush(); std::filesystem::remove_all(root); }
    std::unique_ptr<StereoTracker> tracker(bool fallback=false, bool disparity_only=false, bool roi_pyramid=false) {
        auto c=makeTrackerConfig(4,true); c.temporal.enabled=false; c.stereo_flow.enabled=true;
        c.stereo_flow.allow_left_pnp=fallback;
        c.stereo_flow.disparity_only=disparity_only;
        c.stereo_flow.roi_pyramid=roi_pyramid;
        BinaryCornerExtractor::Config b; b.corners=10; b.scale=1; b.pixel_to_meter_scale_class0=.002;
        b.roi_pad_pixels=0;
        TinyTargetExtractor::Config t; t.square_size_m_class0=.095;
        auto cam=camera();
        auto tracker=std::make_unique<StereoTracker>(cam.K,cam.R_rl,cam.t_rl,(root/"0_degrees.png").string(),
                                                    c,b,root.string(),t,root.string());
        tracker->setOutputDir(root.string()); tracker->setVerboseConsole(false);
        return tracker;
    }
};
void test_tracker_flow_and_visualization() {
    Fixture f; auto tracker=f.tracker();
    TEST_ASSERT(tracker->prefersLeftBcFlow(f.roi));
    tracker->setFrameNumber(7);
    auto r=tracker->process(f.left,f.right,true,&f.roi,nullptr);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(r.success && r.stereo_success);
    TEST_ASSERT(r.pose_source==PoseSource::StereoGpnp);
    TEST_ASSERT_EQ(r.stereo_method,std::string("LK"));
    TEST_ASSERT_EQ(tracker->frameCount(),1);
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    TEST_ASSERT(tracker->getLogs()[0].stereo_success);
    TEST_ASSERT(r.stereo_match_ms>0 && r.extract_ms>=r.stereo_match_ms && r.pnp_ms>0);
    for (const auto& name:{"stereo_f7.png","binary_corner_stereo_f7.png","binary_corner_binary_f7.png",
                           "binary_corner_axes_f7.png","binary_corner_template_f7.png","binary_corner_reproj_f7.png", "binary_corner_flow_status_f7.png"})
        TEST_ASSERT(!cv::imread((f.root/name).string()).empty());
    for (size_t i=0;i<r.idx_from_filtered.size();++i)
        TEST_ASSERT(cv::norm(r.pts_left_good[i]-r.pts_left_match[r.idx_from_filtered[i]])<1e-5);
    const auto preview=std::filesystem::path("output/bc_flow_validation/synthetic");
    std::filesystem::create_directories(preview);
    for (const auto& entry:std::filesystem::directory_iterator(f.root))
        if (entry.path().extension()==".png" && entry.path().filename()!="0_degrees.png")
            std::filesystem::copy_file(entry.path(),preview/entry.path().filename(),std::filesystem::copy_options::overwrite_existing);
}
void test_tracker_rejects_bad_matches_without_false_success() {
    Fixture f; auto tracker=f.tracker();
    f.right.setTo(0);
    auto r=tracker->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT(!r.success && !r.stereo_success && !r.gpnp_success);
    TEST_ASSERT(!r.failure_reason.empty());
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    TEST_ASSERT_EQ(tracker->frameCount(),1);
}
void test_tracker_explicit_left_fallback_and_missing_right_visualization() {
    Fixture f; auto tracker=f.tracker(true);
    tracker->setFrameNumber(9); tracker->setVisualizeDetailed(false);
    auto r=tracker->process(f.left,cv::Mat(),true,&f.roi,nullptr);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(r.success && !r.stereo_success);
    TEST_ASSERT(r.pose_source==PoseSource::LeftPnpFallback);
    TEST_ASSERT(r.fallback_used);
    TEST_ASSERT_EQ(tracker->getLogs().size(),size_t(1));
    TEST_ASSERT(!cv::imread((f.root/"stereo_f9.png").string()).empty());
}
void test_legacy_process_interface_remains_available() {
    Fixture f; auto tracker=f.tracker();
    RoiGroup right=f.roi; right.primary.x-=60;
    auto r=tracker->process(f.left,f.right,false,&f.roi,&right);
    TEST_ASSERT(r.success);
    TEST_ASSERT_EQ(r.stereo_method,std::string("dual_bc"));
    TEST_ASSERT_EQ(tracker->frameCount(),1);
}
void test_disparity_only_skips_all_pose_solvers() {
    Fixture f; auto tracker=f.tracker(true,true);
    tracker->setFrameNumber(11);
    auto r=tracker->process(f.left,f.right,true,&f.roi,nullptr);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(r.disparity_only && r.disparity_success);
    TEST_ASSERT(!r.success && !r.gpnp_success && !r.stereo_success);
    TEST_ASSERT(r.pose_source==PoseSource::Unspecified && r.pnp_ms==0);
    TEST_ASSERT(r.timing["initial_pnp"]==0 && r.timing["gpnp"]==0);
    TEST_ASSERT(r.failure_reason.empty());
    TEST_ASSERT(tracker->getLogs()[0].disparity_success);
    TEST_ASSERT(!std::filesystem::exists(f.root/"binary_corner_axes_f11.png"));
    TEST_ASSERT(std::filesystem::exists(f.root/"disparity_f11.csv"));
    for (double d:r.disparity) TEST_ASSERT(std::abs(d-60)<.5);
    f.right.setTo(0);
    auto rejected=tracker->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT(rejected.disparity_only && !rejected.disparity_success && !rejected.success);
    TEST_ASSERT(rejected.pnp_ms==0 && !rejected.fallback_used);
    RoiGroup dual=f.roi; dual.is_dual=true;
    auto skipped=tracker->process(f.left,f.right,false,&dual,&dual);
    TEST_ASSERT(skipped.disparity_only && !skipped.success && skipped.pnp_ms==0);
}
void test_roi_tracker_pose_and_disparity_exports() {
    Fixture f;auto pose_tracker=f.tracker(false,false,true);
    auto pose=pose_tracker->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT(pose.success && pose.stereo_success && pose.flow_roi.area()>0 && !pose.flow_full_fallback);
    auto tracker=f.tracker(false,true,true);tracker->setFrameNumber(13);
    auto disparity=tracker->process(f.left,f.right,true,&f.roi,nullptr);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(disparity.disparity_success && disparity.flow_roi.area()>0 && disparity.pnp_ms==0);
    TEST_ASSERT(tracker->getLogs()[0].flow_roi==disparity.flow_roi);
    TEST_ASSERT(std::filesystem::exists(f.root/"disparity_f13.csv"));
    TEST_ASSERT(!cv::imread((f.root/"stereo_f13.png").string()).empty());
}
} // namespace
REGISTER_TEST(test_seeded_large_disparity);
namespace {
std::vector<cv::Point2f> planarPoints() {
    return {{100,80},{200,80},{300,100},{350,170},{300,260},{200,290},{100,240},{70,150},{170,160},{260,190}};
}
FlowMatchResult planarObservations(const std::vector<cv::Point2f>& points) {
    FlowMatchResult out;out.diagnostics.resize(points.size());
    for(size_t i=0;i<points.size();++i) {
        auto& d=out.diagnostics[i];d.accepted=true;d.reason="accepted";d.fb_error=.01;d.ncc=.96;
        d.tracked_right={points[i].x-(70+.012f*points[i].x-.017f*points[i].y),points[i].y+1.774f};
    }
    return out;
}
void test_planar_filter_removes_residual_outlier_preserving_tilt() {
    auto points=planarPoints();auto a=planarObservations(points),b=a;
    a.diagnostics[9].tracked_right.x-=.6371f;b=a;
    StereoFlowConfig cfg;cfg.planar_geometry_filter=true;
    filterPlanarGeometry(points,cfg,true,a);filterPlanarGeometry(points,cfg,true,b);
    TEST_ASSERT_EQ(a.planar_rejected,1);TEST_ASSERT(!a.diagnostics[9].accepted);
    TEST_ASSERT_EQ(a.planar_status,std::string("disparity_plane"));
    for(int i=0;i<9;++i) {
        TEST_ASSERT(a.diagnostics[i].accepted);TEST_ASSERT(a.diagnostics[i].planar_error_px<.001);
        TEST_ASSERT(a.diagnostics[i].tracked_right==b.diagnostics[i].tracked_right);
    }
}
void test_planar_filter_skips_tt_and_disabled_config() {
    auto points=planarPoints();points.resize(4);auto out=planarObservations(points);
    out.diagnostics[3].tracked_right.x-=5;
    StereoFlowConfig cfg;filterPlanarGeometry(points,cfg,true,out);
    TEST_ASSERT_EQ(out.planar_status,std::string("disabled"));
    cfg.planar_geometry_filter=true;filterPlanarGeometry(points,cfg,true,out);
    TEST_ASSERT_EQ(out.planar_status,std::string("insufficient_redundancy"));
    TEST_ASSERT(out.diagnostics[3].accepted&&out.planar_rejected==0);
}
void test_planar_filter_reports_no_consensus_without_arbitrary_rejection() {
    auto points=planarPoints();auto out=planarObservations(points);
    const float bad[]={0,7,-9,22,-14,16,-3,31,-25,11};
    for(int i=0;i<10;++i)out.diagnostics[i].tracked_right.x+=bad[i];
    StereoFlowConfig cfg;cfg.planar_geometry_filter=true;
    filterPlanarGeometry(points,cfg,true,out);
    TEST_ASSERT_EQ(out.planar_status,std::string("uncertain_no_consensus"));
    for(const auto& d:out.diagnostics)TEST_ASSERT(d.accepted);
}
void test_planar_homography_rejects_outlier() {
    auto points=planarPoints();auto out=planarObservations(points);
    for(int i=0;i<10;++i) {
        const auto p=points[i];const float z=1+.00003f*p.x-.00004f*p.y;
        out.diagnostics[i].tracked_right={(.99f*p.x-.01f*p.y-35)/z,(.005f*p.x+1.01f*p.y+1.4f)/z};
    }
    out.diagnostics[9].tracked_right.y+=3;
    StereoFlowConfig cfg;cfg.planar_geometry_filter=true;
    filterPlanarGeometry(points,cfg,false,out);
    TEST_ASSERT_EQ(out.planar_status,std::string("plane_homography"));
    TEST_ASSERT_EQ(out.planar_rejected,1);TEST_ASSERT(!out.diagnostics[9].accepted);
}
void test_planar_filter_actual_lk_and_original_indices() {
    auto pair=texturedPair(60);const double dy=1.774;
    const cv::Mat warp=(cv::Mat_<double>(2,3)<<.99,-.02,-50,0,1,dy);
    cv::warpAffine(pair.left,pair.right,warp,pair.left.size(),cv::INTER_LINEAR,cv::BORDER_REPLICATE);
    pair.predicted.clear();
    for(auto p:pair.points)pair.predicted.emplace_back(.99f*p.x-.02f*p.y-50,p.y+dy);
    StereoFlowConfig cfg;cfg.max_epipolar_error_px=3;cfg.planar_geometry_filter=true;
    auto out=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(out.success&&out.planar_status=="disparity_plane");
    for(size_t k=0;k<out.left_indices.size();++k)TEST_ASSERT(out.left_points[k]==pair.points[out.left_indices[k]]);
    cfg.planar_max_error_px=0;TEST_ASSERT_THROWS(validateStereoFlowConfig(cfg),std::invalid_argument);
}
void test_planar_filter_lk_rejects_middle_id_without_renumbering() {
    auto pair=texturedPair(60);
    const cv::Point2f middle(350,240);
    pair.points.insert(pair.points.begin()+2,middle);
    cv::Mat patch(49,49,CV_8UC1);cv::RNG rng(48571);
    rng.fill(patch,cv::RNG::UNIFORM,20,240);cv::GaussianBlur(patch,patch,{3,3},.6);
    patch.copyTo(pair.left(cv::Rect(326,216,49,49)));
    const cv::Mat global=(cv::Mat_<double>(2,3)<<1,0,-60,0,1,0);
    cv::warpAffine(pair.left,pair.right,global,pair.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(90));
    const cv::Rect region(266,216,49,49);
    cv::Mat original=pair.right(region).clone(),corrupted;
    const cv::Mat local=(cv::Mat_<double>(2,3)<<1,0,.8,0,1,0);
    cv::warpAffine(original,corrupted,local,original.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(90));
    corrupted.copyTo(pair.right(region));
    pair.predicted.clear();for(auto p:pair.points)pair.predicted.push_back(p-cv::Point2f(60,0));
    StereoFlowConfig cfg;
    const auto before=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(before.success&&before.left_indices.size()==7);
    cfg.planar_geometry_filter=true;
    const auto after=StereoFlowMatcher(camera(),cfg).match(pair.left,pair.right,pair.points,pair.predicted);
    TEST_ASSERT(after.success&&after.planar_rejected==1);
    TEST_ASSERT(after.left_indices==std::vector<int>({0,1,3,4,5,6}));
    TEST_ASSERT_EQ(after.diagnostics[2].reason,std::string("planar_geometry_outlier"));
    for(size_t k=0;k<after.left_indices.size();++k)TEST_ASSERT(after.left_points[k]==pair.points[after.left_indices[k]]);
}
}
REGISTER_TEST(test_planar_filter_removes_residual_outlier_preserving_tilt);
REGISTER_TEST(test_planar_filter_skips_tt_and_disabled_config);
REGISTER_TEST(test_planar_filter_reports_no_consensus_without_arbitrary_rejection);
REGISTER_TEST(test_planar_homography_rejects_outlier);
REGISTER_TEST(test_planar_filter_actual_lk_and_original_indices);
REGISTER_TEST(test_planar_filter_lk_rejects_middle_id_without_renumbering);
REGISTER_TEST(test_cold_epipolar_search);
REGISTER_TEST(test_cold_epipolar_band_recovers_subpixel_vertical_offsets);
REGISTER_TEST(test_epipolar_band_keeps_quality_and_uniqueness_gates);
REGISTER_TEST(test_epipolar_band_works_with_oblique_epipolar_lines);
REGISTER_TEST(test_partial_occlusion_preserves_ids);
REGISTER_TEST(test_invalid_points_are_not_sent_to_lk);
REGISTER_TEST(test_epipolar_violation_is_rejected);
REGISTER_TEST(test_swapped_pair_negative_disparity_is_rejected);
REGISTER_TEST(test_blank_right_and_invalid_input);
REGISTER_TEST(test_config_defaults_and_validation);
REGISTER_TEST(test_pose_validation_uses_original_indices);
REGISTER_TEST(test_tracker_flow_and_visualization);
REGISTER_TEST(test_tracker_rejects_bad_matches_without_false_success);
REGISTER_TEST(test_tracker_explicit_left_fallback_and_missing_right_visualization);
REGISTER_TEST(test_legacy_process_interface_remains_available);
REGISTER_TEST(test_disparity_only_skips_all_pose_solvers);
REGISTER_TEST(test_roi_equivalence_and_coordinate_restore);
REGISTER_TEST(test_roi_large_disparity_and_cold_search);
REGISTER_TEST(test_roi_vertical_offset_and_image_edges);
REGISTER_TEST(test_roi_occlusion_and_full_frame_retry);
REGISTER_TEST(test_roi_bypasses_large_region_and_general_extrinsics);
REGISTER_TEST(test_roi_tracker_pose_and_disparity_exports);
int main() { return gpnp_test::TestRegistry::instance().runAll(); }
