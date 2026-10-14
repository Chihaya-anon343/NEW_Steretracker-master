#include "../framework/TestAssert.hpp"
#include "common/Config.hpp"
#include "tracker/StereoTracker.hpp"
#include "utils/AsyncImageSaver.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <iomanip>

using namespace gpnp;
namespace {
struct Fixture {
    std::filesystem::path root;
    cv::Mat left,right;
    RoiGroup roi;
    TrackerConfig cfg=makeTrackerConfig(4,true,400,400,900,400,0);
    BinaryCornerExtractor::Config bc;
    TinyTargetExtractor::Config tt;
    Fixture(int size=16,double d=35.25,double dy=1.774) {
        root=std::filesystem::path("tests/output")/("disparity_chain_"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        cv::Mat mask(50,50,CV_8UC1,cv::Scalar(0));mask(cv::Rect(5,5,40,40)).setTo(255);
        cv::imwrite((root/"0_degrees.png").string(),mask);
        std::ofstream points(root/"0_degrees.txt");
        points<<"Corner_0: 5, 5\nCorner_1: 45, 5\nCorner_2: 45, 45\nCorner_3: 5, 45\n";points.close();
        left=cv::Mat(240,320,CV_8UC1);cv::RNG rng(88321);
        for(int y=0;y<left.rows;++y)for(int x=0;x<left.cols;++x)
            left.at<uchar>(y,x)=static_cast<uchar>(rng.uniform(0,25));
        for(int y=91;y<91+size;++y)for(int x=166;x<166+size;++x)
            left.at<uchar>(y,x)=static_cast<uchar>(rng.uniform(175,250));
        cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-d,0,1,dy);
        cv::warpAffine(left,right,transform,left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
        roi={{160,85,size+12,size+12},{},false};
        cfg.temporal.enabled=false;cfg.stereo_flow.enabled=true;cfg.stereo_flow.disparity_only=true;
        cfg.stereo_flow.disparity_full_chain=true;cfg.stereo_flow.roi_pyramid=true;
        cfg.stereo_flow.max_epipolar_error_px=3;cfg.stereo_flow.max_disparity_px=100;
        bc.corners=4;bc.scale=1;bc.roi_pad_pixels=3;
        tt.roi_pad_pixels=5;
    }
    ~Fixture(){utils::AsyncImageSaver::flush();if(!std::getenv("ST_KEEP_CHAIN_OUTPUT"))std::filesystem::remove_all(root);}
    std::unique_ptr<StereoTracker> tracker() {
        Eigen::Matrix3d k;k<<500,0,160,0,500,120,0,0,1;
        auto t=std::make_unique<StereoTracker>(k,Eigen::Matrix3d::Identity(),Eigen::Vector3d(60,0,0),
            (root/"0_degrees.png").string(),cfg,bc,root.string(),tt,root.string());
        t->setOutputDir(root.string());t->setVerboseConsole(false);return t;
    }
};
void assertDisparity(const PipelineResult& r,double expected,size_t n=4) {
    TEST_ASSERT(r.disparity_only&&r.disparity_success);
    TEST_ASSERT(!r.success&&!r.gpnp_success&&!r.stereo_success&&r.pnp_ms==0);
    TEST_ASSERT(r.pose_source==PoseSource::Unspecified&&r.pose_model_points.empty());
    TEST_ASSERT_EQ(r.disparity.size(),n);
    TEST_ASSERT_EQ(r.flow_regions.size(),n);TEST_ASSERT_EQ(r.flow_template_indices.size(),n);
    for(size_t i=0;i<n;++i) {
        TEST_ASSERT_NEAR(r.disparity[i],expected,.2);
        TEST_ASSERT_NEAR(r.disparity[i],r.pts_left_good[i].x-r.pts_right_good[i].x,1e-6);
        TEST_ASSERT(r.idx_from_filtered[i]>=0&&r.idx_from_filtered[i]<r.n_kp_left);
    }
}
void test_direct_tiny_flow_and_visual_csv() {
    Fixture f;auto t=f.tracker();TEST_ASSERT(t->prefersLeftStereoFlow(f.roi));
    TEST_ASSERT(!t->prefersLeftBcFlow(f.roi));t->setFrameNumber(21);
    auto r=t->process(f.left,f.right,true,&f.roi,nullptr);assertDisparity(r,35.25);
    TEST_ASSERT_EQ(r.strategy_name,std::string("TinyTarget"));TEST_ASSERT(!r.fallback_used);
    TEST_ASSERT_EQ(t->getLogs().size(),size_t(1));TEST_ASSERT(r.flow_pyramid_builds>=1);
    utils::AsyncImageSaver::flush();
    TEST_ASSERT(!cv::imread((f.root/"stereo_f21.png").string()).empty());
    TEST_ASSERT(std::filesystem::exists(f.root/"tiny_target_flow_status_f21.png"));
    std::ifstream csv(f.root/"disparity_f21.csv");std::string header,line;std::getline(csv,header);
    TEST_ASSERT(header.find("left_index,left_x,left_y,right_x,right_y,disparity_px,")==0);
    TEST_ASSERT(header.find("strategy,frame_match_ok")!=std::string::npos);
    int rows=0;while(std::getline(csv,line)){++rows;TEST_ASSERT(line.find(",TinyTarget,1")!=std::string::npos);}
    TEST_ASSERT_EQ(rows,4);
}
void test_bc_match_failure_degrades_to_tt_and_reuses_pyramids() {
    Fixture f(48);f.cfg.stereo_flow.min_matches=11;auto t=f.tracker();
    auto r=t->process(f.left,f.right,false,&f.roi,nullptr);assertDisparity(r,35.25);
    TEST_ASSERT(r.fallback_used&&r.strategy_name=="TinyTarget");
    TEST_ASSERT(r.route_detail.find("C0BC:")!=std::string::npos&&r.route_detail.find("C0TT:OK")!=std::string::npos);
    TEST_ASSERT(r.flow_pyramid_builds<=2); // ROI and full, each built only once across attempts.
    TEST_ASSERT(r.extract_ms>=r.stereo_match_ms&&r.stereo_match_ms>0);
    TEST_ASSERT_EQ(t->getLogs().size(),size_t(1));
}
void test_bc_extraction_failure_degrades_without_right_detector() {
    Fixture f(24);f.bc.corners=3;f.cfg.tiny_max_area=100;
    cv::Mat mask(f.left.size(),CV_8UC1,cv::Scalar(0));
    cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{{{166,91},{190,91},{178,115}}},255);
    cv::RNG rng(571);
    for(int y=91;y<115;++y)for(int x=166;x<190;++x)
        if(!mask.at<uchar>(y,x))f.left.at<uchar>(y,x)=static_cast<uchar>(rng.uniform(0,25));
    cv::Mat shift=(cv::Mat_<double>(2,3)<<1,0,-35.25,0,1,1.774);
    cv::warpAffine(f.left,f.right,shift,f.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
    auto t=f.tracker();int calls=0;
    auto r=t->process(f.left,f.right,false,&f.roi,nullptr,[&](){++calls;return std::make_pair(f.roi,f.roi);});
    assertDisparity(r,35.25);TEST_ASSERT_EQ(calls,0);TEST_ASSERT(r.fallback_used);
    TEST_ASSERT(r.route_detail.find("C0BC:bc_extraction_failed")!=std::string::npos);
}
void test_class1_direct_bc_tt_and_parameter_reset() {
    Fixture f(48);auto t=f.tracker();RoiGroup c1=f.roi;c1.is_class1=true;
    auto r=t->process(f.left,f.right,false,&c1,nullptr);assertDisparity(r,35.25);
    TEST_ASSERT(r.is_class1&&r.strategy_name=="BinaryCorner");
    for(int region:r.flow_regions)TEST_ASSERT_EQ(region,1);
    f.cfg.tiny_max_area_class1=10000;auto tiny=f.tracker();
    auto rt=tiny->process(f.left,f.right,false,&c1,nullptr);assertDisparity(rt,35.25);
    TEST_ASSERT(rt.is_class1&&rt.strategy_name=="TinyTarget");
    auto normal=tiny->process(f.left,f.right,false,&f.roi,nullptr);assertDisparity(normal,35.25);
    TEST_ASSERT(!normal.is_class1&&normal.strategy_name=="BinaryCorner");
}
void test_class1_bc_to_tt_and_supplied_right_roi() {
    Fixture f(48);f.cfg.stereo_flow.min_matches=11;f.roi.is_class1=true;auto t=f.tracker();
    auto r=t->process(f.left,f.right,false,&f.roi,&f.roi);assertDisparity(r,35.25);
    TEST_ASSERT(r.is_class1&&r.fallback_used&&r.strategy_name=="TinyTarget");
    TEST_ASSERT(r.route_detail.find("C1TT:OK")!=std::string::npos);
}
void test_invalid_inputs_never_fall_through_to_pose() {
    Fixture f;auto t=f.tracker();
    auto r=t->process(f.left,cv::Mat(),false,&f.roi,nullptr);
    TEST_ASSERT(r.disparity_only&&!r.disparity_success&&!r.success&&r.pnp_ms==0);
    TEST_ASSERT_EQ(r.failure_reason,std::string("invalid_stereo_images"));
    auto bad=f.roi;bad.primary.x=-10;
    r=t->process(f.left,f.right,false,&bad,nullptr);TEST_ASSERT_EQ(r.failure_reason,std::string("invalid_left_roi"));
    TEST_ASSERT_EQ(t->getLogs().size(),size_t(2));
}
void test_occlusion_rejects_incomplete_tt_and_records_partial_status() {
    Fixture f;auto t=f.tracker();f.right.setTo(0);
    auto r=t->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT(r.disparity_only&&!r.disparity_success&&!r.success&&r.pnp_ms==0);
    TEST_ASSERT(!r.fallback_used&&!r.failure_reason.empty());
}
void test_config_default_validation_and_legacy_selector() {
    TEST_ASSERT(!StereoFlowConfig{}.disparity_full_chain);
    cv::FileStorage fs("{\"method\":\"left_bc_lk\",\"disparity_only\":true,\"disparity_full_chain\":true}",
        cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(readStereoFlowConfig(fs.root()).disparity_full_chain);
    StereoFlowConfig bad;bad.disparity_full_chain=true;bool rejected=false;
    try{validateStereoFlowConfig(bad);}catch(const std::invalid_argument&){rejected=true;}
    TEST_ASSERT(rejected);
    Fixture f;f.cfg.stereo_flow.disparity_full_chain=false;auto t=f.tracker();
    TEST_ASSERT(!t->prefersLeftStereoFlow(f.roi));
    auto r=t->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT_EQ(r.failure_reason,std::string("disparity_only_requires_supported_flow_strategy"));
}
void test_negative_disparity_bc_is_terminal() {
    Fixture f(48,-35.25,0);auto t=f.tracker();
    auto r=t->process(f.left,f.right,false,&f.roi,nullptr);
    TEST_ASSERT(!r.disparity_success&&!r.fallback_used&&!r.success);
    TEST_ASSERT_EQ(r.failure_reason,std::string("negative_disparity_check_input_order"));
    TEST_ASSERT(r.route_detail.find("C0TT:")==std::string::npos);
}
void test_temporal_switch_and_cache_remain_pose_free() {
    Fixture f(48);f.cfg.temporal.enabled=true;f.cfg.temporal.hold_frames=2;
    f.cfg.temporal.area_ema_alpha=1;auto t=f.tracker();
    auto bc=t->process(f.left,f.right,false,&f.roi,nullptr);assertDisparity(bc,35.25);
    RoiGroup small=f.roi;small.primary={162,87,28,28}; // Target truncation may fail, but must not authorize a pose.
    auto r=t->process(f.left,f.right,false,&small,nullptr);
    TEST_ASSERT(r.disparity_only&&!r.success&&r.pnp_ms==0);
    auto recovered=t->process(f.left,f.right,false,&f.roi,nullptr);assertDisparity(recovered,35.25);
    TEST_ASSERT(bc.is_first_frame&&r.is_first_frame&&recovered.is_first_frame);
    TEST_ASSERT_EQ(t->getLogs().size(),size_t(3));
}
void test_tiny_rotated_and_small_targets() {
    for(int size:{12,16,24})for(double angle:{0.,25.,90.}) {
        Fixture f(size,23.5,-1.774);f.cfg.tiny_max_area=10000;
        cv::Mat transform=cv::getRotationMatrix2D({166.f+size/2.f,91.f+size/2.f},angle,1);
        cv::warpAffine(f.left,f.left,transform,f.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
        cv::Mat shift=(cv::Mat_<double>(2,3)<<1,0,-23.5,0,1,-1.774);
        cv::warpAffine(f.left,f.right,shift,f.left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,0);
        // Include the rotated object's bounding box and context, rather than truncating it.
        f.roi.primary={154,79,size+24,size+24};auto t=f.tracker();
        auto r=t->process(f.left,f.right,false,&f.roi,nullptr);
        assertDisparity(r,23.5);TEST_ASSERT_EQ(r.strategy_name,std::string("TinyTarget"));
    }
}
void runExperiment() {
    const std::filesystem::path output="output/disparity_full_chain_experiment/tiny";
    std::filesystem::create_directories(output);
    std::ofstream samples(output/"samples.csv"),matrix(output/"matrix.csv");
    samples<<"size,path,iteration,ms,success,pairs,mae_px,max_error_px\n";
    matrix<<"size,disparity,dy,success,pairs,mae_px,max_error_px\n";
    samples<<std::setprecision(10);matrix<<std::setprecision(10);
    auto errors=[](const PipelineResult& r,double expected) {
        double sum=0,maximum=0;for(double d:r.disparity){const double e=std::abs(d-expected);sum+=e;maximum=std::max(maximum,e);}
        return std::make_pair(r.disparity.empty()?-1:sum/r.disparity.size(),maximum);
    };
    for(int size:{8,12,16,24,32,48})for(double d:{5.5,35.25,75.5})for(double dy:{-1.774,0.,1.774}) {
        Fixture f(size,d,dy);f.cfg.tiny_max_area=100000;auto t=f.tracker();t->setOutputDir("");
        auto r=t->process(f.left,f.right,false,&f.roi,nullptr);auto e=errors(r,d);
        matrix<<size<<','<<d<<','<<dy<<','<<r.disparity_success<<','<<r.disparity.size()<<','<<e.first<<','<<e.second<<'\n';
    }
    for(int size:{16,24,48}) {
        Fixture f(size);f.cfg.tiny_max_area=100000;auto t=f.tracker();t->setOutputDir("");
        TinyTargetExtractor legacy(f.tt,f.root.string());legacy.setDebugCapture(false);
        const cv::Rect lroi(f.roi.primary.x-5,f.roi.primary.y-5,f.roi.primary.width+10,f.roi.primary.height+10);
        const cv::Rect rroi(lroi.x-35,lroi.y+2,lroi.width,lroi.height);
        Eigen::Matrix3d k;k<<500,0,160,0,500,120,0,0,1;
        auto camera=makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),{60,0,0});
        auto old=[&]() {
            auto r=legacy.extract(f.left(lroi),f.right(rroi),{},{});
            if(r.pts_left_match.size()!=4||r.pts_right_good.size()!=4)return PipelineResult{};
            for(auto& p:r.pts_left_match)p+=cv::Point2f(lroi.x,lroi.y);
            for(auto& p:r.pts_right_good)p+=cv::Point2f(rroi.x,rroi.y);
            auto checked=StereoFlowMatcher(camera,f.cfg.stereo_flow).match(f.left,f.right,r.pts_left_match,r.pts_right_good,lroi);
            r.disparity.clear();for(size_t j=0;j<checked.left_points.size();++j)
                r.disparity.push_back(checked.left_points[j].x-checked.right_points[j].x);
            r.disparity_success=checked.success;return r;
        };
        auto flow=[&](){return t->process(f.left,f.right,false,&f.roi,nullptr);};
        for(int i=0;i<8;++i){old();flow();}
        for(int i=0;i<30;++i)for(int order=0;order<2;++order) {
            const bool left_flow=(i+order)%2==1;
            const auto start=std::chrono::steady_clock::now();auto r=left_flow?flow():old();
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            auto e=errors(r,35.25);
            samples<<size<<','<<(left_flow?"left_tt_flow":"dual_tt_checked")<<','<<i<<','<<ms<<','
                <<r.disparity_success<<','<<r.disparity.size()<<','<<e.first<<','<<e.second<<'\n';
        }
    }
}
}
REGISTER_TEST(test_direct_tiny_flow_and_visual_csv);
REGISTER_TEST(test_bc_match_failure_degrades_to_tt_and_reuses_pyramids);
REGISTER_TEST(test_bc_extraction_failure_degrades_without_right_detector);
REGISTER_TEST(test_class1_direct_bc_tt_and_parameter_reset);
REGISTER_TEST(test_class1_bc_to_tt_and_supplied_right_roi);
REGISTER_TEST(test_invalid_inputs_never_fall_through_to_pose);
REGISTER_TEST(test_occlusion_rejects_incomplete_tt_and_records_partial_status);
REGISTER_TEST(test_config_default_validation_and_legacy_selector);
REGISTER_TEST(test_negative_disparity_bc_is_terminal);
REGISTER_TEST(test_temporal_switch_and_cache_remain_pose_free);
REGISTER_TEST(test_tiny_rotated_and_small_targets);
int main(){const int failures=gpnp_test::TestRegistry::instance().runAll();
    if(!failures&&std::getenv("ST_CHAIN_EXPERIMENT"))runExperiment();
    return failures;
}
