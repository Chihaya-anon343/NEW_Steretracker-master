#include "../framework/TestAssert.hpp"
#include "feature/TinyTargetExtractor.hpp"
#include "common/TinyTargetPolicy.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

using namespace gpnp;
namespace {
// The four reference vertices below belong to this synthetic fixture's source
// square. They are deliberately virtual bbox vertices, not claimed physical
// contour vertices or validation of the production 295/17 mm model.
std::vector<cv::Point2f> pattern() {
    return {{16,0},{16,36},{0,36},{0,49},{49,49},{49,36},{34,36},{34,14},{49,14},{49,0}};
}
struct TemplateFixture {
    std::filesystem::path directory;
    explicit TemplateFixture(bool symmetric=false,bool empty=false) {
        const auto nonce=std::chrono::high_resolution_clock::now().time_since_epoch().count();
        directory=std::filesystem::temp_directory_path()/("tt_robustness_"+std::to_string(nonce));
        std::filesystem::create_directory(directory);
        if(empty)return;
        cv::Mat mask=cv::Mat::zeros(50,50,CV_8U);
        const auto points=symmetric?std::vector<cv::Point2f>{{0,0},{49,0},{49,49},{0,49}}:pattern();
        std::vector<cv::Point> polygon;for(auto p:points)polygon.emplace_back(cvRound(p.x),cvRound(p.y));
        cv::fillPoly(mask,std::vector<std::vector<cv::Point>>{polygon},255);
        if(!cv::imwrite((directory/"0_degrees.png").string(),mask))throw std::runtime_error("Cannot write TT fixture");
        std::ofstream annotation(directory/"0_degrees.txt");
        for(size_t i=0;i<points.size();++i)annotation<<"Corner_"<<i<<": "<<points[i].x<<", "<<points[i].y<<'\n';
    }
    ~TemplateFixture() {
        // Remove only the unique fixture directory immediately inside temp.
        std::error_code error;
        if(directory.parent_path()==std::filesystem::temp_directory_path() &&
           directory.filename().string().find("tt_robustness_")==0)
            std::filesystem::remove_all(directory,error);
    }
};
struct Scene {cv::Mat gray;std::vector<cv::Point2f> reference;};
Scene scene(double degrees,bool square=false,double scale=1.5) {
    const double a=degrees*CV_PI/180.;
    auto transform=[&](cv::Point2f p){
        const double x=(p.x-24.5)*scale,y=(p.y-24.5)*scale;
        return cv::Point2f(float(85+std::cos(a)*x-std::sin(a)*y),float(85+std::sin(a)*x+std::cos(a)*y));
    };
    Scene result;result.gray=cv::Mat::zeros(170,170,CV_8U);
    auto points=square?std::vector<cv::Point2f>{{0,0},{49,0},{49,49},{0,49}}:pattern();
    std::vector<cv::Point> polygon;
    for(auto p:points){p=transform(p);polygon.emplace_back(cvRound(p.x),cvRound(p.y));}
    cv::fillPoly(result.gray,std::vector<std::vector<cv::Point>>{polygon},255);
    for(auto p:std::vector<cv::Point2f>{{0,0},{49,0},{49,49},{0,49}})result.reference.push_back(transform(p));
    return result;
}
TinyTargetExtractor::Config robustConfig() {
    TinyTargetExtractor::Config cfg;cfg.robustness.enabled=true;cfg.robustness.diagnostics=true;
    cfg.square_size_m_class0=.295;cfg.square_size_m_class1=.017;
    return cfg;
}
void assertSameObservations(const PipelineResult& a,const PipelineResult& b) {
    TEST_ASSERT(a.success==b.success);TEST_ASSERT(a.pts_left_match==b.pts_left_match);
    TEST_ASSERT(a.pts_left_good==b.pts_left_good);TEST_ASSERT(a.pts_right_good==b.pts_right_good);
    TEST_ASSERT(a.n_kp_left==b.n_kp_left);TEST_ASSERT(a.good_matches.size()==b.good_matches.size());
    for(size_t i=0;i<a.good_matches.size();++i) {
        TEST_ASSERT(a.good_matches[i].queryIdx==b.good_matches[i].queryIdx);
        TEST_ASSERT(a.good_matches[i].trainIdx==b.good_matches[i].trainIdx);
    }
}
void test_tt_diagnostics_only_preserves_legacy_observations() {
    TemplateFixture fixture;TinyTargetExtractor::Config legacy;
    auto diagnostic=legacy;diagnostic.robustness.diagnostics=true;
    TinyTargetExtractor a(legacy,fixture.directory.string()),b(diagnostic,fixture.directory.string());
    a.setMonoContext(true);b.setMonoContext(true);a.setDebugCapture(false);b.setDebugCapture(false);
    for(bool class1:{false,true})for(double yaw:{0.,45.,90.,180.,270.}) {
        auto s=scene(yaw);a.setUseClass1(class1);b.setUseClass1(class1);
        auto old=a.extractMono(s.gray,{}),observed=b.extractMono(s.gray,{});
        assertSameObservations(old,observed);
        TEST_ASSERT(a.lastMatchedAngle()==b.lastMatchedAngle());
        TEST_ASSERT(a.lastMatchOverlap()==b.lastMatchOverlap());
        TEST_ASSERT(b.lastQuality().span_px>0);TEST_ASSERT(b.lastQuality().initial_corners.size()==4);
    }
}
void test_tt_fourway_preserves_physical_ids_and_unique_winding() {
    TemplateFixture fixture;auto cfg=robustConfig();cfg.robustness.guarded_refinement=false;
    TinyTargetExtractor extractor(cfg,fixture.directory.string());extractor.setMonoContext(true);
    for(double yaw:{0.,44.9,45.,45.1,90.,135.,180.,225.,270.,315.}) {
        auto s=scene(yaw);auto r=extractor.extractMono(s.gray,{});const auto& q=extractor.lastQuality();
        TEST_ASSERT_MSG(r.success,"yaw="<<yaw<<" status="<<q.status<<" iou="<<q.direction_iou<<" gap="<<q.direction_gap);
        TEST_ASSERT(q.direction_reliable);TEST_ASSERT(q.geometry_valid);TEST_ASSERT(r.pts_left_match.size()==4);
        TEST_ASSERT(std::abs(cv::contourArea(r.pts_left_match))>100);
        for(int i=0;i<4;++i) {
            TEST_ASSERT_MSG(cv::norm(r.pts_left_match[i]-s.reference[i])<2.0,"yaw="<<yaw<<" id="<<i<<" point="<<r.pts_left_match[i]<<" reference="<<s.reference[i]);
            TEST_ASSERT(r.good_matches[i].trainIdx==i);
            for(int j=i+1;j<4;++j)TEST_ASSERT(cv::norm(r.pts_left_match[i]-r.pts_left_match[j])>10);
        }
    }
}
void test_tt_fourway_rejects_unresolvable_direction() {
    TemplateFixture fixture(true);TinyTargetExtractor extractor(robustConfig(),fixture.directory.string());
    extractor.setMonoContext(true);auto r=extractor.extractMono(scene(0,true).gray,{});
    TEST_ASSERT(!r.success);TEST_ASSERT(extractor.lastQuality().status=="direction_ambiguous");
    TEST_ASSERT(extractor.lastQuality().direction_gap<.001);TEST_ASSERT(r.pts_left_match.empty());
}
void test_tt_missing_template_and_low_shape_are_not_default_zero_degree() {
    TemplateFixture empty(false,true),fixture;
    TinyTargetExtractor extractor(robustConfig(),empty.directory.string());extractor.setMonoContext(true);
    auto r=extractor.extractMono(scene(0).gray,{});
    TEST_ASSERT(!r.success);TEST_ASSERT(extractor.lastQuality().status=="no_templates");
    auto cfg=robustConfig();cfg.robustness.min_direction_iou=.99;
    TinyTargetExtractor wrong(cfg,fixture.directory.string());wrong.setMonoContext(true);
    r=wrong.extractMono(scene(0,true).gray,{});
    TEST_ASSERT(!r.success);TEST_ASSERT(wrong.lastQuality().status=="direction_low_iou");
}
void test_tt_success_then_empty_or_flat_does_not_keep_stale_state() {
    TemplateFixture fixture;TinyTargetExtractor extractor(robustConfig(),fixture.directory.string());
    extractor.setMonoContext(true);TEST_ASSERT(extractor.extractMono(scene(0).gray,{}).success);
    auto empty=extractor.extractMono({},{});TEST_ASSERT(!empty.success);
    TEST_ASSERT(extractor.lastMatchedAngle()==-1);TEST_ASSERT(extractor.lastMatchOverlap()==0);
    TEST_ASSERT(extractor.lastMatchedTemplate()==nullptr);TEST_ASSERT(extractor.lastLeftDebug().super_binary.empty());
    TEST_ASSERT(extractor.lastQuality().initial_corners.empty());TEST_ASSERT(extractor.lastQuality().status=="empty_input");
    auto flat=extractor.extractMono(cv::Mat::zeros(80,80,CV_8U),{});TEST_ASSERT(!flat.success);
    TEST_ASSERT(extractor.lastQuality().status=="no_component");TEST_ASSERT(extractor.lastQuality().quadrant==-1);
}
void test_tt_guarded_refinement_has_original_pixel_bounds_and_capture_independence() {
    TemplateFixture fixture;auto cfg=robustConfig();cfg.robustness.max_refine_shift_px=.15;
    TinyTargetExtractor extractor(cfg,fixture.directory.string());extractor.setMonoContext(true);
    for(double scale:{.45,1.5}) {
        auto s=scene(17,false,scale);cv::GaussianBlur(s.gray,s.gray,cv::Size(3,3),.7);
        extractor.setDebugCapture(true);auto captured=extractor.extractMono(s.gray,{});auto q=extractor.lastQuality();
        TEST_ASSERT_MSG(captured.success,q.status);TEST_ASSERT(q.initial_corners.size()==4);TEST_ASSERT(q.final_corners.size()==4);
        TEST_ASSERT(q.accepted_refinement.size()==4);
        int accepted=0;
        for(int i=0;i<4;++i) {
            const double distance=cv::norm(q.final_corners[i]-q.initial_corners[i]);
            TEST_ASSERT(distance<=.15001);
            if(q.accepted_refinement[i])++accepted;
            else TEST_ASSERT(distance<1e-6);
        }
        TEST_ASSERT(accepted==q.refined_count);
        extractor.setDebugCapture(false);auto quiet=extractor.extractMono(s.gray,{});
        assertSameObservations(captured,quiet);
        TEST_ASSERT(extractor.lastQuality().normalized.empty());TEST_ASSERT(extractor.lastQuality().reference.empty());
    }
}
void test_tt_model_class_switch_and_offset_do_not_accumulate() {
    TemplateFixture fixture;auto cfg=robustConfig();cfg.template_geometry.enabled=true;
    cfg.template_geometry.class1_center_in_class0_mm={12,-7};
    TinyTargetExtractor extractor(cfg,fixture.directory.string());extractor.setMonoContext(true);
    const auto original=extractor.templateData().pts_3d;TEST_ASSERT(original.size()==4);
    for(int repeat=0;repeat<3;++repeat) {
        extractor.setUseClass1(true);const auto small=extractor.templateData().pts_3d;
        TEST_ASSERT(small.size()==4);TEST_ASSERT_NEAR((small[1]-small[0]).norm(),17,1e-12);
        Eigen::Vector3d center=Eigen::Vector3d::Zero();for(const auto& p:small)center+=p/4;
        TEST_ASSERT_NEAR(center.x(),12,1e-12);TEST_ASSERT_NEAR(center.y(),-7,1e-12);
        TEST_ASSERT(extractor.extractMono(scene(90).gray,{}).success);
        extractor.setUseClass1(false);const auto restored=extractor.templateData().pts_3d;
        for(int i=0;i<4;++i)TEST_ASSERT((restored[i]-original[i]).norm()==0);
    }
}
void test_tt_robustness_is_inert_in_stereo_context() {
    TemplateFixture fixture;TinyTargetExtractor::Config legacy;auto robust=robustConfig();
    legacy.square_size_m_class0=robust.square_size_m_class0;
    legacy.square_size_m_class1=robust.square_size_m_class1;
    TinyTargetExtractor a(legacy,fixture.directory.string()),b(robust,fixture.directory.string());
    a.setDebugCapture(false);b.setDebugCapture(false);auto s=scene(45);
    assertSameObservations(a.extractMono(s.gray,{}),b.extractMono(s.gray,{}));
    assertSameObservations(a.extract(s.gray,s.gray,{},{}),b.extract(s.gray,s.gray,{},{}));
    b.setMonoContext(true);TEST_ASSERT(b.extractMono(s.gray,{}).success);
    b.setMonoContext(false);assertSameObservations(a.extractMono(s.gray,{}),b.extractMono(s.gray,{}));
    TEST_ASSERT(b.lastQuality().quadrant==-1);TEST_ASSERT(!b.lastQuality().direction_reliable);
}
void test_tt_robustness_configuration_rejects_invalid_types_and_bounds() {
    for(const std::string& value:{"[]","{\"enabled\":\"true\"}","{\"enabled\":2}",
        "{\"min_direction_iou\":1.1}","{\"min_direction_gap\":-0.1}","{\"min_span_px\":0}",
        "{\"min_gradient_ratio\":2}","{\"max_refine_shift_px\":5}","{\"min_span_px\":\"four\"}"}) {
        cv::FileStorage fs("{\"robustness\":"+value+"}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
        TEST_ASSERT_THROWS(readTinyTargetRobustnessConfig(fs["robustness"]),std::exception);
    }
    auto cfg=robustConfig();cfg.robustness.min_direction_iou=std::numeric_limits<double>::quiet_NaN();
    TEST_ASSERT_THROWS(validateTinyTargetRobustnessConfig(cfg.robustness),std::exception);
    TemplateFixture fixture;
    for(int scale:{0,-1,1000}) {cfg=robustConfig();cfg.scale_factor=scale;
        TEST_ASSERT_THROWS(TinyTargetExtractor(cfg,fixture.directory.string()),std::exception);}
    cfg=robustConfig();cfg.target_size={0,50};TEST_ASSERT_THROWS(TinyTargetExtractor(cfg,fixture.directory.string()),std::exception);
}
TinyTargetRoiEvidence evidence() {
    TinyTargetRoiEvidence e;e.extracted=true;e.finite=true;e.geometry_valid=true;e.points=4;
    e.legacy_iou=.70;e.direction_iou=.85;e.direction_gap=.20;e.emitted_direction_consistent=true;
    e.span_x=26;e.span_y=26;return e;
}
void test_tt_roi_policy_legacy_threshold_is_unchanged() {
    RoiSelectionConfig config;auto e=evidence();
    TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e.legacy_iou=.75;e.geometry_valid=false;
    TEST_ASSERT(evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e.legacy_iou=.749999;TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
}
void test_tt_roi_policy_small_target_needs_independent_direction_and_class_size() {
    RoiSelectionConfig config;config.tt_verification.enabled=true;auto e=evidence();
    TEST_ASSERT(evaluateTinyTargetRoi(e,config,false,36,36).reason=="tt_shape_below_strict");
    config.tt_verification.allow_small_target_path=true;
    auto decision=evaluateTinyTargetRoi(e,config,false,36,36);
    TEST_ASSERT(decision.accepted);TEST_ASSERT(decision.small_target_path);
    TEST_ASSERT(!evaluateTinyTargetRoi(e,config,true,36,36).accepted);
    e.span_x=e.span_y=20;TEST_ASSERT(evaluateTinyTargetRoi(e,config,true,30,30).accepted);
    e.direction_gap=.01;TEST_ASSERT(evaluateTinyTargetRoi(e,config,true,30,30).reason=="tt_direction_ambiguous");
    e.direction_gap=.20;e.emitted_direction_consistent=false;
    TEST_ASSERT(evaluateTinyTargetRoi(e,config,true,30,30).reason=="tt_direction_inconsistent");
    e.emitted_direction_consistent=true;e.direction_iou=.1;
    TEST_ASSERT(evaluateTinyTargetRoi(e,config,true,30,30).reason=="tt_direction_shape");
    e.legacy_iou=.8;TEST_ASSERT(evaluateTinyTargetRoi(e,config,true,30,30).accepted);
}
void test_tt_roi_policy_rejects_floor_invalid_geometry_and_nonfinite() {
    RoiSelectionConfig config;config.tt_verification.enabled=true;
    config.tt_verification.allow_small_target_path=true;auto valid=evidence();
    auto e=valid;e.legacy_iou=.64;TEST_ASSERT(evaluateTinyTargetRoi(e,config,false,36,36).reason=="tt_shape_below_floor");
    e=valid;e.geometry_valid=false;TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e=valid;e.span_x=2;TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e=valid;e.points=3;TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e=valid;e.legacy_iou=std::numeric_limits<double>::quiet_NaN();TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e=valid;e.span_x=std::numeric_limits<double>::quiet_NaN();TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
    e=valid;e.direction_gap=std::numeric_limits<double>::infinity();TEST_ASSERT(!evaluateTinyTargetRoi(e,config,false,36,36).accepted);
}
void test_tt_roi_configuration_checks_score_ordering_and_types() {
    for(const std::string& value:{"[]","{\"enabled\":\"yes\"}","{\"enabled\":2}",
        "{\"min_shape_class0\":1.2}","{\"small_min_shape_class1\":0.9}",
        "{\"small_span_class0\":0}","{\"min_direction_gap\":-1}","{\"small_span_class1\":\"x\"}",
        "{\"allow_small_target_path\":\"true\"}","{\"allow_small_target_path\":2}"}) {
        cv::FileStorage fs("{\"verification\":"+value+"}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
        TEST_ASSERT_THROWS(readTinyTargetVerificationConfig(fs["verification"]),std::exception);
    }
    cv::FileStorage fs("{\"verification\":{\"enabled\":true,\"small_span_class1\":30}}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    auto config=readTinyTargetVerificationConfig(fs["verification"]);TEST_ASSERT(config.enabled);TEST_ASSERT(config.small_span_class1==30);
}
REGISTER_TEST(test_tt_diagnostics_only_preserves_legacy_observations);
REGISTER_TEST(test_tt_fourway_preserves_physical_ids_and_unique_winding);
REGISTER_TEST(test_tt_fourway_rejects_unresolvable_direction);
REGISTER_TEST(test_tt_missing_template_and_low_shape_are_not_default_zero_degree);
REGISTER_TEST(test_tt_success_then_empty_or_flat_does_not_keep_stale_state);
REGISTER_TEST(test_tt_guarded_refinement_has_original_pixel_bounds_and_capture_independence);
REGISTER_TEST(test_tt_model_class_switch_and_offset_do_not_accumulate);
REGISTER_TEST(test_tt_robustness_is_inert_in_stereo_context);
REGISTER_TEST(test_tt_robustness_configuration_rejects_invalid_types_and_bounds);
REGISTER_TEST(test_tt_roi_policy_legacy_threshold_is_unchanged);
REGISTER_TEST(test_tt_roi_policy_small_target_needs_independent_direction_and_class_size);
REGISTER_TEST(test_tt_roi_policy_rejects_floor_invalid_geometry_and_nonfinite);
REGISTER_TEST(test_tt_roi_configuration_checks_score_ordering_and_types);
}
int main(){g_verbose_console=false;cv::setNumThreads(1);return gpnp_test::TestRegistry::instance().runAll();}
