#include "../framework/TestAssert.hpp"
#include "matching/MonoCornerIdentity.hpp"
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
using namespace gpnp;
namespace {
std::vector<cv::Point2f> shape() {
    return {{16,0},{16,36},{0,36},{0,50},{50,50},{50,36},{34,36},{34,14},{50,14},{50,0}};
}
struct Scene {std::vector<cv::Point2f> points;cv::Mat mask;};
Scene scene(double yaw,double perspective=0) {
    double a=yaw*CV_PI/180;
    cv::Matx33d T(1,0,320,0,1,240,0,0,1),C(1,0,-25,0,1,-25,0,0,1);
    cv::Matx33d A(4*std::cos(a),-4*std::sin(a),0,4*std::sin(a),4*std::cos(a),0,
                  perspective,-perspective*.6,1);
    Scene s; cv::perspectiveTransform(shape(),s.points,cv::Mat(T*A*C));
    s.mask=cv::Mat::zeros(480,640,CV_8U);std::vector<cv::Point> polygon;
    for(auto p:s.points) polygon.emplace_back(cvRound(p.x),cvRound(p.y));
    cv::fillPoly(s.mask,std::vector<std::vector<cv::Point>>{polygon},255);return s;
}
void test_mono_identity_cyclic_and_reversed_perspective() {
    MonoCornerIdentityConfig cfg;cfg.method="hybrid";MonoCornerIdentity matcher(cfg);matcher.setModel(shape());
    for(double yaw:{0.,45.,90.,175.,270.}) for(double perspective:{0.,.006}) {
        auto s=scene(yaw,perspective);
        for(int direction:{1,-1}) for(int shift=0;shift<10;++shift) {
            std::vector<cv::Point2f> observed;for(int i=0;i<10;++i) observed.push_back(s.points[(shift+direction*i+10)%10]);
            auto result=matcher.match(observed,s.mask,true);
            TEST_ASSERT(result.accepted);TEST_ASSERT(result.indices.size()==10);
            for(int i=0;i<10;++i) TEST_ASSERT(cv::norm(observed[result.indices[i]]-s.points[i])<.01);
            TEST_ASSERT(!result.normalized.empty());TEST_ASSERT(result.iou>.90);
        }
    }
}
void test_fourway_fast_path_and_strict_ablation() {
    MonoCornerIdentityConfig cfg;cfg.method="tt_fourway";MonoCornerIdentity matcher(cfg);matcher.setModel(shape());
    for(double yaw:{0.,90.,180.,270.}) {
        auto s=scene(yaw);auto r=matcher.match(s.points,s.mask);
        TEST_ASSERT(r.accepted);TEST_ASSERT(r.path=="tt_fourway");TEST_ASSERT(r.hypotheses==1);
        TEST_ASSERT(r.normalized.empty()); // capture=false must not retain snapshots
    }
}
void test_shape_evidence_required_and_bad_inputs() {
    MonoCornerIdentity matcher;matcher.setModel(shape());auto s=scene(20);
    auto empty=cv::Mat::zeros(s.mask.size(),CV_8U);
    TEST_ASSERT(!matcher.match(s.points,empty).accepted);
    auto duplicate=s.points;duplicate[1]=duplicate[0];
    TEST_ASSERT(matcher.match(duplicate,s.mask).status=="duplicate_points");
    auto bad=s.points;bad[1].x=std::numeric_limits<float>::quiet_NaN();
    TEST_ASSERT(!matcher.match(bad,s.mask).accepted);
    bad.pop_back();TEST_ASSERT(!matcher.match(bad,s.mask).accepted);
    auto small=shape();for(auto& p:small) p*=.2f;
    TEST_ASSERT(matcher.match(small,s.mask).status=="below_resolvable_span");
    auto corrupted=s.points;corrupted[2]+=cv::Point2f(70,50);corrupted[5]+=cv::Point2f(-60,50);corrupted[8]+=cv::Point2f(50,-70);
    TEST_ASSERT(!matcher.match(corrupted,s.mask).accepted);
}
void test_inner_component_cannot_use_outer_roi_scale() {
    MonoCornerIdentity matcher;matcher.setModel(shape());auto s=scene(0);
    for(auto& p:s.points) p=cv::Point2f(320,240)+(p-cv::Point2f(320,240))*.2f;
    TEST_ASSERT(matcher.match(s.points,s.mask).status=="component_too_small_for_roi");
}
void test_supported_outlier_and_rng_preserved() {
    MonoCornerIdentity matcher;matcher.setModel(shape());auto s=scene(25,.002);
    auto noisy=s.points;noisy[2]+=cv::Point2f(30,20);
    auto state=cv::theRNG().state;auto r=matcher.match(noisy,s.mask);
    TEST_ASSERT(r.accepted);TEST_ASSERT(cv::theRNG().state==state);
    for(int i=0;i<10;++i) TEST_ASSERT(r.indices[i]==i);
}
void test_symmetric_template_rejected_as_ambiguous() {
    MonoCornerIdentity matcher;std::vector<cv::Point2f> square{{0,0},{50,0},{50,50},{0,50}};matcher.setModel(square);
    cv::Mat image=cv::Mat::zeros(200,200,CV_8U);cv::rectangle(image,{50,50},{150,150},255,-1);
    auto r=matcher.match({{50,50},{150,50},{150,150},{50,150}},image);
    TEST_ASSERT(!r.accepted);TEST_ASSERT(r.status=="ambiguous_identity");
}
void test_config_rejects_invalid_boundaries() {
    for(const std::string& value:{"{\"mono_identity\":{\"enabled\":\"true\"}}",
        "{\"mono_identity\":{\"method\":\"unknown\"}}","{\"mono_identity\":{\"normalized_size\":4}}",
        "{\"mono_identity\":{\"min_iou\":1.1}}","{\"mono_identity\":{\"min_score_gap\":-1}}",
        "{\"mono_identity\":{\"max_fit_error_px\":0}}","{\"mono_identity\":{\"frontend\":\"unknown\"}}",
        "{\"mono_identity\":{\"refine_original\":\"yes\"}}"}) {
        bool threw=false;try {cv::FileStorage fs(value,cv::FileStorage::READ|cv::FileStorage::MEMORY);readMonoCornerIdentityConfig(fs["mono_identity"]);}
        catch(const std::exception&) {threw=true;} TEST_ASSERT(threw);
    }
    cv::FileStorage fs("{\"mono_identity\":{\"enabled\":true,\"method\":\"hybrid\"}}",cv::FileStorage::READ|cv::FileStorage::MEMORY);
    TEST_ASSERT(readMonoCornerIdentityConfig(fs["mono_identity"]).enabled);
}
void test_stereo_context_and_legacy_extraction_identical() {
    BinaryCornerExtractor::Config old_cfg;old_cfg.pixel_to_meter_scale_class0=.0059;
    auto new_cfg=old_cfg;new_cfg.mono_identity.enabled=true;
    BinaryCornerExtractor old(old_cfg,"data/NewMuBan(reordered)"),isolated(new_cfg,"data/NewMuBan(reordered)");
    old.setDebugCapture(false);isolated.setDebugCapture(false);auto s=scene(90,.004);
    for(bool stereo:{false,true}) {
        auto a=stereo ? old.extract(s.mask,s.mask,{},{}):old.extractMono(s.mask,{});
        auto b=stereo ? isolated.extract(s.mask,s.mask,{},{}):isolated.extractMono(s.mask,{});
        TEST_ASSERT(a.success==b.success);TEST_ASSERT(a.pts_left_match==b.pts_left_match);
        TEST_ASSERT(a.pts_left_good==b.pts_left_good);TEST_ASSERT(a.pts_right_good==b.pts_right_good);
        TEST_ASSERT(isolated.lastMonoIdentity().status=="disabled");
    }
    isolated.setMonoContext(true);auto b=isolated.extractMono(s.mask,{});
    TEST_ASSERT(b.success);TEST_ASSERT(isolated.lastMonoIdentity().accepted);
}
void test_class1_requires_explicit_opt_in() {
    BinaryCornerExtractor::Config cfg;cfg.mono_identity.enabled=true;cfg.pixel_to_meter_scale_class0=.0059;
    BinaryCornerExtractor a(cfg,"data/NewMuBan(reordered)");a.setMonoContext(true);a.setUseClass1(true);a.setDebugCapture(false);
    auto s=scene(90,.004);auto baseline=a.extractMono(s.mask,{});
    TEST_ASSERT(a.lastMonoIdentity().status=="class1_disabled");
    cfg.mono_identity.class1_enabled=true;BinaryCornerExtractor b(cfg,"data/NewMuBan(reordered)");
    b.setMonoContext(true);b.setUseClass1(true);b.setDebugCapture(false);auto updated=b.extractMono(s.mask,{});
    TEST_ASSERT(baseline.success);TEST_ASSERT(updated.success);TEST_ASSERT(b.lastMonoIdentity().accepted);
}
void test_normalized_frontend_perspective_and_original_refinement() {
    double unrefined_error=0,refined_error=0;int total_refined=0;
    for(bool refine:{false,true}) {
        MonoCornerIdentityConfig cfg;cfg.frontend="normalized";cfg.refine_original=refine;
        MonoCornerIdentity matcher(cfg);matcher.setModel(shape());
        for(double yaw:{0.,32.,90.,175.,270.}) for(double perspective:{0.,.002,.006}) {
            auto s=scene(yaw,perspective);auto state=cv::theRNG().state;
            auto r=matcher.extractNormalized(s.mask,true);
            TEST_ASSERT(r.accepted);TEST_ASSERT(r.points.size()==10);
            TEST_ASSERT(r.path.find("normalized_")==0);TEST_ASSERT(cv::theRNG().state==state);
            for(int i=0;i<10;++i) {
                double error=cv::norm(r.points[i]-s.points[i]);
                // Raw polygon vertices are intentionally an ablation, before
                // edge-intersection localization. The production path must
                // satisfy the tighter bound and improve aggregate error below.
                TEST_ASSERT_MSG(error<(refine?4.:6.),"refine="<<refine<<" yaw="<<yaw<<" perspective="<<perspective<<" id="<<i<<" error="<<error);
                (refine?refined_error:unrefined_error)+=error*error;
            }
            TEST_ASSERT(!r.coarse_normalized.empty());TEST_ASSERT(!r.foreground.empty());
            if(!refine) TEST_ASSERT(r.refined_points==0);
            total_refined+=r.refined_points;
        }
    }
    TEST_ASSERT(total_refined>0);TEST_ASSERT(refined_error<unrefined_error);
}
void test_normalized_no_invented_vertices_or_wrong_component() {
    MonoCornerIdentity matcher;matcher.setModel(shape());
    cv::Mat blank=cv::Mat::zeros(480,640,CV_8U);
    TEST_ASSERT(!matcher.extractNormalized(blank).accepted);
    cv::rectangle(blank,{180,100},{460,380},255,-1);
    auto wrong=matcher.extractNormalized(blank);
    TEST_ASSERT(!wrong.accepted);TEST_ASSERT(wrong.status=="normalized_corner_count");
    auto small=scene(0);cv::resize(small.mask,small.mask,{},.2,.2,cv::INTER_NEAREST);
    cv::Mat padded=cv::Mat::zeros(480,640,CV_8U);small.mask.copyTo(padded(cv::Rect(0,0,small.mask.cols,small.mask.rows)));
    TEST_ASSERT(matcher.extractNormalized(padded).status=="component_too_small_for_roi");
}
void test_normalized_context_switch_and_capture_independence() {
    BinaryCornerExtractor::Config cfg;cfg.pixel_to_meter_scale_class0=.0059;
    cfg.mono_identity.enabled=true;cfg.mono_identity.frontend="normalized";
    auto legacy=cfg;legacy.mono_identity.frontend="legacy";
    BinaryCornerExtractor a(cfg,"data/NewMuBan(reordered)"),b(legacy,"data/NewMuBan(reordered)");
    auto s=scene(32,.002);
    // An enabled normalized configuration must still be inert in stereo context.
    auto left=a.extractMono(s.mask,{}),right=b.extractMono(s.mask,{});
    TEST_ASSERT(left.pts_left_match==right.pts_left_match);
    a.setMonoContext(true);a.setDebugCapture(true);auto debug=a.extractMono(s.mask,{});
    TEST_ASSERT(debug.success);auto points=debug.pts_left_match;
    TEST_ASSERT(a.lastMatchedTemplate()->angle==0);
    a.setDebugCapture(false);auto quiet=a.extractMono(s.mask,{});
    TEST_ASSERT(quiet.success);TEST_ASSERT(points==quiet.pts_left_match);
    TEST_ASSERT(a.lastMonoIdentity().foreground.empty());
    TEST_ASSERT(quiet.good_matches.size()==10);TEST_ASSERT(quiet.pts_right_good.empty());
    for(int i=0;i<10;++i) TEST_ASSERT(quiet.good_matches[i].trainIdx==i);
    a.setUseClass1(true);b.setMonoContext(true);b.setUseClass1(true);
    TEST_ASSERT(a.extractMono(s.mask,{}).pts_left_match==b.extractMono(s.mask,{}).pts_left_match);
}
void test_quality_filter_preserves_sparse_model_ids() {
    MonoCornerIdentityConfig cfg;cfg.quality_filter=true;
    MonoCornerIdentity matcher(cfg);matcher.setModel(shape());auto s=scene(25,.002);
    for(int count:{1,2}) {
        auto noisy=s.points;noisy[2]+=cv::Point2f(30,20);
        if(count==2)noisy[8]+=cv::Point2f(-25,35);
        const auto state=cv::theRNG().state;auto result=matcher.match(noisy,s.mask,true);
        TEST_ASSERT(result.accepted);TEST_ASSERT(result.indices.size()==(size_t)(10-count));
        TEST_ASSERT(result.template_ids.size()==result.indices.size());TEST_ASSERT(cv::theRNG().state==state);
        TEST_ASSERT(result.rejected_ids==(count==1?std::vector<int>{2}:std::vector<int>{2,8}));
        for(size_t i=0;i<result.indices.size();++i){TEST_ASSERT(result.template_ids[i]!=2);TEST_ASSERT(result.indices[i]==result.template_ids[i]);}
    }
}
void test_partial_correspondence_missing_and_extra_points() {
    MonoCornerIdentityConfig cfg;cfg.quality_filter=true;cfg.partial_matching=true;
    MonoCornerIdentity matcher(cfg);matcher.setModel(shape());
    for(double yaw:{0.,45.,90.,175.,270.})for(double perspective:{0.,.002})for(int missing:{1,2}) {
        auto s=scene(yaw,perspective);std::vector<cv::Point2f> observed;
        for(int i=0;i<10;++i)if(i!=2&&(missing==1||i!=8))observed.push_back(s.points[i]);
        std::reverse(observed.begin(),observed.end());auto result=matcher.match(observed,s.mask,true);
        TEST_ASSERT_MSG(result.accepted,"yaw="<<yaw<<" perspective="<<perspective<<" missing="<<missing<<" status="<<result.status);
        TEST_ASSERT(result.indices.size()==(size_t)(10-missing));TEST_ASSERT(result.hypotheses<=cfg.max_partial_hypotheses);
        for(size_t i=0;i<result.indices.size();++i)TEST_ASSERT(cv::norm(observed[result.indices[i]]-s.points[result.template_ids[i]])<.01);
    }
    auto s=scene(32,.002);std::vector<cv::Point2f> extra;
    for(int count:{1,2}) {
        extra.clear();
        for(int i=0;i<10;++i) {
            extra.push_back(s.points[i]);
            if(i==2||(count==2&&i==6))extra.push_back((s.points[i]+s.points[i+1])*.5f);
        }
        auto r=matcher.match(extra,s.mask);TEST_ASSERT_MSG(r.accepted,r.status);TEST_ASSERT(r.indices.size()==10);
        for(size_t i=0;i<r.indices.size();++i)TEST_ASSERT(cv::norm(extra[r.indices[i]]-s.points[r.template_ids[i]])<.01);
    }
    auto insufficient=s.points;insufficient.resize(7);TEST_ASSERT(!matcher.match(insufficient,s.mask).accepted);
    cv::Mat wrong=cv::Mat::zeros(s.mask.size(),CV_8U);cv::rectangle(wrong,{200,120},{440,360},255,-1);
    TEST_ASSERT(!matcher.match(extra,wrong).accepted);
}
REGISTER_TEST(test_quality_filter_preserves_sparse_model_ids);
REGISTER_TEST(test_partial_correspondence_missing_and_extra_points);
REGISTER_TEST(test_mono_identity_cyclic_and_reversed_perspective);
REGISTER_TEST(test_fourway_fast_path_and_strict_ablation);
REGISTER_TEST(test_shape_evidence_required_and_bad_inputs);
REGISTER_TEST(test_inner_component_cannot_use_outer_roi_scale);
REGISTER_TEST(test_supported_outlier_and_rng_preserved);
REGISTER_TEST(test_symmetric_template_rejected_as_ambiguous);
REGISTER_TEST(test_config_rejects_invalid_boundaries);
REGISTER_TEST(test_stereo_context_and_legacy_extraction_identical);
REGISTER_TEST(test_class1_requires_explicit_opt_in);
REGISTER_TEST(test_normalized_frontend_perspective_and_original_refinement);
REGISTER_TEST(test_normalized_no_invented_vertices_or_wrong_component);
REGISTER_TEST(test_normalized_context_switch_and_capture_independence);
}
int main(){g_verbose_console=false;cv::setNumThreads(1);return gpnp_test::TestRegistry::instance().runAll();}
