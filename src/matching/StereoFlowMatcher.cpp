#include "matching/StereoFlowMatcher.hpp"
#include "common/Config.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <set>

namespace gpnp {
namespace {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
bool finite(const cv::Point2f& p) { return std::isfinite(p.x) && std::isfinite(p.y); }
bool inside(const cv::Point2f& p, const cv::Mat& image, int margin) {
    return finite(p) && p.x >= margin && p.y >= margin &&
           p.x < image.cols - margin - 1 && p.y < image.rows - margin - 1;
}
double correlation(const cv::Mat& left, const cv::Mat& right,
                   cv::Point2f pl, cv::Point2f pr, int size) {
    const int margin = size / 2;
    if (!inside(pl, left, margin) || !inside(pr, right, margin)) return -1;
    cv::Mat a, b;
    cv::getRectSubPix(left, cv::Size(size, size), pl, a, CV_32F);
    cv::getRectSubPix(right, cv::Size(size, size), pr, b, CV_32F);
    a -= cv::mean(a)[0]; b -= cv::mean(b)[0];
    const double aa = a.dot(a), bb = b.dot(b);
    if (aa < size * size || bb < size * size) return -1; // No distinctive local signal.
    return std::clamp(a.dot(b) / std::sqrt(aa * bb), -1.0, 1.0);
}
// A coarse search compares many right patches to the same left patch.
// Normalize the left patch once; final LK validation still uses correlation().
class CoarseCorrelation {
public:
    CoarseCorrelation(const cv::Mat& left, cv::Point2f point, int size) : size_(size) {
        if (!inside(point, left, size / 2)) return;
        cv::getRectSubPix(left, cv::Size(size, size), point, normalized_, CV_32F);
        normalized_ -= cv::mean(normalized_)[0];
        const double energy = normalized_.dot(normalized_);
        if (energy < size * size) { normalized_.release(); return; }
        normalized_ /= std::sqrt(energy);
    }
    double score(const cv::Mat& right, cv::Point2f point) const {
        if (normalized_.empty() || !inside(point, right, size_ / 2)) return -1;
        cv::Mat patch;
        cv::getRectSubPix(right, cv::Size(size_, size_), point, patch, CV_32F);
        patch -= cv::mean(patch)[0];
        const double energy = patch.dot(patch);
        if (energy < size_ * size_) return -1;
        return std::clamp(normalized_.dot(patch) / std::sqrt(energy), -1.0, 1.0);
    }
    cv::Mat grid(const cv::Mat& right, cv::Point2f first, int columns, int rows) const {
        cv::Mat scores(rows, columns, CV_32F, cv::Scalar(-1));
        if (normalized_.empty()) return scores;
        // All unit-spaced windows share the same subpixel phase. Sample their
        // union once, then evaluate normalized NCC with OpenCV's batch kernel.
        cv::Mat strip;
        cv::getRectSubPix(right, cv::Size(columns + size_ - 1, rows + size_ - 1),
            first + cv::Point2f((columns - 1) / 2.f, (rows - 1) / 2.f), strip, CV_32F);
        cv::matchTemplate(strip, normalized_, scores, cv::TM_CCOEFF_NORMED);
        cv::Mat sums, squares;
        cv::integral(strip, sums, squares, CV_64F, CV_64F);
        auto rectangle = [&](const cv::Mat& integral, int x, int y) {
            return integral.at<double>(y + size_, x + size_) - integral.at<double>(y, x + size_)
                 - integral.at<double>(y + size_, x) + integral.at<double>(y, x);
        };
        for (int y = 0; y < rows; ++y) for (int x = 0; x < columns; ++x) {
            const double sum = rectangle(sums, x, y);
            const double energy = rectangle(squares, x, y) - sum * sum / (size_ * size_);
            // Keep the original weak-texture rejection, rather than relying on
            // matchTemplate's treatment of constant windows.
            if (energy < size_ * size_) scores.at<float>(y, x) = -1;
        }
        return scores;
    }
private:
    int size_;
    cv::Mat normalized_;
};
double hullArea(const std::vector<cv::Point2f>& points) {
    if (points.size() < 3) return 0;
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    return std::abs(cv::contourArea(hull));
}
Eigen::Matrix3d skew(const Eigen::Vector3d& t) {
    Eigen::Matrix3d s;
    s << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
    return s;
}
} // namespace

void filterPlanarGeometry(const std::vector<cv::Point2f>& points,
                          const StereoFlowConfig& cfg, bool horizontal, FlowMatchResult& out) {
    if(!cfg.planar_geometry_filter)return;
    validateStereoFlowConfig(cfg);
    const auto start=Clock::now();
    auto done=[&](const std::string& status){out.planar_status=status;out.planar_ms=milliseconds(start);};
    std::vector<int> ids;
    if(out.diagnostics.size()!=points.size()){done("invalid_diagnostic_count");return;}
    for(int i=0;i<static_cast<int>(points.size());++i)if(out.diagnostics[i].accepted) {
        if(!finite(points[i])||!finite(out.diagnostics[i].tracked_right)){done("invalid_point");return;}
        ids.push_back(i);
    }
    const int n=static_cast<int>(ids.size()),sample_size=horizontal?3:4;
    if(n<cfg.planar_min_points){done("insufficient_redundancy");return;}
    const int required=std::max(sample_size+2,static_cast<int>(std::ceil(cfg.planar_min_inlier_ratio*n)));
    cv::Point2d center(0,0);double scale=0;
    for(int i:ids){center.x+=points[i].x;center.y+=points[i].y;}
    center*=1.0/n;
    for(int i:ids)scale=std::max(scale,std::hypot(points[i].x-center.x,points[i].y-center.y));
    if(scale<1e-6){done("degenerate_distribution");return;}
    auto row=[&](int k){const auto& p=points[ids[k]];return Eigen::Vector3d((p.x-center.x)/scale,(p.y-center.y)/scale,1);};
    auto disparity=[&](int k){int i=ids[k];return static_cast<double>(points[i].x-out.diagnostics[i].tracked_right.x);};
    // Fixed local RNG: independent of YOLO/OpenCV global RNG and repeatable across runs.
    std::mt19937 rng(0x504c414e);std::uniform_int_distribution<int> pick(0,n-1);
    std::set<std::vector<int>> samples;
    if(horizontal && static_cast<long long>(n)*(n-1)*(n-2)/6<=cfg.planar_max_hypotheses) {
        for(int a=0;a<n-2;++a)for(int b=a+1;b<n-1;++b)for(int c=b+1;c<n;++c)samples.insert({a,b,c});
    } else {
        for(int attempt=0;attempt<cfg.planar_max_hypotheses*8 && static_cast<int>(samples.size())<cfg.planar_max_hypotheses;++attempt) {
            std::set<int> sample;while(static_cast<int>(sample.size())<sample_size)sample.insert(pick(rng));
            samples.emplace(sample.begin(),sample.end());
        }
    }
    auto homography=[&](const std::vector<int>& use, bool minimal) {
        std::vector<cv::Point2f> a,b;
        for(int k:use){a.push_back(points[ids[k]]);b.push_back(out.diagnostics[ids[k]].tracked_right);}
        if(hullArea(a)<1e-6||hullArea(b)<1e-6)return cv::Mat();
        cv::Mat h=minimal?cv::getPerspectiveTransform(a,b):cv::findHomography(a,b,0);
        if(h.empty()||!cv::checkRange(h)||std::abs(cv::determinant(h))<1e-12)return cv::Mat();
        return h;
    };
    auto homographyError=[&](const cv::Mat& h,const cv::Mat& inv,int k) {
        auto distance=[](const cv::Mat& m,cv::Point2f from,cv::Point2f to) {
            const double z=m.at<double>(2,0)*from.x+m.at<double>(2,1)*from.y+m.at<double>(2,2);
            if(std::abs(z)<1e-12)return std::numeric_limits<double>::infinity();
            const double x=(m.at<double>(0,0)*from.x+m.at<double>(0,1)*from.y+m.at<double>(0,2))/z;
            const double y=(m.at<double>(1,0)*from.x+m.at<double>(1,1)*from.y+m.at<double>(1,2))/z;
            return std::hypot(x-to.x,y-to.y);
        };
        int i=ids[k];return std::max(distance(h,points[i],out.diagnostics[i].tracked_right),
                                   distance(inv,out.diagnostics[i].tracked_right,points[i]));
    };
    Eigen::Vector3d best_plane=Eigen::Vector3d::Zero();cv::Mat best_h;
    std::vector<int> best;double best_cost=std::numeric_limits<double>::infinity();
    for(const auto& sample:samples) {
        Eigen::Vector3d plane=Eigen::Vector3d::Zero();cv::Mat h,inv;
        if(horizontal) {
            Eigen::Matrix3d a;Eigen::Vector3d d;
            for(int j=0;j<3;++j){a.row(j)=row(sample[j]).transpose();d[j]=disparity(sample[j]);}
            if(std::abs(a.determinant())<1e-6)continue;
            plane=a.partialPivLu().solve(d);if(!plane.allFinite())continue;
        } else {h=homography(sample,true);if(h.empty())continue;inv=h.inv();}
        std::vector<int> inliers;double cost=0;
        for(int k=0;k<n;++k) {
            const double e=horizontal?std::abs(row(k).dot(plane)-disparity(k)):homographyError(h,inv,k);
            if(e<=cfg.planar_max_error_px)inliers.push_back(k);
            cost+=std::min(e*e,cfg.planar_max_error_px*cfg.planar_max_error_px);
        }
        if(inliers.size()>best.size()||(inliers.size()==best.size()&&cost<best_cost)) {
            best=std::move(inliers);best_cost=cost;best_plane=plane;best_h=h;
        }
    }
    if(static_cast<int>(best.size())<required){done("uncertain_no_consensus");return;}
    // Refit using consensus, then reassess before changing any accepted decision.
    for(int iteration=0;iteration<2;++iteration) {
        if(horizontal) {
            Eigen::Matrix3d a=Eigen::Matrix3d::Zero();Eigen::Vector3d b=Eigen::Vector3d::Zero();
            for(int k:best){const auto r=row(k);a+=r*r.transpose();b+=r*disparity(k);}
            if(std::abs(a.determinant())<1e-9){done("degenerate_distribution");return;}
            best_plane=a.ldlt().solve(b);if(!best_plane.allFinite()){done("invalid_model");return;}
        } else {best_h=homography(best,false);if(best_h.empty()){done("invalid_model");return;}}
        const cv::Mat inv=horizontal?cv::Mat():best_h.inv();
        std::vector<int> updated;
        for(int k=0;k<n;++k) {
            double e=horizontal?std::abs(row(k).dot(best_plane)-disparity(k)):homographyError(best_h,inv,k);
            if(e<=cfg.planar_max_error_px)updated.push_back(k);
        }
        if(static_cast<int>(updated.size())<required){done("uncertain_refit");return;}
        if(updated==best)break;
        best=std::move(updated);
    }
    const cv::Mat inv=horizontal?cv::Mat():best_h.inv();
    for(int k=0;k<n;++k) {
        auto& d=out.diagnostics[ids[k]];
        d.planar_error_px=horizontal?std::abs(row(k).dot(best_plane)-disparity(k)):homographyError(best_h,inv,k);
        if(d.planar_error_px>cfg.planar_max_error_px){d.accepted=false;d.reason="planar_geometry_outlier";++out.planar_rejected;}
    }
    done(horizontal?"disparity_plane":"plane_homography");
}

StereoFlowMatcher::StereoFlowMatcher(const StereoCameraParams& camera,
                                       const StereoFlowConfig& config)
    : camera_(camera), config_(config) {
    validateStereoFlowConfig(config_);
    if (!camera.K.allFinite() || !camera.K_inv.allFinite() ||
        !camera.R_rl.allFinite() || !camera.t_rl.allFinite() || camera.t_rl.norm() < 1e-9)
        throw std::invalid_argument("Stereo flow requires finite, nonzero-baseline camera calibration");
    const Eigen::Matrix3d r = camera.R_rl.transpose();
    const Eigen::Vector3d t = -r * camera.t_rl;
    fundamental_ = camera.K_inv.transpose() * skew(t) * r * camera.K_inv;
    horizontal_ = (camera.R_rl - Eigen::Matrix3d::Identity()).norm() < 1e-6 &&
                  std::abs(camera.t_rl.y()) < 1e-6 && std::abs(camera.t_rl.z()) < 1e-6 &&
                  camera.t_rl.x() > 0;
}

cv::Rect StereoFlowMatcher::searchRegion(const cv::Mat& left,
                                          const std::vector<cv::Point2f>& points,
                                          const std::vector<cv::Point2f>& initial,
                                          const cv::Rect& target_roi) const {
    const cv::Rect full(0,0,left.cols,left.rows);
    if (!config_.roi_pyramid || !horizontal_ || left.empty()) return full;
    double x0=left.cols,y0=left.rows,x1=0,y1=0;
    bool any=false;
    for (const auto& p:points) if (inside(p,left,config_.ncc_patch_size/2)) {
        x0=std::min(x0,static_cast<double>(p.x)); y0=std::min(y0,static_cast<double>(p.y));
        x1=std::max(x1,static_cast<double>(p.x)); y1=std::max(y1,static_cast<double>(p.y)); any=true;
    }
    if (!any) return full;
    const cv::Rect bounded=target_roi & cv::Rect(0,0,left.cols,left.rows);
    if (bounded.area()>0) {
        x0=std::min(x0,static_cast<double>(bounded.x));y0=std::min(y0,static_cast<double>(bounded.y));
        x1=std::max(x1,static_cast<double>(bounded.x)+bounded.width);
        y1=std::max(y1,static_cast<double>(bounded.y)+bounded.height);
    }
    x0-=config_.max_disparity_px; // Union of left target and complete right disparity search.
    for (const auto& p:initial) if (inside(p,left,config_.ncc_patch_size/2)) {
        x0=std::min(x0,static_cast<double>(p.x));y0=std::min(y0,static_cast<double>(p.y));
        x1=std::max(x1,static_cast<double>(p.x));y1=std::max(y1,static_cast<double>(p.y));
    }
    const int alignment=1<<config_.max_level;
    const double pad=(config_.window_size/2.0+2)*alignment+config_.ncc_patch_size/2.0+
                     config_.max_epipolar_error_px+config_.roi_extra_margin_px;
    // Align origins with the full-frame pyramid sampling grid.
    const int bx=static_cast<int>(std::floor(std::max(0.0,x0-pad)/alignment))*alignment;
    const int by=static_cast<int>(std::floor(std::max(0.0,y0-pad)/alignment))*alignment;
    const int ex=static_cast<int>(std::min(static_cast<double>(left.cols),std::ceil((x1+pad+1)/alignment)*alignment));
    const int ey=static_cast<int>(std::min(static_cast<double>(left.rows),std::ceil((y1+pad+1)/alignment)*alignment));
    const cv::Rect region(bx,by,ex-bx,ey-by);
    if (region.width<=0 || region.height<=0 ||
        static_cast<double>(region.area())/(static_cast<double>(left.cols)*left.rows)>=config_.roi_max_area_ratio)
        return full;
    return region;
}

FlowMatchResult StereoFlowMatcher::match(const cv::Mat& left, const cv::Mat& right,
                                          const std::vector<cv::Point2f>& points,
                                          const std::vector<cv::Point2f>& initial,
                                          const cv::Rect& target_roi) const {
    if (left.empty() || right.empty() || left.type()!=CV_8UC1 || right.type()!=CV_8UC1 ||
        left.size()!=right.size() || points.empty() || (!initial.empty() && initial.size()!=points.size()))
        return matchFull(left,right,points,initial);
    const auto start=Clock::now();
    const cv::Rect region=searchRegion(left,points,initial,target_roi);
    if (region==cv::Rect(0,0,left.cols,left.rows)) return matchFull(left,right,points,initial);
    const int bx=region.x,by=region.y;
    const cv::Point2f offset(static_cast<float>(bx),static_cast<float>(by));
    auto local_points=points,local_initial=initial;
    for (auto& p:local_points) p-=offset;
    for (auto& p:local_initial) p-=offset;
    auto local_camera=camera_;
    local_camera.K(0,2)-=bx;local_camera.K(1,2)-=by;
    local_camera.K_inv=local_camera.K.inverse();
    cv::Mat cropped_left=left(region).clone(),cropped_right=right(region).clone();
    const double crop_ms=milliseconds(start);
    auto result=StereoFlowMatcher(local_camera,config_).matchFull(cropped_left,cropped_right,local_points,local_initial);
    const double attempt_ms=milliseconds(start);
    if (!result.success && config_.roi_fallback_full) {
        auto full=matchFull(left,right,points,initial);
        full.roi_failure_reason=result.failure_reason;
        full.full_frame_fallback=true;
        full.pyramid_ms+=result.pyramid_ms;full.lk_ms+=result.lk_ms;
        full.coarse_ms+=result.coarse_ms;full.check_ms+=result.check_ms;
        full.planar_ms+=result.planar_ms;
        full.coarse_attempts+=result.coarse_attempts;
        full.pyramid_builds+=result.pyramid_builds;
        result=std::move(full);
    } else {
        for (auto& p:result.left_points) p+=offset;
        for (auto& p:result.right_points) p+=offset;
        for (auto& d:result.diagnostics) {
            if (d.initial_right!=cv::Point2f(-1,-1)) d.initial_right+=offset;
            if (d.tracked_right!=cv::Point2f(-1,-1)) d.tracked_right+=offset;
        }
    }
    result.roi_used=true;result.search_roi=region;result.crop_ms=crop_ms;result.roi_attempt_ms=attempt_ms;
    result.total_ms=milliseconds(start);
    return result;
}

FlowMatchResult StereoFlowMatcher::matchFull(const cv::Mat& left, const cv::Mat& right,
                                          const std::vector<cv::Point2f>& points,
                                          const std::vector<cv::Point2f>& initial,
                                          const std::vector<cv::Mat>* prepared_left,
                                          const std::vector<cv::Mat>* prepared_right) const {
    const auto start = Clock::now();
    FlowMatchResult out;
    out.diagnostics.resize(points.size());
    auto finish = [&]() { out.total_ms = milliseconds(start); return out; };
    if (left.empty() || right.empty() || left.type() != CV_8UC1 || right.type() != CV_8UC1 ||
        left.size() != right.size()) {
        out.failure_reason = "invalid_stereo_images";
        return finish();
    }
    if (!initial.empty() && initial.size() != points.size()) {
        out.failure_reason = "initial_point_count_mismatch";
        return finish();
    }
    if (points.empty()) { out.failure_reason = "no_left_points"; return finish(); }
    const int margin = config_.ncc_patch_size / 2;
    if (left.cols <= 2 * margin + 2 || left.rows <= 2 * margin + 2) {
        out.failure_reason = "images_too_small";
        return finish();
    }

    auto epipolar = [&](int i, const cv::Point2f& p) {
        const Eigen::Vector3d line = fundamental_ * Eigen::Vector3d(points[i].x, points[i].y, 1);
        const double n = line.head<2>().norm();
        return n > 1e-12 ? std::abs(line.dot(Eigen::Vector3d(p.x, p.y, 1))) / n
                         : std::numeric_limits<double>::infinity();
    };
    auto disparityOk = [&](int i, const cv::Point2f& p) {
        const double d = points[i].x - p.x;
        return !horizontal_ || (d >= config_.min_disparity_px && d <= config_.max_disparity_px);
    };

    // At most one bounded coarse search per original point, including cold starts.
    auto coarse = [&](int i, cv::Point2f& seed) {
        auto& diag = out.diagnostics[i];
        diag.coarse_attempted = true;
        ++out.coarse_attempts;
        const auto t = Clock::now();
        const Eigen::Vector3d line = fundamental_ * Eigen::Vector3d(points[i].x, points[i].y, 1);
        const bool scan_x = std::abs(line.y()) >= std::abs(line.x());
        double lo = margin, hi = (scan_x ? right.cols : right.rows) - margin - 2;
        if (horizontal_) {
            lo = std::max(lo, points[i].x - config_.max_disparity_px);
            hi = std::min(hi, points[i].x - config_.min_disparity_px);
        }
        std::vector<std::pair<double, cv::Point2f>> candidates;
        const double line_norm = line.head<2>().norm();
        const cv::Point2f normal = line_norm > 1e-12
            ? cv::Point2f(line.x() / line_norm, line.y() / line_norm) : cv::Point2f();
        const cv::Point2f tangent(-normal.y, normal.x);
        // Include zero and both tolerance boundaries, with <=1 px spacing for
        // ordinary tolerances. Cap transverse samples at nine, so work remains
        // bounded even with an unusually large configured tolerance.
        const int half_band = config_.coarse_epipolar_band
            ? static_cast<int>(std::min(4.0, std::ceil(config_.max_epipolar_error_px))) : 0;
        const double normal_step = half_band ? config_.max_epipolar_error_px / half_band : 0;
        std::optional<CoarseCorrelation> patch;
        if (config_.coarse_epipolar_band) patch.emplace(left, points[i], config_.ncc_patch_size);
        const int step = std::max(1, static_cast<int>(std::ceil((hi - lo + 1) / config_.max_search_candidates)));
        cv::Mat batched_scores;
        if (config_.coarse_epipolar_band && horizontal_ && hi >= lo && step == 1 &&
            std::abs(normal_step - 1) < 1e-9) {
            const int columns = static_cast<int>(std::floor(hi - lo)) + 1;
            batched_scores = patch->grid(right, cv::Point2f(lo, points[i].y - half_band), columns, 2 * half_band + 1);
        }
        if (line_norm > 1e-12 && hi >= lo) {
            for (double x = lo; x <= hi; x += step) {
                cv::Point2f p;
                if (scan_x) p = cv::Point2f(static_cast<float>(x), static_cast<float>(-(line.x()*x + line.z())/line.y()));
                else p = cv::Point2f(static_cast<float>(-(line.y()*x + line.z())/line.x()), static_cast<float>(x));
                if (!config_.coarse_epipolar_band) {
                    // Preserve original sampling, NCC and uniqueness behavior for A/B.
                    if (inside(p, right, margin) && disparityOk(i, p)) {
                        ++diag.coarse_candidates;
                        candidates.emplace_back(correlation(left, right, points[i], p, config_.ncc_patch_size), p);
                    }
                    continue;
                }
                std::pair<double, cv::Point2f> best{-1, p};
                bool sampled = false;
                for (int row = -half_band; row <= half_band; ++row) {
                    const cv::Point2f q = p + normal * static_cast<float>(row * normal_step);
                    if (!inside(q, right, margin) || !disparityOk(i, q)) continue;
                    ++diag.coarse_candidates;
                    const double score = batched_scores.empty() ? patch->score(right, q)
                        : std::clamp(static_cast<double>(batched_scores.at<float>(
                              half_band + (normal.y > 0 ? row : -row), static_cast<int>(std::round(x - lo)))), -1.0, 1.0);
                    if (!sampled || score > best.first) best = {score, q};
                    sampled = true;
                }
                // Keep one peak per station; nearby normal offsets belong to
                // the same correspondence, not separate uniqueness competitors.
                if (sampled) candidates.push_back(best);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        if (config_.coarse_epipolar_band && step > 1 && !candidates.empty()) {
            // Large/general-image searches may have >1 px longitudinal spacing.
            // Refine up to four distinct coarse peaks before the NCC gate;
            // otherwise a true peak between samples can be rejected prematurely.
            std::vector<cv::Point2f> peaks;
            for (const auto& candidate : candidates) {
                bool separate = true;
                for (auto peak : peaks)
                    if (std::abs((candidate.second - peak).dot(tangent)) < config_.ncc_patch_size / 2.0)
                        separate = false;
                if (separate) peaks.push_back(candidate.second);
                if (peaks.size() == 4) break;
            }
            const cv::Point2f direction = scan_x
                ? cv::Point2f(1, -line.x()/line.y()) : cv::Point2f(-line.y()/line.x(), 1);
            const int half_samples = std::min(8, 2 * step);
            for (auto peak : peaks) {
                // Project the peak back onto the ideal line before sampling offsets.
                const double distance = line.dot(Eigen::Vector3d(peak.x, peak.y, 1)) / line_norm;
                const cv::Point2f base = peak - normal * static_cast<float>(distance);
                for (int column = -half_samples; column <= half_samples; ++column) {
                    const auto center = base + direction * static_cast<float>(column * static_cast<double>(step) / half_samples);
                    std::pair<double, cv::Point2f> best{-1, center};
                    bool sampled = false;
                    for (int row = -half_band; row <= half_band; ++row) {
                        const auto q = center + normal * static_cast<float>(row * normal_step);
                        if (!inside(q, right, margin) || !disparityOk(i, q)) continue;
                        ++diag.coarse_candidates;
                        const double score = patch->score(right, q);
                        if (!sampled || score > best.first) best = {score, q};
                        sampled = true;
                    }
                    if (sampled) candidates.push_back(best);
                }
            }
            std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        }
        bool ok = !candidates.empty() && candidates.front().first >= config_.min_ncc;
        if (!candidates.empty()) diag.coarse_best_ncc = candidates.front().first;
        if (!ok && config_.coarse_epipolar_band) diag.reason = "coarse_low_ncc_or_no_candidate";
        if (ok) {
            const auto& best = candidates.front();
            for (size_t j = 1; j < candidates.size(); ++j) {
                const cv::Point2f delta = candidates[j].second - best.second;
                const double separation = config_.coarse_epipolar_band
                    ? std::abs(delta.dot(tangent)) : cv::norm(delta);
                if (separation < config_.ncc_patch_size / 2.0) continue;
                diag.coarse_second_ncc = candidates[j].first;
                if (best.first - candidates[j].first < config_.uniqueness_margin) {
                    ok = false;
                    if (config_.coarse_epipolar_band) diag.reason = "coarse_nonunique_peak";
                }
                break;
            }
            if (ok) seed = best.second;
        }
        if (!ok && !config_.coarse_epipolar_band) diag.reason = "coarse_ambiguous_or_no_match";
        out.coarse_ms += milliseconds(t);
        return ok;
    };

    std::vector<cv::Mat> owned_left, owned_right;
    const auto& pyramid_left=prepared_left ? *prepared_left : owned_left;
    const auto& pyramid_right=prepared_right ? *prepared_right : owned_right;
    const auto pyramid_start = Clock::now();
    const cv::Size win(config_.window_size, config_.window_size);
    if (!prepared_left || !prepared_right) {
        cv::buildOpticalFlowPyramid(left, owned_left, win, config_.max_level);
        cv::buildOpticalFlowPyramid(right, owned_right, win, config_.max_level);
        out.pyramid_builds=1;
    }
    out.pyramid_ms = milliseconds(pyramid_start);

    auto track = [&](const std::vector<int>& ids, const std::vector<cv::Point2f>& seeds) {
        if (ids.empty()) return;
        const auto t = Clock::now();
        std::vector<cv::Point2f> pl, pr = seeds;
        for (int i : ids) pl.push_back(points[i]);
        std::vector<uchar> status, back_status;
        std::vector<float> error, back_error;
        const cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                        config_.max_iterations, config_.epsilon);
        cv::calcOpticalFlowPyrLK(pyramid_left, pyramid_right, pl, pr, status, error,
            win, config_.max_level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW, config_.min_eigenvalue);
        std::vector<int> back_ids;
        std::vector<cv::Point2f> back_from, back_to;
        for (size_t k = 0; k < ids.size(); ++k) {
            auto& d = out.diagnostics[ids[k]];
            d.initial_right = seeds[k]; d.tracked_right = pr[k]; d.accepted = false;
            if (status[k] && inside(pr[k], right, margin)) {
                back_ids.push_back(static_cast<int>(k));
                back_from.push_back(pr[k]); back_to.push_back(pl[k]);
            } else d.reason = "forward_lk_or_border";
        }
        if (!back_ids.empty())
            cv::calcOpticalFlowPyrLK(pyramid_right, pyramid_left, back_from, back_to, back_status, back_error,
                win, config_.max_level, criteria, cv::OPTFLOW_USE_INITIAL_FLOW, config_.min_eigenvalue);
        out.lk_ms += milliseconds(t);
        const auto check_start = Clock::now();
        for (size_t j = 0; j < back_ids.size(); ++j) {
            const int k = back_ids[j], i = ids[k];
            auto& d = out.diagnostics[i];
            d.fb_error = finite(back_to[j]) ? cv::norm(back_to[j] - points[i]) : std::numeric_limits<double>::infinity();
            d.epipolar_error = epipolar(i, pr[k]);
            d.ncc = correlation(left, right, points[i], pr[k], config_.ncc_patch_size);
            if (!back_status[j] || d.fb_error > config_.max_fb_error_px) d.reason = "backward_or_fb_error";
            else if (d.epipolar_error > config_.max_epipolar_error_px) d.reason = "epipolar_error";
            else if (!disparityOk(i, pr[k])) d.reason = "disparity_range";
            else if (d.ncc < config_.min_ncc) d.reason = "photometric_error";
            else { d.accepted = true; d.reason = "accepted"; }
        }
        out.check_ms += milliseconds(check_start);
    };

    std::vector<int> ids;
    std::vector<cv::Point2f> seeds;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (!inside(points[i], left, margin)) { out.diagnostics[i].reason = "left_point_or_border"; continue; }
        cv::Point2f seed;
        if (!initial.empty() && inside(initial[i], right, margin)) seed = initial[i];
        else if (!config_.coarse_retry || !coarse(i, seed)) continue;
        ids.push_back(i); seeds.push_back(seed);
    }
    track(ids, seeds);
    ids.clear(); seeds.clear();
    if (config_.coarse_retry) {
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            auto& d = out.diagnostics[i];
            if (d.accepted || d.coarse_attempted || !inside(points[i], left, margin)) continue;
            cv::Point2f seed;
            if (coarse(i, seed)) { ids.push_back(i); seeds.push_back(seed); }
        }
        track(ids, seeds);
    }

    const auto check_start = Clock::now();
    std::vector<int> ranking;
    for (int i = 0; i < static_cast<int>(points.size()); ++i)
        if (out.diagnostics[i].accepted) ranking.push_back(i);
    std::sort(ranking.begin(), ranking.end(), [&](int a, int b) {
        return out.diagnostics[a].fb_error + 1 - out.diagnostics[a].ncc <
               out.diagnostics[b].fb_error + 1 - out.diagnostics[b].ncc;
    });
    std::vector<int> unique;
    for (int i : ranking) {
        bool duplicate = false;
        for (int j : unique)
            if (cv::norm(out.diagnostics[i].tracked_right - out.diagnostics[j].tracked_right) < config_.duplicate_distance_px)
                duplicate = true;
        if (duplicate) { out.diagnostics[i].accepted = false; out.diagnostics[i].reason = "duplicate_right_point"; }
        else unique.push_back(i);
    }
    std::vector<cv::Point2f> valid_left;
    filterPlanarGeometry(points,config_,horizontal_,out);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (inside(points[i], left, margin)) valid_left.push_back(points[i]);
        if (!out.diagnostics[i].accepted) continue;
        out.left_indices.push_back(i);
        out.left_points.push_back(points[i]); out.right_points.push_back(out.diagnostics[i].tracked_right);
    }
    const double all_area = hullArea(valid_left);
    int negative_pairs = 0;
    if (horizontal_) for (size_t i=0; i<points.size(); ++i) {
        const auto& d = out.diagnostics[i];
        if (inside(d.tracked_right, right, margin) && d.fb_error <= config_.max_fb_error_px &&
            d.epipolar_error <= config_.max_epipolar_error_px && d.ncc >= config_.min_ncc &&
            points[i].x - d.tracked_right.x <= -config_.min_disparity_px) ++negative_pairs;
    }
    if (out.left_points.size() < static_cast<size_t>(config_.min_matches))
        out.failure_reason = negative_pairs >= config_.min_matches
            ? "negative_disparity_check_input_order" : "insufficient_stereo_pairs";
    else if (static_cast<double>(out.left_points.size()) / points.size() < config_.min_valid_ratio)
        out.failure_reason = "low_valid_ratio";
    else if (all_area <= 1e-6 || hullArea(out.left_points) / all_area < config_.min_coverage_ratio)
        out.failure_reason = "poor_spatial_coverage";
    else out.success = true;
    out.check_ms += milliseconds(check_start);
    return finish();
}

StereoFlowFrame::StereoFlowFrame(const StereoCameraParams& camera, const StereoFlowConfig& config,
                               const cv::Mat& left, const cv::Mat& right, const cv::Rect& target_union)
    : config_(config),left_(left),right_(right),full_matcher_(std::make_unique<StereoFlowMatcher>(camera,config)) {
    const auto start=Clock::now();
    const cv::Rect full(0,0,left.cols,left.rows);
    const auto bounded=target_union & full;
    region_=full;
    if (!left.empty() && !right.empty() && left.type()==CV_8UC1 && right.type()==CV_8UC1 &&
        left.size()==right.size() && bounded.area()>0) {
        const cv::Point2f center(bounded.x+bounded.width/2.f,bounded.y+bounded.height/2.f);
        region_=full_matcher_->searchRegion(left,{center},{},bounded);
        if (region_!=full) {
            cropped_left_=left(region_).clone();cropped_right_=right(region_).clone();
            auto local_camera=camera;
            local_camera.K(0,2)-=region_.x;local_camera.K(1,2)-=region_.y;
            local_camera.K_inv=local_camera.K.inverse();
            local_matcher_=std::make_unique<StereoFlowMatcher>(local_camera,config);
        }
    }
    pending_crop_ms_=milliseconds(start);
}

FlowMatchResult StereoFlowFrame::run(bool full, const std::vector<cv::Point2f>& points,
                                   const std::vector<cv::Point2f>& initial, int min_matches) {
    const auto start=Clock::now();
    const auto& l=full ? left_ : cropped_left_;
    const auto& r=full ? right_ : cropped_right_;
    auto& lp=full ? full_left_pyramid_ : local_left_pyramid_;
    auto& rp=full ? full_right_pyramid_ : local_right_pyramid_;
    auto& matcher=full ? full_matcher_ : local_matcher_;
    // Only the frame-level point-count policy differs for four-corner TT.
    // Copying the matcher leaves shared pyramid parameters and quality gates intact.
    auto attempt_matcher=*matcher;
    if(min_matches>0)attempt_matcher.config_.min_matches=min_matches;
    // Let the ordinary matcher report invalid inputs before building a pyramid.
    if (l.empty() || r.empty() || l.type()!=CV_8UC1 || r.type()!=CV_8UC1 || l.size()!=r.size() ||
        points.empty() || (!initial.empty() && points.size()!=initial.size()))
        return attempt_matcher.matchFull(l,r,points,initial);
    bool built=lp.empty();
    if (built) {
        const cv::Size win(config_.window_size,config_.window_size);
        cv::buildOpticalFlowPyramid(l,lp,win,config_.max_level);
        cv::buildOpticalFlowPyramid(r,rp,win,config_.max_level);
    }
    const double build_ms=milliseconds(start);
    auto result=attempt_matcher.matchFull(l,r,points,initial,&lp,&rp);
    result.pyramid_ms+=built ? build_ms : 0;
    result.pyramid_builds=built ? 1 : 0;
    result.total_ms=milliseconds(start);
    return result;
}

FlowMatchResult StereoFlowFrame::match(const std::vector<cv::Point2f>& points,
                                     const std::vector<cv::Point2f>& initial, int min_matches) {
    const auto start=Clock::now();
    const double crop_ms=pending_crop_ms_;pending_crop_ms_=0;
    if (!local_matcher_) {
        auto result=run(true,points,initial,min_matches);
        result.total_ms+=crop_ms;
        return result;
    }
    const cv::Point2f offset(region_.x,region_.y);
    auto local_points=points,local_initial=initial;
    for(auto& p:local_points)p-=offset;
    for(auto& p:local_initial)p-=offset;
    auto result=run(false,local_points,local_initial,min_matches);
    result.roi_attempt_ms=milliseconds(start)+crop_ms;
    if (!result.success && config_.roi_fallback_full) {
        auto fallback=run(true,points,initial,min_matches);
        fallback.roi_failure_reason=result.failure_reason;
        fallback.full_frame_fallback=true;
        fallback.roi_attempt_ms=result.roi_attempt_ms;
        fallback.pyramid_ms+=result.pyramid_ms;fallback.lk_ms+=result.lk_ms;
        fallback.coarse_ms+=result.coarse_ms;fallback.check_ms+=result.check_ms;
        fallback.planar_ms+=result.planar_ms;
        fallback.coarse_attempts+=result.coarse_attempts;fallback.pyramid_builds+=result.pyramid_builds;
        result=std::move(fallback);
    } else {
        for(auto& p:result.left_points)p+=offset;
        for(auto& p:result.right_points)p+=offset;
        for(auto& d:result.diagnostics) {
            if(d.initial_right!=cv::Point2f(-1,-1))d.initial_right+=offset;
            if(d.tracked_right!=cv::Point2f(-1,-1))d.tracked_right+=offset;
        }
    }
    result.roi_used=true;result.search_roi=region_;result.crop_ms=crop_ms;
    result.total_ms=milliseconds(start)+crop_ms;
    return result;
}

FlowPoseCheck checkFlowPose(const PoseEstimate& pose, const PipelineResult& result,
                            const std::vector<Eigen::Vector3d>& model,
                            const StereoCameraParams& camera, bool stereo,
                            double mean_limit, double point_limit) {
    FlowPoseCheck out;
    if (!pose.success || !pose.R.allFinite() || !pose.t.allFinite() ||
        (pose.R.transpose()*pose.R - Eigen::Matrix3d::Identity()).norm() > 1e-3 ||
        std::abs(pose.R.determinant() - 1) > 1e-3) { out.reason = "invalid_pose"; return out; }
    std::vector<int> lut(result.pts_left_match.size(), -1);
    for (const auto& m : result.good_matches)
        if (m.queryIdx >= 0 && m.queryIdx < static_cast<int>(lut.size())) lut[m.queryIdx] = m.trainIdx;
    const size_t count = stereo ? result.pts_left_good.size() : result.good_matches.size();
    if (!count || (stereo && (result.pts_right_good.size() != count || result.idx_from_filtered.size() != count))) {
        out.reason = "invalid_pose_correspondences"; return out;
    }
    for (size_t k = 0; k < count; ++k) {
        const int i = stereo ? result.idx_from_filtered[k] : result.good_matches[k].queryIdx;
        if (i < 0 || i >= static_cast<int>(lut.size()) || lut[i] < 0 || lut[i] >= static_cast<int>(model.size())) {
            out.reason = "invalid_model_index"; return out;
        }
        const auto& pl = stereo ? result.pts_left_good[k] : result.pts_left_match[i];
        const Eigen::Vector3d pc = pose.R * model[lut[i]] + pose.t;
        const Eigen::Vector3d pr = camera.R_rl.transpose() * (pc - camera.t_rl);
        if (!pc.allFinite() || pc.z() <= 1e-6 || (stereo && (!pr.allFinite() || pr.z() <= 1e-6))) {
            out.reason = "nonpositive_point_depth"; return out;
        }
        const Eigen::Vector3d uv = camera.K * pc;
        const double el = std::hypot(uv.x()/uv.z() - pl.x, uv.y()/uv.z() - pl.y);
        double er = 0;
        if (stereo) {
            const Eigen::Vector3d ur = camera.K * pr;
            const auto& observed = result.pts_right_good[k];
            er = std::hypot(ur.x()/ur.z() - observed.x, ur.y()/ur.z() - observed.y);
        }
        if (!std::isfinite(el) || !std::isfinite(er)) { out.reason = "nonfinite_reprojection"; return out; }
        out.mean_left_px += el; out.mean_right_px += er;
        out.max_error_px = std::max({out.max_error_px, el, er});
    }
    out.mean_left_px /= count; out.mean_right_px /= count;
    out.success = out.mean_left_px <= mean_limit && out.mean_right_px <= mean_limit && out.max_error_px <= point_limit;
    if (!out.success) out.reason = "reprojection_error";
    return out;
}
} // namespace gpnp
