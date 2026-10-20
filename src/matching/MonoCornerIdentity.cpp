#include "matching/MonoCornerIdentity.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gpnp {
void validateMonoCornerIdentityConfig(const MonoCornerIdentityConfig& c) {
    if(c.frontend!="legacy" && c.frontend!="normalized")
        throw std::invalid_argument("mono_identity.frontend must be legacy/normalized");
    if (c.method!="hybrid" && c.method!="projective" && c.method!="tt_fourway")
        throw std::invalid_argument("mono_identity.method must be hybrid/projective/tt_fourway");
    if (c.normalized_size<32 || c.normalized_size>256)
        throw std::invalid_argument("mono_identity.normalized_size must be 32..256");
    for(double v:{c.min_span_px,c.min_roi_span_ratio,c.min_iou,c.min_score_gap,c.max_fit_error_px,c.max_fit_error_ratio})
        if(!std::isfinite(v)) throw std::invalid_argument("mono_identity thresholds must be finite");
    if(c.min_span_px<4 || c.min_roi_span_ratio<0 || c.min_roi_span_ratio>1 || c.min_iou<=0 || c.min_iou>1 || c.min_score_gap<0 || c.min_score_gap>1 ||
       c.max_fit_error_px<=0 || c.max_fit_error_ratio<0 || c.max_fit_error_ratio>0.1)
        throw std::invalid_argument("mono_identity thresholds out of range");
    if (c.min_reliable_points < 4 || c.min_reliable_points > 10 ||
        c.max_partial_hypotheses < 1 || c.max_partial_hypotheses > 64 ||
        !std::isfinite(c.min_model_coverage) || c.min_model_coverage <= 0 ||
        c.min_model_coverage > 1)
      throw std::invalid_argument(
          "mono_identity partial/quality thresholds out of range");
}
MonoCornerIdentityConfig readMonoCornerIdentityConfig(const cv::FileNode& node) {
    MonoCornerIdentityConfig c;
    if(node.empty()) return c;
    if(!node.isMap()) throw std::invalid_argument("mono_identity must be an object");
    if(!node["enabled"].empty()) {
        auto v=node["enabled"];
        if(!v.isInt() || ((int)v!=0 && (int)v!=1))
            throw std::invalid_argument("mono_identity.enabled must be boolean");
        c.enabled=(int)v!=0;
    }
    if(!node["class1_enabled"].empty()) {
        auto v=node["class1_enabled"];
        if(!v.isInt() || ((int)v!=0 && (int)v!=1)) throw std::invalid_argument("mono_identity.class1_enabled must be boolean");
        c.class1_enabled=(int)v!=0;
    }
    if(!node["frontend"].empty()) {
        if(!node["frontend"].isString()) throw std::invalid_argument("mono_identity.frontend must be string");
        c.frontend=(std::string)node["frontend"];
    }
    if(!node["refine_original"].empty()) {
        auto v=node["refine_original"];
        if(!v.isInt() || ((int)v!=0 && (int)v!=1)) throw std::invalid_argument("mono_identity.refine_original must be boolean");
        c.refine_original=(int)v!=0;
    }
    if(!node["method"].empty()) {
        if(!node["method"].isString()) throw std::invalid_argument("mono_identity.method must be string");
        c.method=(std::string)node["method"];
    }
    if(!node["normalized_size"].empty()) {
        if(!node["normalized_size"].isInt()) throw std::invalid_argument("normalized_size must be integer");
        c.normalized_size=(int)node["normalized_size"];
    }
    auto number=[&](const char* key,double& target) {
        auto v=node[key]; if(v.empty()) return;
        if(!v.isInt() && !v.isReal()) throw std::invalid_argument(std::string(key)+" must be numeric");
        target=(double)v;
    };
    number("min_span_px",c.min_span_px); number("min_iou",c.min_iou);
    number("min_roi_span_ratio",c.min_roi_span_ratio);
    number("min_score_gap",c.min_score_gap); number("max_fit_error_px",c.max_fit_error_px);
    number("max_fit_error_ratio",c.max_fit_error_ratio);
    number("min_model_coverage", c.min_model_coverage);
    for (auto item :
         {std::pair<const char *, bool *>{"quality_filter", &c.quality_filter},
          {"partial_matching", &c.partial_matching}}) {
      auto v = node[item.first];
      if (v.empty())
        continue;
      if (!v.isInt() || ((int)v != 0 && (int)v != 1))
        throw std::invalid_argument("mono_identity boolean expected");
      *item.second = (int)v != 0;
    }
    for (auto item : {std::pair<const char *, int *>{"min_reliable_points",
                                                     &c.min_reliable_points},
                      {"max_partial_hypotheses", &c.max_partial_hypotheses}}) {
      auto v = node[item.first];
      if (v.empty())
        continue;
      if (!v.isInt())
        throw std::invalid_argument("mono_identity integer expected");
      *item.second = (int)v;
    }
    validateMonoCornerIdentityConfig(c); return c;
}

namespace {
bool finite(const cv::Point2f& p) { return std::isfinite(p.x) && std::isfinite(p.y); }
double squaredNorm(const cv::Point2f& p) { return p.dot(p); }
double span(const std::vector<cv::Point2f>& p) {
    float x0=p[0].x,x1=x0,y0=p[0].y,y1=y0;
    for(const auto& q:p) { x0=std::min(x0,q.x); x1=std::max(x1,q.x); y0=std::min(y0,q.y); y1=std::max(y1,q.y); }
    return std::max(x1-x0,y1-y0);
}
double overlap(const cv::Mat& a,const cv::Mat& b) {
    cv::Mat both,either; cv::bitwise_and(a,b,both); cv::bitwise_or(a,b,either);
    const int count=cv::countNonZero(either);
    return count ? (double)cv::countNonZero(both)/count : 0.;
}
std::vector<int> cycle(int n,int start,int direction) {
    std::vector<int> indices(n);
    for(int i=0;i<n;++i) indices[i]=(start+direction*i+n)%n;
    return indices;
}
struct Candidate {
    bool valid{false}; std::vector<int> ids; cv::Mat inverse,normalized;
    std::vector<int> model_ids, rejected_ids;
    std::vector<cv::Point2f> rejected_points;
    std::vector<double> residuals;
    double fit{0},iou{0},score{1e9};
};

// Fit only the middle of an observed edge: rounded/morphologically altered
// vertices must not pull the supporting line away from the actual boundary.
bool fitSupportingLine(const std::vector<cv::Point2f>& samples,
                       double length, cv::Vec4f& line) {
    if(samples.size()<4) return false;
    cv::fitLine(samples,line,cv::DIST_HUBER,0,.01,.01);
    double sum=0;
    for(auto p:samples) {
        double d=(p.x-line[2])*line[1]-(p.y-line[3])*line[0];sum+=d*d;
    }
    return std::sqrt(sum/samples.size())<=std::max(.8,.005*length);
}

std::vector<bool> supportingLines(const std::vector<cv::Point>& contour,
        const std::vector<cv::Point2f>& vertices, std::vector<cv::Vec4f>& lines) {
    struct Edge {
        cv::Point2f origin, delta;
        double length, squared_length, band;
        std::vector<cv::Point2f> samples;
    };
    const size_t count=vertices.size();
    std::vector<Edge> edges;
    edges.reserve(count);
    for(size_t i=0;i<count;++i) {
        const auto delta=vertices[(i+1)%count]-vertices[i];
        const double length=cv::norm(delta);
        edges.push_back({vertices[i],delta,length,length*length,
                         std::max(2.,.025*length),{}});
    }
    // Preserve each edge's original point order and allow shared support.
    // This fuses contour reads; geometric membership is still O(points*edges).
    for(auto p:contour) for(auto& edge:edges) {
        if(edge.length<6) continue;
        const cv::Point2f delta=cv::Point2f(p)-edge.origin;
        const double t=delta.dot(edge.delta)/edge.squared_length;
        const double d=std::abs(delta.x*edge.delta.y-delta.y*edge.delta.x)/edge.length;
        if(t>=.15 && t<=.85 && d<=edge.band) edge.samples.emplace_back(p);
    }
    std::vector<bool> valid(count,false);
    lines.resize(count);
    for(size_t i=0;i<count;++i)
        if(edges[i].length>=6)
            valid[i]=fitSupportingLine(edges[i].samples,edges[i].length,lines[i]);
    return valid;
}

// Owned by one extraction call only. A changed mask requires a new context.
struct ContourGeometry {
    std::vector<cv::Point> contour;
    cv::Rect bounds;
    cv::RotatedRect rectangle;
    bool rectangle_ready{false};
    void extract(const cv::Mat& mask) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if(contours.empty()) return;
        auto largest=std::max_element(contours.begin(),contours.end(),
            [](const auto& a,const auto& b){return cv::contourArea(a)<cv::contourArea(b);});
        contour=std::move(*largest);
        bounds=cv::boundingRect(contour);
    }
    const cv::RotatedRect& minRectangle() {
        if(!rectangle_ready) {
            rectangle=cv::minAreaRect(contour);
            rectangle_ready=true;
        }
        return rectangle;
    }
    std::vector<cv::Point> simpleContour() const {
        // Compress a CHAIN_APPROX_NONE chain without rescanning the image.
        if(contour.size()<3) return contour;
        std::vector<cv::Point> simple;
        for(size_t i=0;i<contour.size();++i) {
            const auto previous=contour[(i+contour.size()-1)%contour.size()];
            const auto next=contour[(i+1)%contour.size()];
            if(contour[i]-previous != next-contour[i]) simple.push_back(contour[i]);
        }
        return simple;
    }
};

double cross(cv::Point2f a,cv::Point2f b) {return a.x*b.y-a.y*b.x;}
bool coversModel(const std::vector<cv::Point2f> &subset,
                 const std::vector<cv::Point2f> &model, double ratio) {
  if (subset.size() < 4)
    return false;
  std::vector<cv::Point2f> full_hull, part_hull;
  cv::convexHull(model, full_hull);
  cv::convexHull(subset, part_hull);
  auto a = cv::boundingRect(model), b = cv::boundingRect(subset);
  return cv::contourArea(part_hull) >= ratio * cv::contourArea(full_hull) &&
         b.width >= ratio * a.width && b.height >= ratio * a.height;
}
} // namespace

MonoCornerIdentity::MonoCornerIdentity(const MonoCornerIdentityConfig& c):config_(c) {
    validateMonoCornerIdentityConfig(c);
}
void MonoCornerIdentity::setModel(const std::vector<cv::Point2f>& canonical) {
    model_.clear(); reference_.release();
    if(canonical.size()<4) return;
    for(const auto& p:canonical) if(!finite(p)) return;
    auto box=cv::boundingRect(canonical);
    if(box.width<2 || box.height<2 || std::abs(cv::contourArea(canonical))<1e-3) return;
    float x0=canonical[0].x,x1=x0,y0=canonical[0].y,y1=y0;
    for(auto p:canonical) { x0=std::min(x0,p.x);x1=std::max(x1,p.x);y0=std::min(y0,p.y);y1=std::max(y1,p.y); }
    const float margin=config_.normalized_size/16.f;
    const float extent=config_.normalized_size-1-2*margin;
    if(x1-x0<1e-6 || y1-y0<1e-6) return;
    for(auto p:canonical) model_.emplace_back(margin+(p.x-x0)*extent/(x1-x0),margin+(p.y-y0)*extent/(y1-y0));
    reference_=cv::Mat::zeros(config_.normalized_size,config_.normalized_size,CV_8U);
    std::vector<cv::Point> polygon;
    for(auto p:model_) polygon.emplace_back(cvRound(p.x),cvRound(p.y));
    cv::fillPoly(reference_,std::vector<std::vector<cv::Point>>{polygon},cv::Scalar(255));
}

MonoCornerIdentityResult MonoCornerIdentity::extractNormalized(const cv::Mat& mask,bool capture) const {
    const auto begin=std::chrono::steady_clock::now();
    auto milliseconds=[](auto start) {return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();};
    MonoCornerIdentityResult out;out.status="invalid_input";
    ContourGeometry geometry;
    auto finish=[&]() {out.elapsed_ms=milliseconds(begin);if(capture) {out.reference=reference_;out.foreground=mask;out.source_contour=geometry.simpleContour();}return out;};
    if(model_.size()<4 || mask.empty() || mask.type()!=CV_8UC1) return finish();
    geometry.extract(mask);
    const auto& contour=geometry.contour;
    if(contour.empty()) {out.status="empty_contour";return finish();}
    const auto bounds=geometry.bounds;
    if(std::max(bounds.width,bounds.height)<config_.min_span_px) {out.status="below_resolvable_span";return finish();}
    if(bounds.width<mask.cols*config_.min_roi_span_ratio || bounds.height<mask.rows*config_.min_roi_span_ratio) {
        out.status="component_too_small_for_roi";return finish();
    }
    const auto& rect=geometry.minRectangle();cv::Point2f raw[4];rect.points(raw);
    if(rect.size.width<2 || rect.size.height<2) {out.status="degenerate_contour";return finish();}
    std::vector<cv::Point2f> quad(raw,raw+4);
    const float m=config_.normalized_size/16.f,e=config_.normalized_size-1-m;
    std::vector<cv::Point2f> square{{m,m},{e,m},{e,e},{m,e}};
    if(cv::contourArea(quad,true)*cv::contourArea(square,true)<0) std::reverse(quad.begin(),quad.end());
    cv::Mat initial=cv::getPerspectiveTransform(quad,square),normalized;
    cv::warpPerspective(mask,normalized,initial,reference_.size(),cv::INTER_NEAREST,cv::BORDER_CONSTANT,0);
    // Rotate the small image, not the original ROI; just one original-image warp.
    std::vector<cv::Mat> images(4),transforms(4);images[0]=normalized;transforms[0]=initial;
    const double edge=config_.normalized_size-1;
    cv::Mat rotation=(cv::Mat_<double>(3,3)<<0,-1,edge,1,0,0,0,0,1);
    std::vector<std::pair<double,int>> scores;
    for(int k=0;k<4;++k) {
        if(k) {cv::rotate(images[k-1],images[k],cv::ROTATE_90_CLOCKWISE);transforms[k]=rotation*transforms[k-1];}
        scores.emplace_back(overlap(images[k],reference_),k);
    }
    std::sort(scores.begin(),scores.end(),[](const auto& a,const auto& b){return a.first>b.first;});
    const int best=scores[0].second;
    out.coarse_iou=scores[0].first;out.normalized_tests=4;
    if(capture) out.coarse_normalized=images[best];
    out.normalization_ms=milliseconds(begin);

    auto corner_start=std::chrono::steady_clock::now();
    // Normalize the actual dense contour as floating point coordinates. The
    // raster image is for shape matching only: re-extracting its pixel contour
    // would quantize a large ROI down to 128px and can erase a real corner.
    std::vector<cv::Point2f> original_contour;original_contour.reserve(contour.size());
    for(auto p:contour) original_contour.emplace_back(p);
    std::vector<cv::Point2f> nc;cv::perspectiveTransform(original_contour,nc,transforms[best]);
    std::vector<cv::Point2f> polygon;
    std::vector<cv::Point2f> partial_polygon;
    auto remember = [&]() {
      const int count = (int)polygon.size();
      if (count >= config_.min_reliable_points && count <= 12 &&
          count != (int)model_.size() &&
          (partial_polygon.empty() ||
           std::abs(count - (int)model_.size()) <
               std::abs((int)partial_polygon.size() - (int)model_.size())))
        partial_polygon = polygon;
    };
    const double perimeter=cv::arcLength(nc,true);
    // Exact contour-supported vertices only. No insertion of synthetic points
    // or deletion to force a ten-point output when the structure is missing.
    double previous_epsilon=.0001;
    for(int step=0;step<=80;++step) {
        const double epsilon=.001+.0005*step;
        cv::approxPolyDP(nc,polygon,perimeter*epsilon,true);
        remember();
        if(polygon.size()==model_.size()) break;
        if(polygon.size()<model_.size()) {
            double low=previous_epsilon,high=epsilon;
            for(int iteration=0;iteration<24;++iteration) {
                double mid=(low+high)/2;
                cv::approxPolyDP(nc,polygon,perimeter*mid,true);
                remember();
                if(polygon.size()==model_.size()) break;
                if(polygon.size()>model_.size())low=mid;else high=mid;
            }
            break;
        }
        previous_epsilon=epsilon;
    }
    if (polygon.size() != model_.size() && config_.partial_matching &&
        !partial_polygon.empty())
      polygon = partial_polygon;
    if (polygon.size() != model_.size() &&
        (!config_.partial_matching ||
         polygon.size() < (size_t)config_.min_reliable_points ||
         polygon.size() > 12)) {
      out.status = "normalized_corner_count";
      out.corners_ms = milliseconds(corner_start);
      return finish();
    }
    std::vector<cv::Point2f> points;
    cv::perspectiveTransform(polygon,points,transforms[best].inv());
    out.corners_ms=milliseconds(corner_start);
    const auto before=points;
    auto refine_start=std::chrono::steady_clock::now();
    int refined=0;
    if(config_.refine_original) {
        const int n=(int)points.size();std::vector<cv::Vec4f> lines;
        const auto valid=supportingLines(contour,before,lines);
        for(int i=0;i<n;++i) {
            const int prev=(i+n-1)%n,next=(i+1)%n;
            if(!valid[prev] || !valid[i]) continue;
            cv::Point2f a(lines[prev][2],lines[prev][3]),v(lines[prev][0],lines[prev][1]);
            cv::Point2f b(lines[i][2],lines[i][3]),w(lines[i][0],lines[i][1]);
            const double determinant=cross(v,w);
            if(std::abs(determinant)<.2) continue;
            cv::Point2f intersection=a+v*(float)(cross(b-a,w)/determinant);
            if(!finite(intersection) || intersection.x<0 || intersection.y<0 || intersection.x>=mask.cols || intersection.y>=mask.rows ||
               cv::norm(intersection-before[i])>std::max(2.,.025*span(before))) continue;
            if(cross(before[i]-before[prev],before[next]-before[i])*
               cross(intersection-before[prev],before[next]-intersection)<=0) continue;
            points[i]=intersection;++refined;
        }
        if(cv::contourArea(points,true)*cv::contourArea(before,true)<=0) {points=before;refined=0;}
    }
    out.refinement_ms=milliseconds(refine_start);
    const double normalization_ms=out.normalization_ms,corners_ms=out.corners_ms,refinement_ms=out.refinement_ms;
    auto validation_start=std::chrono::steady_clock::now();
    out=matchImpl(points,mask,capture,transforms[best],images[best],scores[0].first,scores[0].first-scores[1].first);
    // A ten-vertex approximation can still contain a spurious vertex. Retry a
    // neighbouring contour scale only after the fast exact path has failed.
    if (!out.accepted && config_.partial_matching && !partial_polygon.empty() &&
        partial_polygon != polygon) {
      std::vector<cv::Point2f> alternative;
      cv::perspectiveTransform(partial_polygon, alternative,
                               transforms[best].inv());
      auto retry =
          matchImpl(alternative, mask, capture, transforms[best], images[best],
                    scores[0].first, scores[0].first - scores[1].first);
      if (retry.accepted) {
        points = std::move(alternative);
        out = std::move(retry);
        refined = 0;
      }
    }
    out.normalization_ms=normalization_ms;out.corners_ms=corners_ms;out.refinement_ms=refinement_ms;
    out.validation_ms=milliseconds(validation_start);out.normalized_tests+=4;out.refined_points=refined;
    if(capture) {out.coarse_normalized=images[best];out.foreground=mask;}
    if(out.accepted) {
      for (int index : out.indices) {
        out.points.push_back(points[index]);
        out.before_refinement.push_back(refined == 0 ? points[index]
                                                     : before[index]);
      }
        out.path="normalized_"+out.path;
    }
    return finish();
}

MonoCornerIdentityResult MonoCornerIdentity::match(const std::vector<cv::Point2f>& observed,
        const cv::Mat& mask,bool capture) const {
    return matchImpl(observed,mask,capture,{},{},0,0);
}

MonoCornerIdentityResult MonoCornerIdentity::matchImpl(const std::vector<cv::Point2f>& observed,
        const cv::Mat& mask,bool capture,const cv::Mat& supplied_transform,
        const cv::Mat& supplied_image,double supplied_iou,double supplied_gap) const {
    MonoCornerIdentityResult out; out.status="invalid_input";
    const auto begin=std::chrono::steady_clock::now();
    auto finish=[&]() {
        out.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        if(capture) out.reference=reference_;
        return out;
    };
    const int n = (int)model_.size(), observed_count = (int)observed.size();
    out.detected_count = observed_count;
    const bool partial = observed_count != n;
    if (n < 4 || observed.empty() ||
        (partial && (!config_.partial_matching ||
                     observed_count < config_.min_reliable_points ||
                     observed_count > 12)) ||
        mask.empty() || mask.type() != CV_8UC1)
      return finish();
    for(auto p:observed) if(!finite(p)) return finish();
    const double size=span(observed);
    if(size<config_.min_span_px) { out.status="below_resolvable_span"; return finish(); }
    auto bounds=cv::boundingRect(observed);
    if(bounds.width<mask.cols*config_.min_roi_span_ratio || bounds.height<mask.rows*config_.min_roi_span_ratio) {
        // Class0 and class1 have similar silhouettes but different physical
        // scales. Do not relabel a tiny inner mark found inside a class0 ROI.
        out.status="component_too_small_for_roi";return finish();
    }
    for (int i = 0; i < observed_count; ++i)
      for (int j = 0; j < i; ++j)
        if(cv::norm(observed[i]-observed[j])<0.5) { out.status="duplicate_points";return finish(); }
    const double model_area=cv::contourArea(model_,true), image_area=cv::contourArea(observed,true);
    if(std::abs(image_area)<1.) { out.status="degenerate_contour";return finish(); }
    // The board is seen from its printed side. Normalize traversal, never admit
    // a reflected image-to-model mapping to explain a wrong correspondence.
    const int direction=model_area*image_area>0 ? 1 : -1;
    const double tolerance=std::max(config_.max_fit_error_px,config_.max_fit_error_ratio*size);
    cv::Mat validity(mask.size(),CV_8U,cv::Scalar(255));
    auto fit = [&](const std::vector<int> &indices,
                   const std::vector<int> &template_ids = std::vector<int>{}) {
      Candidate c;
      c.ids = indices;
      ++out.hypotheses;
      c.model_ids = template_ids;
      if (c.model_ids.empty()) {
        c.model_ids.resize(n);
        std::iota(c.model_ids.begin(), c.model_ids.end(), 0);
      }
      const int count = (int)indices.size();
      if (count < 4 || c.model_ids.size() != indices.size())
        return c;
      std::vector<cv::Point2f> ordered, model;
      for (int i : indices)
        ordered.push_back(observed[i]);
      for (int i : c.model_ids)
        model.push_back(model_[i]);
      if (!supplied_transform.empty() && count == n) {
        int inconsistent = 0;
        for (int i = 0; i < n; ++i) {
          int prev = (i + n - 1) % n, next = (i + 1) % n;
          if (cross(model_[i] - model_[prev], model_[next] - model_[i]) *
                  cross(ordered[i] - ordered[prev],
                        ordered[next] - ordered[i]) <=
              0)
            ++inconsistent;
        }
        if (inconsistent > std::max(2, n / 5))
          return c;
      }
      cv::Mat H = cv::findHomography(model, ordered, 0), support;
      auto fit_error = [&](const cv::Mat &transform, const cv::Mat &inliers) {
        if (transform.empty() || !cv::checkRange(transform))
          return std::numeric_limits<double>::infinity();
        std::vector<cv::Point2f> fitted;
        cv::perspectiveTransform(model, fitted, transform);
        double sum = 0;
        int used = 0;
        for (int i = 0; i < count; ++i)
          if (inliers.empty() || inliers.at<unsigned char>(i)) {
            sum += squaredNorm(fitted[i] - ordered[i]);
            ++used;
          }
        return used ? std::sqrt(sum / used)
                    : std::numeric_limits<double>::infinity();
      };
      c.fit = fit_error(H, {});
      bool individual_outlier = false;
      if (config_.quality_filter && !H.empty()) {
        std::vector<cv::Point2f> fitted;
        cv::perspectiveTransform(model, fitted, H);
        for (int i = 0; i < count; ++i)
          if (cv::norm(fitted[i] - ordered[i]) > tolerance)
            individual_outlier = true;
      }
      const int minimum = (config_.quality_filter || partial)
                              ? std::min(n, config_.min_reliable_points)
                              : std::max(4, (int)std::ceil(.8 * n));
      if (c.fit > tolerance || individual_outlier) {
        struct RngGuard {
          uint64 state = cv::theRNG().state;
          ~RngGuard() { cv::theRNG().state = state; }
        } rng_guard;
        H = cv::findHomography(model, ordered, cv::RANSAC, tolerance, support,
                               128, .99);
        if (H.empty() || support.empty() || cv::countNonZero(support) < minimum)
          return c;
        c.fit = fit_error(H, support);
      }
      if (!std::isfinite(c.fit) || c.fit > tolerance)
        return c;
      const int unsupported =
          support.empty() ? 0 : count - cv::countNonZero(support);
      if (config_.quality_filter || partial) {
        std::vector<cv::Point2f> projected;
        cv::perspectiveTransform(model, projected, H);
        std::vector<cv::Point2f> reliable_model, reliable_image;
        std::vector<int> reliable_ids, reliable_sources;
        for (int i = 0; i < count; ++i) {
          double error = cv::norm(projected[i] - ordered[i]);
          if (error <= tolerance &&
              (support.empty() || support.at<unsigned char>(i))) {
            reliable_model.push_back(model[i]);
            reliable_image.push_back(ordered[i]);
            reliable_ids.push_back(c.model_ids[i]);
            reliable_sources.push_back(c.ids[i]);
          } else {
            c.rejected_points.push_back(ordered[i]);
            c.rejected_ids.push_back(c.model_ids[i]);
          }
        }
        if ((int)reliable_ids.size() < minimum ||
            !coversModel(reliable_model, model_, config_.min_model_coverage))
          return c;
        H = cv::findHomography(reliable_model, reliable_image, 0);
        if (H.empty() || !cv::checkRange(H))
          return c;
        cv::perspectiveTransform(reliable_model, projected, H);
        double sum = 0;
        for (size_t i = 0; i < projected.size(); ++i) {
          double error = cv::norm(projected[i] - reliable_image[i]);
          if (error > tolerance)
            return c;
          c.residuals.push_back(error);
          sum += error * error;
        }
        c.fit = std::sqrt(sum / projected.size());
        c.ids = std::move(reliable_sources);
        c.model_ids = std::move(reliable_ids);
      }
      double minimum_w = std::numeric_limits<double>::max(), maximum_w = 0;
      int sign = 0;
      const double edge = config_.normalized_size - 1.;
      for (auto p : std::vector<cv::Point2d>{
               {0, 0}, {edge, 0}, {edge, edge}, {0, edge}}) {
        double w = H.at<double>(2, 0) * p.x + H.at<double>(2, 1) * p.y +
                   H.at<double>(2, 2);
        if (!std::isfinite(w) || std::abs(w) < 1e-10 ||
            (sign && (w > 0 ? 1 : -1) != sign))
          return c;
        sign = w > 0 ? 1 : -1;
        minimum_w = std::min(minimum_w, std::abs(w));
        maximum_w = std::max(maximum_w, std::abs(w));
      }
      if (minimum_w / maximum_w < 1e-4 ||
          cv::invert(H, c.inverse, cv::DECOMP_LU) == 0 ||
          !cv::checkRange(c.inverse))
        return c;
      std::vector<cv::Point2f> full;
      cv::perspectiveTransform(model_, full, H);
      if (cv::contourArea(full, true) * model_area <= 0)
        return c;
      cv::warpPerspective(mask, c.normalized, c.inverse, reference_.size(),
                          cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
      cv::Mat visible, visible_model;
      cv::warpPerspective(validity, visible, c.inverse, reference_.size(),
                          cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);
      cv::bitwise_and(visible, reference_, visible_model);
      ++out.normalized_tests;
      if (cv::countNonZero(visible_model) < .98 * cv::countNonZero(reference_))
        return c;
      c.iou = overlap(c.normalized, reference_);
      c.score = 1. - c.iou + .25 * c.fit / tolerance + .05 * unsupported +
                .01 * (n - (int)c.ids.size());
      c.valid = c.iou >= config_.min_iou;
      return c;
    };
    auto accept = [&](const Candidate &c, const char *path, double gap) {
      out.accepted = true;
      out.status = "accepted";
      out.path = path;
      out.indices = c.ids;
      out.template_ids = c.model_ids;
      out.rejected_points = c.rejected_points;
      out.rejected_ids = c.rejected_ids;
      out.residuals_px = c.residuals;
      if (partial)
        for (int i = 0; i < observed_count; ++i) {
          if (std::find(c.ids.begin(), c.ids.end(), i) != c.ids.end())
            continue;
          bool already = false;
          for (auto p : out.rejected_points)
            if (cv::norm(p - observed[i]) < .01)
              already = true;
          if (!already) {
            out.rejected_points.push_back(observed[i]);
            out.rejected_ids.push_back(-1);
          }
        }
      out.fit_error_px = c.fit;
      out.iou = c.iou;
      out.score_gap = gap;
      out.image_to_template = c.inverse;
      if (capture)
        out.normalized = c.normalized;
    };

    if (partial) {
      // Bounded cyclic sequence alignment preserves contour adjacency. Only
      // real observed vertices enter H/PnP; missing model points stay missing.
      cv::Mat coarse = supplied_transform;
      if (coarse.empty()) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) {
          out.status = "empty_contour";
          return finish();
        }
        auto largest = std::max_element(
            contours.begin(), contours.end(), [](const auto &a, const auto &b) {
              return cv::contourArea(a) < cv::contourArea(b);
            });
        auto rect = cv::minAreaRect(*largest);
        cv::Point2f raw[4];
        rect.points(raw);
        float m = config_.normalized_size / 16.f,
              e = config_.normalized_size - 1 - m;
        std::vector<cv::Point2f> quad(raw, raw + 4),
            square{{m, m}, {e, m}, {e, e}, {m, e}};
        if (cv::contourArea(quad, true) * cv::contourArea(square, true) < 0)
          std::reverse(quad.begin(), quad.end());
        coarse = cv::getPerspectiveTransform(quad, square);
      }
      struct Alignment {
        double score;
        std::vector<int> ids, models;
      };
      std::vector<Alignment> alignments;
      double edge = config_.normalized_size - 1;
      cv::Mat rotate =
          (cv::Mat_<double>(3, 3) << 0, -1, edge, 1, 0, 0, 0, 0, 1);
      for (int rotation = 0; rotation < 4; ++rotation) {
        std::vector<cv::Point2f> normalized;
        cv::perspectiveTransform(observed, normalized, coarse);
        for (int start = 0; start < observed_count; ++start)
          for (int model_start = 0; model_start < n; ++model_start) {
            int width = observed_count + 1;
            std::vector<double> dp((n + 1) * width, 1e9);
            std::vector<char> action(dp.size(), 0);
            dp[0] = 0;
            auto update = [&](int i, int j, double cost, char how) {
              int id = i * width + j;
              if (cost < dp[id]) {
                dp[id] = cost;
                action[id] = how;
              }
            };
            for (int i = 0; i <= n; ++i)
              for (int j = 0; j <= observed_count; ++j) {
                double cost = dp[i * width + j];
                if (i < n)
                  update(i + 1, j, cost + .035, 'm');
                if (j < observed_count)
                  update(i, j + 1, cost + .035, 'o');
                if (i < n && j < observed_count) {
                  int oi = (start + direction * j + observed_count) %
                           observed_count,
                      mi = (model_start + i) % n;
                  double distance =
                      squaredNorm(normalized[oi] - model_[mi]) / (edge * edge);
                  if (distance < .12)
                    update(i + 1, j + 1, cost + distance, 'p');
                }
              }
            Alignment a;
            a.score = dp[n * width + observed_count];
            int i = n, j = observed_count;
            while (i || j) {
              char how = action[i * width + j];
              if (how == 'p') {
                --i;
                --j;
                a.ids.push_back((start + direction * j + observed_count) %
                                observed_count);
                a.models.push_back((model_start + i) % n);
              } else if (how == 'm')
                --i;
              else if (how == 'o')
                --j;
              else
                break;
            }
            if ((int)a.ids.size() < config_.min_reliable_points)
              continue;
            std::vector<std::pair<int, int>> pairs;
            for (size_t k = 0; k < a.ids.size(); ++k)
              pairs.emplace_back(a.models[k], a.ids[k]);
            std::sort(pairs.begin(), pairs.end());
            a.ids.clear();
            a.models.clear();
            for (auto pair : pairs) {
              a.models.push_back(pair.first);
              a.ids.push_back(pair.second);
            }
            auto duplicate = std::find_if(
                alignments.begin(), alignments.end(), [&](const auto &b) {
                  return a.ids == b.ids && a.models == b.models;
                });
            if (duplicate == alignments.end())
              alignments.push_back(std::move(a));
            else
              duplicate->score = std::min(duplicate->score, a.score);
          }
        coarse = rotate * coarse;
      }
      std::sort(alignments.begin(), alignments.end(),
                [](const auto &a, const auto &b) { return a.score < b.score; });
      std::vector<Candidate> candidates;
      for (int k = 0;
           k < std::min((int)alignments.size(), config_.max_partial_hypotheses);
           ++k) {
        auto c = fit(alignments[k].ids, alignments[k].models);
        if (c.valid)
          candidates.push_back(std::move(c));
      }
      std::sort(candidates.begin(), candidates.end(),
                [](const auto &a, const auto &b) { return a.score < b.score; });
      if (candidates.empty()) {
        out.status = "partial_no_geometric_shape_match";
        return finish();
      }
      double gap = 1.;
      for (size_t k = 1; k < candidates.size(); ++k) {
        bool conflict = false;
        for (size_t i = 0; i < candidates[0].ids.size(); ++i)
          for (size_t j = 0; j < candidates[k].ids.size(); ++j)
            if ((candidates[0].ids[i] == candidates[k].ids[j]) !=
                (candidates[0].model_ids[i] == candidates[k].model_ids[j]))
              conflict = true;
        if (conflict) {
          gap = candidates[k].score - candidates[0].score;
          break;
        }
      }
      if (gap < config_.min_score_gap) {
        out.status = "ambiguous_partial_identity";
        return finish();
      }
      accept(candidates[0], "partial_projective", gap);
      return finish();
    }

    if(config_.method!="projective") {
        // The normalized frontend already evaluated all four rotations. Reuse
        // its image/transform instead of repeating warps and contour extraction.
        if(!supplied_transform.empty()) {
            out.coarse_iou=supplied_iou;
            if(capture) out.coarse_normalized=supplied_image;
            if(supplied_iou>=.80 && supplied_gap>=.10) {
                std::vector<cv::Point2f> normalized;
                cv::perspectiveTransform(observed,normalized,supplied_transform);
                std::vector<std::pair<double,std::vector<int>>> rotations;
                for(int start=0;start<n;++start) {
                    auto ids=cycle(n,start,direction);double error=0;
                    for(int i=0;i<n;++i) error+=squaredNorm(normalized[ids[i]]-model_[i]);
                    rotations.emplace_back(std::sqrt(error/n),std::move(ids));
                }
                std::sort(rotations.begin(),rotations.end(),[](const auto& a,const auto& b){return a.first<b.first;});
                double gap=(rotations[1].first-rotations[0].first)/config_.normalized_size;
                if(gap>=config_.min_score_gap) {
                    auto candidate=fit(rotations[0].second);
                    if(candidate.valid && candidate.iou>=.85) {accept(candidate,"tt_fourway",gap);return finish();}
                }
            }
        } else {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
        if(!contours.empty()) {
            auto largest=std::max_element(contours.begin(),contours.end(),[](const auto& a,const auto& b){return cv::contourArea(a)<cv::contourArea(b);});
            auto rect=cv::minAreaRect(*largest);cv::Point2f raw[4];rect.points(raw);
            std::vector<cv::Point2f> quad(raw,raw+4);
            const float m=config_.normalized_size/16.f,e=config_.normalized_size-1-m;
            std::vector<cv::Point2f> square{{m,m},{e,m},{e,e},{m,e}};
            if(cv::contourArea(quad,true)*cv::contourArea(square,true)<0) std::reverse(quad.begin(),quad.end());
            struct Coarse {double iou;cv::Mat transform,normalized;};std::vector<Coarse> coarse;
            if(rect.size.width>=2 && rect.size.height>=2) for(int k=0;k<4;++k) {
                std::vector<cv::Point2f> target;for(int i=0;i<4;++i) target.push_back(square[(i+k)%4]);
                cv::Mat H=cv::getPerspectiveTransform(quad,target),normalized;
                cv::warpPerspective(mask,normalized,H,reference_.size(),cv::INTER_NEAREST,cv::BORDER_CONSTANT,0);
                coarse.push_back({overlap(normalized,reference_),H,normalized}); ++out.normalized_tests;
            }
            std::sort(coarse.begin(),coarse.end(),[](const auto& a,const auto& b){return a.iou>b.iou;});
            if(!coarse.empty()) {
                out.coarse_iou=coarse[0].iou;
                if(capture) out.coarse_normalized=coarse[0].normalized;
                if(coarse[0].iou>=.80 && coarse[0].iou-coarse[1].iou>=.10) {
                    std::vector<cv::Point2f> normalized;
                    cv::perspectiveTransform(observed,normalized,coarse[0].transform);
                    std::vector<std::pair<double,std::vector<int>>> rotations;
                    for(int start=0;start<n;++start) {
                        auto ids=cycle(n,start,direction);double error=0.;
                        for(int i=0;i<n;++i) error+=squaredNorm(normalized[ids[i]]-model_[i]);
                        rotations.emplace_back(std::sqrt(error/n),std::move(ids));
                    }
                    std::sort(rotations.begin(),rotations.end(),[](const auto& a,const auto& b){return a.first<b.first;});
                    const double gap=(rotations[1].first-rotations[0].first)/config_.normalized_size;
                    if(gap>=config_.min_score_gap) {
                        auto candidate=fit(rotations[0].second);
                        if(candidate.valid && candidate.iou>=.85) {
                            accept(candidate,"tt_fourway",gap);return finish();
                        }
                    }
                }
            }
        }
        }
        if(config_.method=="tt_fourway") {out.status="coarse_unreliable";return finish();}
    }
    std::vector<Candidate> candidates;
    for(int start=0;start<n;++start) {
        auto c=fit(cycle(n,start,direction));
        if(c.valid) candidates.push_back(std::move(c));
    }
    std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.score<b.score;});
    if(candidates.empty()) {out.status="no_geometric_shape_match";return finish();}
    const double gap=candidates.size()>1 ? candidates[1].score-candidates[0].score : 1.;
    out.fit_error_px=candidates[0].fit;out.iou=candidates[0].iou;out.score_gap=gap;
    if(capture) out.normalized=candidates[0].normalized;
    if(gap<config_.min_score_gap) {out.status="ambiguous_identity";return finish();}
    accept(candidates[0],"projective",gap);return finish();
}
} // namespace gpnp
