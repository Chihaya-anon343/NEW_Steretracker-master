/**
 * @file ObbCornerExtractor.cpp
 * @brief YOLO-OBB 角点提取器实现（Python main.py 流水线的 C++ 移植）。
 *
 * 移植对应关系（函数: Python main.py 行号, 2026-09-23 版）：
 *   quadCheck      ← quad_check:100
 *   orderCorners   ← order_corners:121
 *   (H1 归一化)    ← normalize_quad:130
 *   processTile    ← process_tile:143
 *   loadTemplates  ← load_templates:205
 *   matchTemplate  ← match_template:275
 *   (反投影)       ← main() 626-643
 */

#include "feature/ObbCornerExtractor.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

namespace gpnp {
namespace {

/// 解析模板 txt 的 "Corner_N: x, y" 行（与 Python 正则等价）。
bool parseCornerLine(const std::string& line, int& idx, cv::Point2f& pt) {
    // 跳过注释与空行
    if (line.empty() || line[0] == '#') return false;
    float x = 0.0f, y = 0.0f;
    int i = -1;
    if (std::sscanf(line.c_str(), " Corner_%d : %f , %f", &i, &x, &y) == 3 ||
        std::sscanf(line.c_str(), "Corner_%d : %f , %f", &i, &x, &y) == 3) {
        idx = i;
        pt = cv::Point2f(x, y);
        return true;
    }
    return false;
}

/// 点集质心。
cv::Point2f centroidOf(const std::vector<cv::Point2f>& pts) {
    cv::Point2f c(0.0f, 0.0f);
    for (const auto& p : pts) c += p;
    if (!pts.empty()) c *= 1.0f / static_cast<float>(pts.size());
    return c;
}

/// 掩码非零像素质心（画布坐标）。
cv::Point2d maskCentroid(const cv::Mat& mask) {
    std::vector<cv::Point> nz;
    cv::findNonZero(mask, nz);
    if (nz.empty()) return {0.0, 0.0};
    double sx = 0.0, sy = 0.0;
    for (const auto& p : nz) {
        sx += p.x;
        sy += p.y;
    }
    return {sx / nz.size(), sy / nz.size()};
}

/// 16×16 白占比图（INTER_AREA 软下采样, 不二值化）。
cv::Mat makeFrac16(const cv::Mat& mask) {
    cv::Mat f;
    cv::resize(mask, f, cv::Size(16, 16), 0, 0, cv::INTER_AREA);
    f.convertTo(f, CV_32F, 1.0 / 255.0);
    return f;
}

/// 2×3 仿射（getRotationMatrix2D 结果）作用于单点。
cv::Point2f applyAffine(const cv::Mat& M, const cv::Point2f& p) {
    return cv::Point2f(
        static_cast<float>(M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y +
                           M.at<double>(0, 2)),
        static_cast<float>(M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y +
                           M.at<double>(1, 2)));
}

/// 单应矩阵作用于点集。
std::vector<cv::Point2f> applyHomography(const cv::Mat& H,
                                         const std::vector<cv::Point2f>& pts) {
    std::vector<cv::Point2f> out;
    if (pts.empty()) return out;
    std::vector<cv::Point2f> tmp = pts;
    cv::perspectiveTransform(tmp, out, H);
    return out;
}

} // namespace

ObbCornerExtractor::ObbCornerExtractor(const Config& cfg) : cfg_(cfg) {
    margin_ = (cfg_.morph_k >= 3) ? (cfg_.morph_k + 2) : 2;
    loadTemplates();
}

// ============================================================================
// 模板加载（← load_templates）
// ============================================================================

bool ObbCornerExtractor::loadTemplates() {
    templates_.clear();
    canonical_ready_ = false;
    canonical_corners_.clear();

    const int size = cfg_.norm_size;
    const int margin = margin_;
    const int inner = size - 2 * margin;
    if (inner < 16) return false;

    std::vector<std::string> pngs;
    for (int deg : {0, 90, 180, 270}) {
        std::ostringstream oss;
        oss << cfg_.template_dir << "/" << deg << "_degrees.png";
        pngs.push_back(oss.str());
    }

    for (const auto& png : pngs) {
        cv::Mat img = cv::imread(png, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        std::string txt = png.substr(0, png.size() - 4) + ".txt";
        std::ifstream fs(txt);
        if (!fs.is_open()) continue;

        // 二值化 → 最大外轮廓 → minAreaRect → 归一化到画布 [margin, margin+inner]
        cv::Mat bw;
        cv::threshold(img, bw, 127, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(bw, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (cnts.empty()) continue;
        const auto cnt = *std::max_element(cnts.begin(), cnts.end(),
                                           [](const auto& a, const auto& b) {
                                               return cv::contourArea(a) < cv::contourArea(b);
                                           });
        cv::RotatedRect rect = cv::minAreaRect(cnt);
        if (rect.size.width < 1 || rect.size.height < 1) continue;

        std::vector<cv::Point2f> src_pts(4);
        {
            cv::Mat bp;
            cv::boxPoints(rect, bp);
            for (int k = 0; k < 4; ++k) {
                src_pts[k] = cv::Point2f(bp.at<float>(k, 0), bp.at<float>(k, 1));
            }
        }
        const std::vector<cv::Point2f> dst_pts = {
            cv::Point2f(static_cast<float>(margin), static_cast<float>(margin)),
            cv::Point2f(static_cast<float>(margin + inner), static_cast<float>(margin)),
            cv::Point2f(static_cast<float>(margin + inner), static_cast<float>(margin + inner)),
            cv::Point2f(static_cast<float>(margin), static_cast<float>(margin + inner))};
        cv::Mat Ht = cv::getPerspectiveTransform(src_pts, dst_pts);

        Template t;
        t.name = png.substr(png.find_last_of("/\\") + 1);
        t.name = t.name.substr(0, t.name.size() - 4);
        t.deg = std::atoi(t.name.c_str());
        cv::warpPerspective(bw, t.mask, Ht, cv::Size(size, size),
                            cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        t.frac16 = makeFrac16(t.mask);
        t.cent = maskCentroid(t.mask);

        // txt 角点经同一单应变换到画布坐标
        std::vector<std::pair<int, cv::Point2f>> cs;
        std::string line;
        while (std::getline(fs, line)) {
            int idx = -1;
            cv::Point2f pt;
            if (parseCornerLine(line, idx, pt)) cs.emplace_back(idx, pt);
        }
        if (cs.size() < 4) continue;
        std::sort(cs.begin(), cs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        std::vector<cv::Point2f> raw;
        for (const auto& c : cs) raw.push_back(c.second);
        t.corners = applyHomography(Ht, raw);

        templates_.push_back(std::move(t));
    }

    // 自标定: 各模板相对 0° 模板的滚转 roll 与编号重排 perm
    const Template* canon = nullptr;
    for (const auto& t : templates_) {
        if (t.deg == 0) { canon = &t; break; }
    }
    if (canon == nullptr && !templates_.empty()) canon = &templates_[0];
    if (canon == nullptr) return false;

    const cv::Point2f c_center(size / 2.0f, size / 2.0f);
    for (auto& t : templates_) {
        double best_d = std::numeric_limits<double>::infinity();
        int best_k = 0;
        std::vector<int> best_perm(t.corners.size(), 0);
        for (int k = 0; k < 4; ++k) {
            cv::Mat R = cv::getRotationMatrix2D(c_center, k * 90.0, 1.0);
            std::vector<cv::Point2f> r;
            r.reserve(t.corners.size());
            for (const auto& p : t.corners) r.push_back(applyAffine(R, p));

            const size_t n = r.size();
            const size_t m = canon->corners.size();
            // 距离近的先配, 一对一
            std::vector<double> dmin(n, std::numeric_limits<double>::infinity());
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < m; ++i) {
                    dmin[j] = std::min<double>(dmin[j], cv::norm(r[j] - canon->corners[i]));
                }
            }
            std::vector<size_t> order(n);
            for (size_t j = 0; j < n; ++j) order[j] = j;
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b) { return dmin[a] < dmin[b]; });

            std::vector<int> perm(n, -1);
            std::vector<char> used(m, 0);
            double tot = 0.0;
            for (size_t j : order) {
                int best_i = -1;
                double bd = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < m; ++i) {
                    if (used[i]) continue;
                    double d = cv::norm(r[j] - canon->corners[i]);
                    if (d < bd) { bd = d; best_i = static_cast<int>(i); }
                }
                if (best_i >= 0) {
                    perm[j] = best_i;
                    used[best_i] = 1;
                    tot += bd;
                }
            }
            if (tot < best_d) {
                best_d = tot;
                best_k = k;
                best_perm = perm;
            }
        }
        t.roll = best_k * 90.0;
        t.perm = best_perm;
    }

    // canonical_corners_[i] = 0° 编号 i 的角点位置（画布坐标）
    // = 将模板角点按 perm 重排: cc[argsort(perm)[i]] = Corner_i 位置
    const Template& c0 = *canon;
    std::vector<int> argsort_perm(c0.perm.size());
    for (size_t i = 0; i < argsort_perm.size(); ++i) argsort_perm[i] = static_cast<int>(i);
    std::sort(argsort_perm.begin(), argsort_perm.end(),
              [&](int a, int b) { return c0.perm[a] < c0.perm[b]; });
    canonical_corners_.resize(c0.corners.size());
    for (size_t i = 0; i < argsort_perm.size(); ++i) {
        canonical_corners_[i] = c0.corners[argsort_perm[i]];
    }
    canonical_ready_ = !canonical_corners_.empty();

    return canonical_ready_;
}

// ============================================================================
// 几何检查 + 角点排序（← quad_check / order_corners）
// ============================================================================

bool ObbCornerExtractor::quadCheck(const std::array<cv::Point2f, 4>& pts, int w, int h,
                                   double min_area, double min_side) {
    for (const auto& p : pts) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
    }
    // 与图相交
    float maxx = std::max({pts[0].x, pts[1].x, pts[2].x, pts[3].x});
    float minx = std::min({pts[0].x, pts[1].x, pts[2].x, pts[3].x});
    float maxy = std::max({pts[0].y, pts[1].y, pts[2].y, pts[3].y});
    float miny = std::min({pts[0].y, pts[1].y, pts[2].y, pts[3].y});
    if (maxx < 0 || minx > w || maxy < 0 || miny > h) return false;

    // 鞋带面积
    double area = 0.0;
    for (int i = 0; i < 4; ++i) {
        const auto& a = pts[i];
        const auto& b = pts[(i + 1) % 4];
        area += a.x * b.y - b.x * a.y;
    }
    area = std::fabs(area) * 0.5;
    if (area < min_area) return false;

    // 凸性: 相邻边向量叉积同号
    cv::Point2f edge[4], nxt[4];
    for (int i = 0; i < 4; ++i) {
        edge[i] = pts[(i + 1) % 4] - pts[i];
    }
    for (int i = 0; i < 4; ++i) nxt[i] = edge[(i + 1) % 4];
    bool all_pos = true, all_neg = true;
    for (int i = 0; i < 4; ++i) {
        double cross = edge[i].x * nxt[i].y - edge[i].y * nxt[i].x;
        if (cross < -1e-6) all_pos = false;
        if (cross > 1e-6) all_neg = false;
    }
    if (!all_pos && !all_neg) return false;   // 凹四边形或自相交

    // 最短边
    double min_len = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 4; ++i) min_len = std::min<double>(min_len, cv::norm(edge[i]));
    if (min_len < min_side) return false;

    return true;
}

std::array<cv::Point2f, 4> ObbCornerExtractor::orderCorners(
    const std::array<cv::Point2f, 4>& pts) {
    const cv::Point2f c = (pts[0] + pts[1] + pts[2] + pts[3]) * 0.25f;
    std::vector<cv::Point2f> v(pts.begin(), pts.end());
    std::sort(v.begin(), v.end(), [&](const cv::Point2f& a, const cv::Point2f& b) {
        return std::atan2(a.y - c.y, a.x - c.x) < std::atan2(b.y - c.y, b.x - c.x);
    });
    // 最靠近包围盒左上角者为起点
    float minx = std::min({v[0].x, v[1].x, v[2].x, v[3].x});
    float miny = std::min({v[0].y, v[1].y, v[2].y, v[3].y});
    int start = 0;
    float bd = std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; ++i) {
        float d = (v[i].x - minx) * (v[i].x - minx) + (v[i].y - miny) * (v[i].y - miny);
        if (d < bd) { bd = d; start = i; }
    }
    std::array<cv::Point2f, 4> out{};
    for (int i = 0; i < 4; ++i) out[i] = v[(start + i) % 4];
    return out;   // [TL, TR, BR, BL] 视觉顺时针
}

// ============================================================================
// tile 后处理（← process_tile）
// ============================================================================

ObbCornerExtractor::TileProc ObbCornerExtractor::processTile(const cv::Mat& tile) const {
    TileProc out;
    const int s2 = tile.rows;

    cv::Mat gray;
    cv::cvtColor(tile, gray, cv::COLOR_BGR2GRAY);
    if (cfg_.denoise_k >= 3) {
        cv::medianBlur(gray, gray, cfg_.denoise_k | 1);
    }
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    out.binary = binary;

    cv::Mat labels, stats, centroids;
    int num = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    if (num <= 1) return out;   // 无前景

    // 最大白色连通域（0 号为背景）
    int lab = 1;
    int best_area = stats.at<int>(1, cv::CC_STAT_AREA);
    for (int i = 2; i < num; ++i) {
        int a = stats.at<int>(i, cv::CC_STAT_AREA);
        if (a > best_area) { best_area = a; lab = i; }
    }
    cv::Mat raw = (labels == lab);
    raw.convertTo(raw, CV_8U, 255.0);

    // 孔洞填充: 单连通域仅一条外轮廓, 整体填充覆盖所有孔洞
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(raw, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cnts.empty()) return out;
    const auto cnt = *std::max_element(cnts.begin(), cnts.end(),
                                       [](const auto& a, const auto& b) {
                                           return cv::contourArea(a) < cv::contourArea(b);
                                       });
    cv::Mat filled = cv::Mat::zeros(raw.size(), CV_8U);
    cv::drawContours(filled, std::vector<std::vector<cv::Point>>{cnt}, -1,
                     cv::Scalar(255), cv::FILLED);

    // H2 转正: 带角度外接矩形 → 画布中心 inner×inner 正方形
    cv::RotatedRect rect1 = cv::minAreaRect(cnt);
    if (rect1.size.width < 1 || rect1.size.height < 1) return out;

    const int m = margin_;
    const int inner = s2 - 2 * m;
    std::vector<cv::Point2f> src_pts(4);
    {
        cv::Mat bp;
        cv::boxPoints(rect1, bp);
        for (int k = 0; k < 4; ++k) src_pts[k] = cv::Point2f(bp.at<float>(k, 0), bp.at<float>(k, 1));
    }
    const std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(static_cast<float>(m), static_cast<float>(m)),
        cv::Point2f(static_cast<float>(m + inner), static_cast<float>(m)),
        cv::Point2f(static_cast<float>(m + inner), static_cast<float>(m + inner)),
        cv::Point2f(static_cast<float>(m), static_cast<float>(m + inner))};
    cv::Mat H2 = cv::getPerspectiveTransform(src_pts, dst_pts);

    cv::Mat mask2;
    cv::warpPerspective(filled, mask2, H2, cv::Size(s2, s2),
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    // 转正后边角优化（方形核与轴对齐角点严格保角）
    if (cfg_.morph_k >= 3) {
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(cfg_.morph_k, cfg_.morph_k));
        cv::morphologyEx(mask2, mask2, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask2, mask2, cv::MORPH_CLOSE, kernel);
    }

    cnts.clear();
    cv::findContours(mask2, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (cnts.empty()) return out;   // 形态学把过小目标抹除

    out.mask = mask2;
    const auto cnt2 = *std::max_element(cnts.begin(), cnts.end(),
                                        [](const auto& a, const auto& b) {
                                            return cv::contourArea(a) < cv::contourArea(b);
                                        });
    out.rect = cv::minAreaRect(cnt2);
    const double peri = cv::arcLength(cnt2, true);
    std::vector<cv::Point> poly_i;
    cv::approxPolyDP(cnt2, poly_i, cfg_.poly_eps * peri, true);
    out.poly.reserve(poly_i.size());
    for (const auto& p : poly_i) out.poly.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    out.H2 = H2;
    out.ok = true;
    return out;
}

// ============================================================================
// 模板比对 + 角点校验替换（← match_template）
// ============================================================================

ObbCornerExtractor::MatchOut ObbCornerExtractor::matchTemplate(
    const cv::Mat& mask, const std::vector<cv::Point2f>& poly) const {
    MatchOut out;
    if (mask.empty() || templates_.empty()) return out;

    // 1) k=16 恒等筛: argmax 模板即朝向
    cv::Mat f = makeFrac16(mask);
    const Template* best = nullptr;
    double best_score = -1.0;
    for (const auto& t : templates_) {
        cv::Mat diff = cv::abs(f - t.frac16);
        cv::Scalar s = cv::mean(diff);
        double score = 1.0 - s[0];
        if (score > best_score) { best_score = score; best = &t; }
    }
    if (best == nullptr) return out;

    out.valid = true;
    out.tmpl = best;
    out.score = best_score;

    // 2) 质心一次配准
    out.dt = maskCentroid(mask) - cv::Point2d(best->cent.x, best->cent.y);
    std::vector<cv::Point2f> tc = best->corners;
    for (auto& p : tc) p += cv::Point2f(static_cast<float>(out.dt.x), static_cast<float>(out.dt.y));
    // 重排为 0° 编号: tc[i] = Corner_i 位置
    if (best->perm.size() == tc.size()) {
        std::vector<int> argsort_perm(best->perm.size());
        for (size_t i = 0; i < argsort_perm.size(); ++i) {
            argsort_perm[i] = static_cast<int>(i);
        }
        std::sort(argsort_perm.begin(), argsort_perm.end(),
                  [&](int a, int b) { return best->perm[a] < best->perm[b]; });
        std::vector<cv::Point2f> reordered(tc.size());
        for (size_t i = 0; i < argsort_perm.size(); ++i) {
            reordered[i] = tc[argsort_perm[i]];
        }
        tc = reordered;
    }

    out.corners = tc;
    out.is_target = (best_score >= cfg_.tmpl_score);
    if (!out.is_target || poly.size() < 3) return out;

    // 3) 角点级校验替换（仅靶标）
    const size_t n_t = tc.size();
    const size_t n_d = poly.size();
    const float tau = static_cast<float>(cfg_.corner_tau);

    // d[i][j] = |tc_i - dp_j|
    auto dist = [&](size_t i, size_t j) {
        return static_cast<double>(cv::norm(tc[i] - poly[j]));
    };

    // 3.1 野点剔除: 离最近模板角点 > tau 的 DP 角点直接删
    std::vector<char> keep(n_d, 0);
    std::vector<size_t> ki;
    for (size_t j = 0; j < n_d; ++j) {
        double dmin = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n_t; ++i) dmin = std::min(dmin, dist(i, j));
        if (dmin <= tau) { keep[j] = 1; ki.push_back(j); }
    }

    // 3.2 一对一指派: 距离近的先配, 保证不重复
    out.corners = tc;
    std::vector<int> pairs(n_t, -1);
    if (!ki.empty()) {
        std::vector<size_t> t_order(n_t);
        for (size_t i = 0; i < n_t; ++i) t_order[i] = i;
        std::sort(t_order.begin(), t_order.end(), [&](size_t a, size_t b) {
            double da = std::numeric_limits<double>::infinity();
            double db = std::numeric_limits<double>::infinity();
            for (size_t j : ki) {
                da = std::min(da, dist(a, j));
                db = std::min(db, dist(b, j));
            }
            return da < db;
        });
        std::vector<char> used(ki.size(), 0);
        for (size_t i : t_order) {
            int best_j = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (size_t jj = 0; jj < ki.size(); ++jj) {
                if (used[jj]) continue;
                double d = dist(i, ki[jj]);
                if (d < bd) { bd = d; best_j = static_cast<int>(jj); }
            }
            if (best_j >= 0 && bd <= tau) {
                out.corners[i] = poly[ki[best_j]];
                pairs[i] = static_cast<int>(ki[best_j]);
                used[best_j] = 1;
            }
        }
        out.n_dp = static_cast<int>(std::count(used.begin(), used.end(), 1));
    }

    // 3.3 先验 1px 邻域含白 → 保留亚像素先验; 否则贴 mask 外轮廓最近点
    std::vector<size_t> rest;
    for (size_t i = 0; i < n_t; ++i) {
        if (pairs[i] >= 0) continue;
        int x = cvRound(tc[i].x), y = cvRound(tc[i].y);
        bool has_white = false;
        for (int dy = -1; dy <= 1 && !has_white; ++dy) {
            for (int dx = -1; dx <= 1 && !has_white; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= mask.cols || yy >= mask.rows) continue;
                if (mask.at<uchar>(yy, xx) != 0) has_white = true;
            }
        }
        if (has_white) {
            ++out.n_keep;
        } else {
            rest.push_back(i);
        }
    }
    if (!rest.empty()) {
        std::vector<std::vector<cv::Point>> cnts;
        cv::findContours(mask, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        if (!cnts.empty()) {
            const auto& pc = *std::max_element(cnts.begin(), cnts.end(),
                                               [](const auto& a, const auto& b) {
                                                   return cv::contourArea(a) < cv::contourArea(b);
                                               });
            for (size_t i : rest) {
                double bd = std::numeric_limits<double>::infinity();
                const cv::Point* bp = nullptr;
                for (const auto& p : pc) {
                    double d = std::hypot(p.x - tc[i].x, p.y - tc[i].y);
                    if (d < bd) { bd = d; bp = &p; }
                }
                if (bp != nullptr && bd <= cfg_.snap_r) {
                    out.corners[i] = cv::Point2f(static_cast<float>(bp->x),
                                                 static_cast<float>(bp->y));
                    ++out.n_snap;
                }
            }
        }
    }

    return out;
}

// ============================================================================
// 主入口: H1 归一化 → tile 后处理 → 模板比对 → 反投影
// ============================================================================

ObbCornerExtractor::Result ObbCornerExtractor::extract(
    const cv::Mat& bgr_img, const std::array<cv::Point2f, 4>& quad_raw) {
    Result res;
    const auto t0 = std::chrono::steady_clock::now();

    do {
        if (bgr_img.empty() || !canonical_ready_) break;

        const int w = bgr_img.cols, h = bgr_img.rows;
        const double min_area = cfg_.min_area_frac * w * h;
        if (!quadCheck(quad_raw, w, h, min_area, cfg_.min_side)) break;

        // H1 透视归一化
        const auto src4 = orderCorners(quad_raw);
        res.quad = src4;
        const std::vector<cv::Point2f> dst_pts = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(cfg_.norm_size), 0),
            cv::Point2f(static_cast<float>(cfg_.norm_size), static_cast<float>(cfg_.norm_size)),
            cv::Point2f(0, static_cast<float>(cfg_.norm_size))};
        cv::Mat H1 = cv::getPerspectiveTransform(
            std::vector<cv::Point2f>(src4.begin(), src4.end()), dst_pts);
        if (!cv::checkRange(H1)) break;                       // NaN/Inf 检查
        if (std::fabs(cv::determinant(H1)) < 1e-8) break;

        cv::Mat tile;
        cv::warpPerspective(bgr_img, tile, H1,
                            cv::Size(cfg_.norm_size, cfg_.norm_size));
        res.debug_tile = tile.clone();

        // tile 后处理
        TileProc proc = processTile(tile);
        if (!proc.ok) break;
        res.debug_mask = proc.mask.clone();

        // 模板比对 + 角点校验替换
        MatchOut mt = matchTemplate(proc.mask, proc.poly);
        if (!mt.valid) break;
        res.is_target = mt.is_target;
        res.score = mt.score;
        res.n_dp = mt.n_dp;
        res.n_keep = mt.n_keep;
        res.n_snap = mt.n_snap;

        // 反投影 (H2·H1)^-1: 画布 → 原图
        cv::Mat M = proc.H2 * H1;
        cv::Mat M_inv;
        cv::invert(M, M_inv, cv::DECOMP_LU);
        if (!cv::checkRange(M_inv)) break;

        // 最终角点（画布坐标 → 原图坐标）
        std::vector<cv::Point2f> canvas_corners;
        if (mt.is_target) {
            canvas_corners = mt.corners;
        } else {
            // 非靶标: 沿用 DP 角点（无法建立 0° 编号对应, 不产出 3D 点）
            canvas_corners = proc.poly;
        }
        res.corners = applyHomography(M_inv, canvas_corners);

        // 3D 点（仅靶标, 0° 编号 ↔ canonical 模板角点物理化）
        if (mt.is_target && cfg_.template_real_size_mm > 0.0 &&
            canonical_corners_.size() == res.corners.size()) {
            const int size = cfg_.norm_size;
            const int inner = size - 2 * margin_;
            const double s_mm = cfg_.template_real_size_mm;
            res.pts_3d.reserve(canonical_corners_.size());
            for (const auto& c : canonical_corners_) {
                const double nx = (c.x - margin_) / inner - 0.5;
                const double ny = (c.y - margin_) / inner - 0.5;
                res.pts_3d.emplace_back(nx * s_mm, ny * s_mm, 0.0);
            }
            res.success = res.corners.size() >= 4;
        } else if (mt.is_target) {
            // 未配置物理尺寸: 仍算提取成功但不产出 3D 点（PnP 由调用方跳过）
            res.success = res.corners.size() >= 4;
        }

        // 外接方形 4 角点反投影（可视化用, 0°右下起顺时针）
        {
            std::vector<cv::Point2f> sq(4);
            {
                cv::Mat bp;
                cv::boxPoints(proc.rect, bp);
                for (int k = 0; k < 4; ++k) {
                    sq[k] = cv::Point2f(bp.at<float>(k, 0), bp.at<float>(k, 1));
                }
            }
            std::array<cv::Point2f, 4> sq4{};
            std::copy(sq.begin(), sq.end(), sq4.begin());
            sq4 = orderCorners(sq4);
            if (mt.is_target && mt.tmpl != nullptr) {
                // 转回 0° 后取右下角 (x+y 最大) 为起点
                const cv::Point2f c_center(cfg_.norm_size / 2.0f, cfg_.norm_size / 2.0f);
                cv::Mat R = cv::getRotationMatrix2D(c_center, mt.tmpl->roll, 1.0);
                int start = 0;
                float best_sum = -std::numeric_limits<float>::infinity();
                for (int k = 0; k < 4; ++k) {
                    const auto r = applyAffine(R, sq4[k]);
                    if (r.x + r.y > best_sum) { best_sum = r.x + r.y; start = k; }
                }
                std::array<cv::Point2f, 4> rolled{};
                for (int k = 0; k < 4; ++k) rolled[k] = sq4[(start + k) % 4];
                sq4 = rolled;
            }
            const auto bp = applyHomography(M_inv,
                                            std::vector<cv::Point2f>(sq4.begin(), sq4.end()));
            if (bp.size() == 4) {
                std::copy(bp.begin(), bp.end(), res.rect_corners.begin());
            }
        }
    } while (false);

    const auto t1 = std::chrono::steady_clock::now();
    res.extract_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

} // namespace gpnp
