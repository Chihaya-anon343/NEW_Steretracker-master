// Fixed ROI quality and paired timing. Ground truth is read only by the Python
// report.
#include "feature/BinaryCornerExtractor.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>

int main(int argc, char **argv) {
  if (argc != 5)
    return 2;
  cv::setNumThreads(1);
  gpnp::g_verbose_console = false;
  cv::FileStorage fs(argv[1], cv::FileStorage::READ);
  if (!fs.isOpened())
    return 3;
  auto b = fs["strategies"]["binary_corner"];
  gpnp::BinaryCornerExtractor::Config c;
  c.corners = (int)b["corners"];
  c.kernel_size = (int)b["kernel_size"];
  c.scale = (double)b["corner_scale"];
  c.target_size = {(int)b["target_width"], (int)b["target_height"]};
  c.pixel_to_meter_scale_class0 = (double)b["pixel_to_meter_scale_class0"];
  c.pixel_to_meter_scale_class1 = (double)b["pixel_to_meter_scale_class1"];
  c.mono_identity = gpnp::readMonoCornerIdentityConfig(b["mono_identity"]);
  c.mono_identity.enabled = true;
  c.mono_identity.frontend = "normalized";
  c.mono_identity.refine_original = true;
  std::vector<std::unique_ptr<gpnp::BinaryCornerExtractor>> extractors;
  for (int mode = 0; mode < 3; ++mode) {
    c.mono_identity.quality_filter = mode > 0;
    c.mono_identity.partial_matching = mode > 1;
    auto e = std::make_unique<gpnp::BinaryCornerExtractor>(
        c, (std::string)b["template_dir"]);
    e->setMonoContext(true);
    e->setDebugCapture(false);
    extractors.push_back(std::move(e));
  }
  const char *modes[] = {"baseline", "quality", "partial"};
  std::ifstream jobs(argv[2]);
  std::ofstream out(argv[3]);
  if (!jobs || !out)
    return 4;
  const int repeats = std::stoi(argv[4]);
  if (repeats < 1)
    return 5;
  out << "job,cls,mode,repeat,success,n";
  for (int i = 0; i < 10; ++i)
    out << ",x" << i << ",y" << i << ",id" << i;
  out << ",status,path,extract_ms,hypotheses,detected,rejected,iou,fit_px\n"
      << std::setprecision(12);
  int job, cls, unused, x, y, w, h;
  std::string path, previous;
  cv::Mat image;
  while (jobs >> job >> path >> cls >> unused >> x >> y >> w >> h) {
    if (path != previous) {
      image = cv::imread(path);
      previous = path;
    }
    if (image.empty())
      return 6;
    cv::Mat gray;
    cv::cvtColor(image(cv::Rect(x, y, w, h)), gray, cv::COLOR_BGR2GRAY);
    for (auto &e : extractors) {
      e->setUseClass1(cls == 1);
      if (repeats > 1)
        e->extractMono(gray, {});
    }
    for (int repeat = 0; repeat < repeats; ++repeat)
      for (int leg = 0; leg < 3; ++leg) {
        const int mode = (repeat + leg) % 3;
        auto &e = *extractors[mode];
        auto begin = std::chrono::steady_clock::now();
        auto result = e.extractMono(gray, {});
        double elapsed = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
        out << job << ',' << cls << ',' << modes[mode] << ',' << repeat << ','
            << result.success << ',' << result.good_matches.size();
        for (int i = 0; i < 10; ++i) {
          if (i < (int)result.good_matches.size())
            out << ',' << result.pts_left_match[i].x + x << ','
                << result.pts_left_match[i].y + y << ','
                << result.good_matches[i].trainIdx;
          else
            out << ",,,";
        }
        const auto &d = e.lastMonoIdentity();
        out << ',' << d.status << ',' << d.path << ',' << elapsed << ','
            << d.hypotheses << ',' << d.detected_count << ','
            << result.bc_rejected_count << ',' << d.iou << ',' << d.fit_error_px
            << '\n';
      }
  }
  return jobs.eof() ? 0 : 7;
}
