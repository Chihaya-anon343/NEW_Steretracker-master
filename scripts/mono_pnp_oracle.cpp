// Diagnostic only: exercise the project's unchanged MonoPnPSolver with known
// correspondences, isolating PnP conditioning from YOLO/contour extraction.
#include "pose/MonoPnPSolver.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/core.hpp>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>

double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return std::acos(std::clamp(a.normalized().dot(b.normalized()), -1.0, 1.0))*180/CV_PI;
}
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream input(argv[1]); std::ofstream out(argv[2]);
    if (!input || !out) return 3;
    gpnp::g_verbose_console = false;
    cv::setNumThreads(1); cv::setRNGSeed(20261015);
    out << "sequence,condition,frame,temporal,success,normal_error_deg,normal_step_deg,gt_step_deg,gap,z_jump,depth_rel_error\n" << std::setprecision(17);
    struct Previous { bool valid=false; int frame=-1; gpnp::PoseSeed pose; Eigen::Vector3d normal; };
    std::map<std::string,Previous> cache;
    gpnp::MonoPnPSolver solver;
    std::string sequence,condition;
    int frame,n; double fx,fy,cx,cy;
    while (input >> sequence >> condition >> frame >> n >> fx >> fy >> cx >> cy) {
        Eigen::Matrix3d R,K; Eigen::Vector3d t;
        K << fx,0,cx, 0,fy,cy, 0,0,1;
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) input >> R(a,b);
        for(int a=0;a<3;++a) input >> t(a);
        std::vector<Eigen::Vector3d> object(n); std::vector<cv::Point2f> image(n);
        for(int i=0;i<n;++i) input >> object[i](0) >> object[i](1) >> object[i](2) >> image[i].x >> image[i].y;
        if (!input) return 4;
        for (int temporal=0;temporal<2;++temporal) {
            auto& prev=cache[sequence+condition+std::to_string(temporal)];
            const auto pose=solver.solve(image,object,K,temporal && prev.valid ? &prev.pose : nullptr,temporal ? .3 : 0.);
            out << sequence << ',' << condition << ',' << frame << ',' << temporal << ',' << pose.success;
            if(pose.success) {
                const auto normal=pose.R.col(2).eval();
                double step=prev.valid ? angle(normal,prev.pose.R.col(2)) : 0;
                double gtstep=prev.valid ? angle(R.col(2),prev.normal) : 0;
                int gap=prev.valid ? frame-prev.frame : 0;
                out << ',' << angle(normal,R.col(2)) << ',' << step << ',' << gtstep << ',' << gap
                    << ',' << (gap==1 && step>20 && step-gtstep>15) << ',' << std::abs(pose.t.z()-t.z())/t.z();
                prev.valid=true; prev.frame=frame; prev.pose.R=pose.R; prev.pose.t=pose.t; prev.normal=R.col(2);
            } else out << ",,,,,,";
            out << '\n';
        }
    }
    return input.eof() ? 0 : 5;
}
