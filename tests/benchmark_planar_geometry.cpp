// Replay saved, already photometrically qualified correspondences through the real filter.
#include "matching/StereoFlowMatcher.hpp"
#include <opencv2/core.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
using namespace gpnp;
struct Group {std::vector<cv::Point2f> left;FlowMatchResult flow;double truth;};
int main(int argc,char** argv) {
    if(argc!=3){std::cerr<<"usage: benchmark_planar_geometry run_csv_directory manifest.json\n";return 2;}
    cv::FileStorage manifest(argv[2],cv::FileStorage::READ);
    std::vector<Group> groups;int frame=0;
    for(const auto& entry:manifest["frames"]) {
        ++frame;const double truth=entry["disparity"];
        std::ifstream csv(std::string(argv[1])+"/disparity_f"+std::to_string(frame)+".csv");
        if(!csv){std::cerr<<"missing frame "<<frame<<'\n';return 2;}
        std::string line;std::getline(csv,line);std::map<int,Group> per_region;
        while(std::getline(csv,line)) {
            std::istringstream stream(line);std::string field;std::vector<std::string> fields;
            while(std::getline(stream,field,','))fields.push_back(field);
            if(fields.size()<12)return 2;
            auto& g=per_region[std::stoi(fields[6])];g.truth=truth;
            g.left.emplace_back(std::stof(fields[1]),std::stof(fields[2]));
            FlowPointDiagnostic d;d.accepted=true;d.reason="accepted";
            d.tracked_right={std::stof(fields[3]),std::stof(fields[4])};g.flow.diagnostics.push_back(d);
        }
        for(auto& item:per_region)groups.push_back(std::move(item.second));
    }
    std::cout<<"threshold_px,accepted,rejected,mae_px,max_error_px,filter_mean_group_ms\n";
    for(double threshold:{0.1,0.2,0.25,0.3,0.5}) {
        StereoFlowConfig cfg;cfg.planar_geometry_filter=true;cfg.planar_max_error_px=threshold;
        long long accepted=0,rejected=0;double error=0,max_error=0,total_ms=0;
        for(int repeat=-1;repeat<10;++repeat)for(const auto& g:groups) {
            auto flow=g.flow;const auto start=std::chrono::steady_clock::now();
            filterPlanarGeometry(g.left,cfg,true,flow);
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(repeat<0)continue;total_ms+=ms;
            if(repeat==0)for(size_t i=0;i<g.left.size();++i) {
                const auto& d=flow.diagnostics[i];if(!d.accepted){++rejected;continue;}
                const double e=std::abs((g.left[i].x-d.tracked_right.x)-g.truth);
                ++accepted;error+=e;max_error=std::max(max_error,e);
            }
        }
        std::cout<<std::setprecision(10)<<threshold<<','<<accepted<<','<<rejected<<','
                 <<error/accepted<<','<<max_error<<','<<total_ms/(10*groups.size())<<'\n';
    }
}
