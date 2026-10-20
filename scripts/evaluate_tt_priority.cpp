// Forced TT extraction on immutable oracle ROIs. This measures observations,
// not a new GT definition: downstream reports must label projected bbox corners
// as virtual references and keep physical-model scale checks separate.
#include "feature/TinyTargetExtractor.hpp"
#include "common/TemplateGeometry.hpp"
#include "common/LogConfig.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace {
#ifdef TT_LEGACY_BASELINE
struct BaselineQuality {
    std::string status="disabled";
    double span_px=0,span_x_px=0,span_y_px=0,area_px=0,legacy_iou=0,direction_iou=0,direction_gap=0;
    int quadrant=-1,legacy_quadrant=-1,refined_count=0;
    double max_refine_shift_px=0,max_proposed_refine_shift_px=0;
    bool direction_reliable=false,geometry_valid=false;
    std::vector<cv::Point2f> initial_corners,final_corners;
    std::array<double,4> direction_scores{{0,0,0,0}};
    std::vector<unsigned char> accepted_refinement;
    std::map<std::string,double> stage_ms;
};
#endif
std::string csv(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result="\"";
    for (const char c:value) { result+=c; if(c=='\"') result+='\"'; }
    return result+'\"';
}
std::string flags(const std::vector<unsigned char>& values) {
    std::string result;
    for (auto v:values) { if(!result.empty()) result+=';'; result+=std::to_string(int(v)); }
    return result;
}
template<class Map> std::string encodedTimings(const Map& times) {
    std::ostringstream text; text<<std::setprecision(17);
    for(const auto& t:times) text<<t.first<<':'<<t.second<<';';
    return text.str();
}
void point(std::ostream& out,const std::vector<cv::Point2f>& points,int i) {
    if(i<int(points.size())) out<<','<<points[i].x<<','<<points[i].y;
    else out<<",,";
}
}
int main(int argc,char** argv) {
    if(argc<5 || argc>6) {
        std::cerr<<"Usage: evaluate_tt_priority config.json jobs.txt output.csv repeats [capture=0]\n";
        return 2;
    }
    try {
        cv::setNumThreads(1);gpnp::g_verbose_console=false;
        const int repeats=std::stoi(argv[4]);
        const bool capture=argc==6 && std::stoi(argv[5])!=0;
        if(repeats<1) return 2;
        cv::FileStorage fs(argv[1],cv::FileStorage::READ);
        if(!fs.isOpened()) return 3;
        const auto t=fs["strategies"]["tiny_target"];
        gpnp::TinyTargetExtractor::Config config;
        if(!t["target_width"].empty())config.target_size.width=int(t["target_width"]);
        if(!t["target_height"].empty())config.target_size.height=int(t["target_height"]);
        if(!t["scale_factor"].empty())config.scale_factor=int(t["scale_factor"]);
        if(!t["square_size_m_class0"].empty())config.square_size_m_class0=double(t["square_size_m_class0"]);
        if(!t["square_size_m_class1"].empty())config.square_size_m_class1=double(t["square_size_m_class1"]);
#ifndef TT_LEGACY_BASELINE
        config.robustness=gpnp::readTinyTargetRobustnessConfig(t["robustness"]);
#endif
        config.template_geometry=gpnp::readTemplateGeometryConfig(fs["template_geometry"]);
        gpnp::TinyTargetExtractor extractor(config,std::string(t["template_dir"]));
        extractor.setDebugCapture(capture);
#ifndef TT_LEGACY_BASELINE
        extractor.setMonoContext(true);
#endif
        std::ifstream input(argv[2]);std::ofstream out(argv[3]);
        if(!input || !out) return 4;
        out<<"job,cls,repeat,success,n,angle,overlap";
        for(int i=0;i<4;++i)out<<",x"<<i<<",y"<<i<<",gx"<<i<<",gy"<<i<<",id"<<i;
        out<<",status,failure_reason,span_px,span_x_px,span_y_px,area_px,legacy_iou,direction_iou,direction_gap,quadrant,legacy_quadrant,direction_reliable,geometry_valid,refined_count,max_refine_shift_px,max_proposed_refine_shift_px";
        for(int i=0;i<4;++i)out<<",raw"<<i<<"x,raw"<<i<<"y,final"<<i<<"x,final"<<i<<"y";
        out<<",direction0,direction1,direction2,direction3,accepted_refinement,extract_ms,stage_ms,pipeline_timing\n"<<std::setprecision(17);
        std::string line,previous;cv::Mat image;
        while(std::getline(input,line)) {
            if(line.empty() || line.front()=='#')continue;
            std::istringstream fields(line);
            int job,cls,method,x,y,w,h;std::string path;
            if(!(fields>>job>>std::quoted(path)>>cls>>method>>x>>y>>w>>h)) {
                std::cerr<<"Invalid job row: "<<line<<'\n';return 6;
            }
            if(cls<0 || cls>1 || method!=1 || w<1 || h<1) return 6;
            if(path!=previous) {image=cv::imread(path);previous=path;}
            if(image.empty()) {std::cerr<<"Cannot read image: "<<path<<'\n';return 5;}
            if(x<0||y<0||x+w>image.cols||y+h>image.rows)return 6;
            const cv::Mat color=image(cv::Rect(x,y,w,h));cv::Mat gray;
            cv::cvtColor(color,gray,cv::COLOR_BGR2GRAY);
            extractor.setUseClass1(cls==1);
            if(repeats>1)extractor.extractMono(gray,{}); // untimed cache warm-up
            for(int repeat=0;repeat<repeats;++repeat) {
                const auto start=std::chrono::steady_clock::now();
                const auto r=extractor.extractMono(gray,{});
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
#ifndef TT_LEGACY_BASELINE
                const auto& q=extractor.lastQuality();
#else
                const BaselineQuality q;
#endif
                out<<job<<','<<cls<<','<<repeat<<','<<r.success<<','<<r.pts_left_match.size()<<','<<extractor.lastMatchedAngle()<<','<<extractor.lastMatchOverlap();
                for(int i=0;i<4;++i) {
                    if(i<int(r.pts_left_match.size())) {
                        const auto p=r.pts_left_match[i];
                        out<<','<<p.x<<','<<p.y<<','<<p.x+x<<','<<p.y+y<<',';
                        if(i<int(r.good_matches.size()))out<<r.good_matches[i].trainIdx;
                    }else out<<",,,,,";
                }
                out<<','<<csv(q.status)<<','<<csv(r.failure_reason)<<','<<q.span_px<<','<<q.span_x_px<<','<<q.span_y_px<<','<<q.area_px<<','<<q.legacy_iou<<','<<q.direction_iou<<','<<q.direction_gap<<','<<q.quadrant<<','<<q.legacy_quadrant<<','<<q.direction_reliable<<','<<q.geometry_valid<<','<<q.refined_count<<','<<q.max_refine_shift_px<<','<<q.max_proposed_refine_shift_px;
                for(int i=0;i<4;++i){point(out,q.initial_corners,i);point(out,q.final_corners,i);}
                for(double s:q.direction_scores)out<<','<<s;
                out<<','<<csv(flags(q.accepted_refinement))<<','<<elapsed<<','<<csv(encodedTimings(q.stage_ms))<<','<<csv(encodedTimings(r.timing))<<'\n';
            }
            if(job%40==0)out.flush();
        }
        return input.eof() && out.good()?0:6;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 7;}
}
