#include "common/Config.hpp"
#include "matching/StereoFlowMatcher.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace gpnp;
struct Case {
    std::string name; double disparity=45,dy=0; int x=900;
    bool occluded=false,blank=false,large_roi=false,cold=false;
};
double quantile(std::vector<double> v,double p) { std::sort(v.begin(),v.end());return v[size_t((v.size()-1)*p)]; }
int main(int argc,char** argv) {
    const int n=argc>1?std::stoi(argv[1]):50;
    const std::string output=argc>2?argv[2]:"output/bc_roi_experiment/synthetic";
    if(n<10) return 1;
    std::filesystem::create_directories(output);
    std::ofstream csv(output+"/samples.csv"),summary(output+"/summary.csv");
    csv<<"case,iteration,path,total_ms,pyramid_ms,lk_ms,crop_ms,coarse_ms,pairs,success,mean_error_px,max_error_px,roi_area,full_retry,roi_used\n";
    summary<<"case,path,mean_ms,p95_ms,success_count,iterations,min_pairs,max_pairs,mean_error_px,max_error_px,roi_area,full_retry_count,max_full_roi_delta_px,same_indices\n";
    Eigen::Matrix3d k;k<<1000,0,960,0,1000,540,0,0,1;
    const auto camera=makeStereoCameraParams(k,Eigen::Matrix3d::Identity(),Eigen::Vector3d(60,0,0));
    const std::vector<Case> cases{{"normal",45,0,900,false,false,false,true},
        {"large_disparity",300,0,900},{"cold_search",120,0,900,false,false,false,true},
        {"vertical_2px",45,2,900},{"subpixel",45.5,1.25,900},
        {"left_edge",80,0,160},{"right_edge",45,0,1730},
        {"occlusion",45,0,900,true},{"blank_right",45,0,900,false,true},
        {"large_roi_bypass",45,0,900,false,false,true}};
    bool all_equivalent=true,all_accurate=true;
    for(const auto& c:cases) {
        cv::Mat left(1080,1920,CV_8UC1,cv::Scalar(90)),right;
        std::vector<cv::Point2f> points{{float(c.x),400},{float(c.x+100),410},{float(c.x+140),470},
                                     {float(c.x+100),530},{float(c.x+20),520},{float(c.x-20),460}};
        cv::RNG rng(5086);
        for(auto p:points) {
            cv::Mat patch(49,49,CV_8UC1);rng.fill(patch,cv::RNG::UNIFORM,20,240);
            cv::GaussianBlur(patch,patch,{3,3},.6);
            patch.copyTo(left(cv::Rect(int(p.x)-24,int(p.y)-24,49,49)));
        }
        const cv::Mat transform=(cv::Mat_<double>(2,3)<<1,0,-c.disparity,0,1,c.dy);
        cv::warpAffine(left,right,transform,left.size(),cv::INTER_LINEAR,cv::BORDER_CONSTANT,cv::Scalar(90));
        if(c.occluded) right(cv::Rect(int(points[0].x-c.disparity)-24,int(points[0].y+c.dy)-24,49,49)).setTo(90);
        if(c.blank) right.setTo(90);
        StereoFlowConfig full;full.max_epipolar_error_px=3;
        auto roi=full;roi.roi_pyramid=true;
        StereoFlowMatcher a(camera,full),b(camera,roi);
        std::vector<cv::Point2f> seeds;
        for(auto p:points) seeds.push_back(c.cold?p:cv::Point2f(p.x-c.disparity+.2f,p.y+c.dy-.15f));
        const cv::Rect region=c.large_roi?cv::Rect(0,0,left.cols,left.rows):cv::Rect(c.x-40,370,220,190);
        for(int i=0;i<10;++i) { a.match(left,right,points,seeds,region);b.match(left,right,points,seeds,region); }
        std::vector<double> times[2];int success[2]{},retries[2]{},lo[2]{100000,100000},hi[2]{},area[2]{};
        double errors[2]{},max_error[2]{},max_delta=0;int observations[2]{};bool same=true;
        for(int i=0;i<n;++i) {
            FlowMatchResult result[2];
            if(i%2) {result[1]=b.match(left,right,points,seeds,region);result[0]=a.match(left,right,points,seeds,region);}
            else {result[0]=a.match(left,right,points,seeds,region);result[1]=b.match(left,right,points,seeds,region);}
            same=same && result[0].success==result[1].success && result[0].left_indices==result[1].left_indices;
            if(result[0].left_indices==result[1].left_indices)
                for(size_t j=0;j<result[0].right_points.size();++j) max_delta=std::max(max_delta,cv::norm(result[0].right_points[j]-result[1].right_points[j]));
            for(int p=0;p<2;++p) {
                const auto& r=result[p];double sum=0,worst=0;
                for(size_t j=0;j<r.right_points.size();++j) {
                    const auto truth=points[r.left_indices[j]]+cv::Point2f(-c.disparity,c.dy);
                    const double e=cv::norm(r.right_points[j]-truth);sum+=e;worst=std::max(worst,e);
                }
                times[p].push_back(r.total_ms);success[p]+=r.success;retries[p]+=r.full_frame_fallback;
                lo[p]=std::min(lo[p],int(r.left_indices.size()));hi[p]=std::max(hi[p],int(r.left_indices.size()));
                area[p]=r.roi_used?r.search_roi.area():left.cols*left.rows;
                errors[p]+=sum;observations[p]+=int(r.right_points.size());max_error[p]=std::max(max_error[p],worst);
                csv<<c.name<<','<<i<<','<<(p?"roi":"full")<<','<<r.total_ms<<','<<r.pyramid_ms<<','<<r.lk_ms<<','<<r.crop_ms<<','<<r.coarse_ms
                   <<','<<r.left_indices.size()<<','<<r.success<<','<<(r.right_points.empty()?0:sum/r.right_points.size())<<','<<worst
                   <<','<<area[p]<<','<<r.full_frame_fallback<<','<<r.roi_used<<'\n';
            }
        }
        for(int p=0;p<2;++p) summary<<c.name<<','<<(p?"roi":"full")<<','<<std::accumulate(times[p].begin(),times[p].end(),0.0)/n
            <<','<<quantile(times[p],.95)<<','<<success[p]<<','<<n<<','<<lo[p]<<','<<hi[p]<<','<<(observations[p]?errors[p]/observations[p]:0)
            <<','<<max_error[p]<<','<<area[p]<<','<<retries[p]<<','<<max_delta<<','<<same<<'\n';
        all_equivalent=all_equivalent && same && max_delta<.1;
        all_accurate=all_accurate && max_error[0]<.5 && max_error[1]<.5 && (c.blank?success[0]==0:success[0]==n);
        std::cout<<c.name<<": equivalent="<<same<<", max_delta="<<max_delta<<", error="<<max_error[1]<<"\n"<<std::flush;
    }
    return all_equivalent && all_accurate?0:2;
}
