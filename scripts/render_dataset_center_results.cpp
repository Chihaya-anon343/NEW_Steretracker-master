#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

std::string fmt(double x){std::ostringstream s;s<<std::fixed<<std::setprecision(1)<<x;return s.str();}
void text(cv::Mat& m,std::string s,cv::Point p,cv::Scalar c={230,230,230},double size=.5){cv::putText(m,s,p,cv::FONT_HERSHEY_SIMPLEX,size,c,1,cv::LINE_AA);}
int main(int argc,char**argv){
    if(argc!=2)return 2;std::string root=argv[1];
    cv::FileStorage fs(root+"/results.json",cv::FileStorage::READ);
    cv::Mat gallery(1560,1980,CV_8UC3,cv::Scalar(24,24,24));
    cv::Mat space(1050,1800,CV_8UC3,cv::Scalar(24,24,24));int index=0;
    for(auto sample:fs["samples"]){
        std::string id=std::string(sample["id"]);
        cv::Mat im=cv::imread(root+"/"+id+"/target_detail.png");if(im.empty())return 3;
        double scale=std::min(660.0/im.cols,780.0/im.rows);cv::resize(im,im,{},scale,scale,cv::INTER_AREA);
        im.copyTo(gallery(cv::Rect((index%3)*660+(660-im.cols)/2,(index/3)*780,im.cols,im.rows)));
        cv::Mat tile(525,600,CV_8UC3,cv::Scalar(24,24,24));
        text(tile,id+" | camera X-Z plane [mm]",{18,27});
        if(int(sample["class0"]["success"])&&int(sample["class1"]["success"])){
            cv::Vec3d t[2];cv::Matx33d R[2];
            for(int cls=0;cls<2;++cls){auto row=sample[cls?"class1":"class0"];for(int i=0;i<3;++i)t[cls][i]=double(row["t_mm"][i]);for(int i=0;i<9;++i)R[cls].val[i]=double(row["R"][i]);}
            double xmin=std::min(t[0][0],t[1][0])-70,xmax=std::max(t[0][0],t[1][0])+70;
            double zmin=std::min(t[0][2],t[1][2])-70,zmax=std::max(t[0][2],t[1][2])+70;
            double s=std::min(430/(xmax-xmin),350/(zmax-zmin)),xc=(xmin+xmax)/2,zc=(zmin+zmax)/2;
            auto map=[&](cv::Vec3d p){return cv::Point(330+int((p[0]-xc)*s),275-int((p[2]-zc)*s));};
            for(int k=0;k<=4;++k){double z=zmin+(zmax-zmin)*k/4;int y=map({xc,0,z}).y;cv::line(tile,{70,y},{570,y},{65,65,65});text(tile,fmt(z),{8,y+5},{170,170,170},.4);}
            cv::line(tile,map(t[0]),map(t[1]),{255,0,255},2,cv::LINE_AA);
            for(int cls=0;cls<2;++cls){
                auto o=map(t[cls]);cv::circle(tile,o,5,cls?cv::Scalar(0,230,255):cv::Scalar(255,230,0),-1,cv::LINE_AA);
                cv::Vec3d a[]={{55,0,0},{0,55,0},{0,0,-55}};cv::Scalar c[]={{0,0,255},{0,255,0},{255,100,20}};
                for(int j=0;j<3;++j)cv::arrowedLine(tile,o,map(t[cls]+R[cls]*a[j]),c[j],2,cv::LINE_AA,0,.18);
                text(tile,"C"+std::to_string(cls)+" X="+fmt(t[cls][0])+" Z="+fmt(t[cls][2]),o+cv::Point(8,18),{230,230,230},.38);
            }
            text(tile,"Z ^",{8,65});text(tile,"X ->",{500,480});
            text(tile,"3D gap "+fmt(double(sample["distance_mm"]))+" mm",{18,508},{230,230,230},.55);
        }
        tile.copyTo(space(cv::Rect(index%3*600,index/3*525,600,525)));++index;
    }
    cv::imwrite(root+"/gallery.png",gallery);cv::imwrite(root+"/camera_space_gallery.png",space);return 0;
}
