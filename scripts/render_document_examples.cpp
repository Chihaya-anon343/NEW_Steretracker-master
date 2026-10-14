// Render measured stereo correspondences and existing template annotations.
// Documentation only: does not run detection, solve pose, or modify input files.
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
namespace fs=std::filesystem;
struct Pair { int id,region; cv::Point2f left,right; };
const cv::Scalar white(245,245,245), dark(28,28,28), c0(30,190,250), c1(100,255,100);
void label(cv::Mat& im,const std::string& text,cv::Point p,double scale=.65,cv::Scalar color=white) {
    cv::putText(im,text,p,cv::FONT_HERSHEY_SIMPLEX,scale,color,1,cv::LINE_AA);
}
cv::Mat load(const fs::path& p) {auto im=cv::imread(p.string());if(im.empty())throw std::runtime_error("Cannot load "+p.string());return im;}
std::vector<cv::Point2f> corners(const fs::path& p) {
    std::ifstream in(p);if(!in)throw std::runtime_error("Cannot read "+p.string());
    std::string line;std::vector<cv::Point2f> points;
    while(std::getline(in,line)) if(line.rfind("Corner_",0)==0) {
        auto value=line.substr(line.find(':')+1);std::replace(value.begin(),value.end(),',',' ');
        std::istringstream s(value);cv::Point2f pt;if(s>>pt.x>>pt.y)points.push_back(pt);
    }
    return points;
}
std::vector<Pair> pairs(const fs::path& p) {
    std::ifstream in(p);if(!in)throw std::runtime_error("Cannot read "+p.string());
    std::string line;std::getline(in,line);std::vector<Pair> out;
    while(std::getline(in,line)) {
        std::istringstream s(line);std::vector<std::string> v;std::string value;
        while(std::getline(s,value,','))v.push_back(value);
        if(v.size()!=12)throw std::runtime_error("Unexpected CSV schema");
        out.push_back({std::stoi(v[0]),std::stoi(v[6]),{std::stof(v[1]),std::stof(v[2])},{std::stof(v[3]),std::stof(v[4])}});
    }
    return out;
}
void numbered(cv::Mat& im,const std::vector<cv::Point2f>& points,cv::Point2f offset,float scale,cv::Scalar col) {
    for(size_t i=0;i<points.size();++i) {
        cv::Point2f p=(points[i]-offset)*scale;
        cv::circle(im,p,4,col,2,cv::LINE_AA);
        label(im,std::to_string(i),p+cv::Point2f(7,-6),.55,col);
    }
}
void templates(const fs::path& out) {
    auto board=load("data/clean_target2.png");auto a=corners("scripts/class0_points.txt"),b=corners("scripts/class1_points.txt");
    numbered(board,a,{0,0},1,c0);
    cv::rectangle(board,{337,399,61,60},c1,2);
    label(board,"class1",{404,396},.6,c1);
    cv::Mat panel(830,1270,CV_8UC3,dark);board.copyTo(panel(cv::Rect(15,55,board.cols,board.rows)));
    label(panel,"Board image + manual indices",{15,30},.85);
    auto detail=load("data/clean_target2.png")(cv::Rect(337,399,61,60)).clone();
    cv::resize(detail,detail,{427,420},0,0,cv::INTER_NEAREST);
    numbered(detail,b,{337,399},7,c1);detail.copyTo(panel(cv::Rect(807,95,427,420)));
    label(panel,"class1 detail (display x7)",{807,75},.67,c1);
    label(panel,"Orange: class0 board indices",{807,565},.6,c0);
    label(panel,"Green: class1 board indices",{807,599},.6,c1);
    label(panel,"Source board: 753 x 756 px",{807,650},.6);
    label(panel,"Configured board: 400 x 400 mm",{807,681},.6);
    label(panel,"Index sets are separate",{807,729},.6);
    label(panel,"Local fallback model is NOT this board",{807,760},.51);
    cv::imwrite((out/"board_template_indices.png").string(),panel);
    cv::Mat gallery(720,1170,CV_8UC3,dark);
    label(gallery,"BC / TT orientation masks: 24 angles, 15 degree steps",{18,30},.8);
    for(int i=0;i<24;++i) {
        auto mask=load(fs::path("data/NewMuBan(reordered)")/(std::to_string(i*15)+"_degrees.png"));
        cv::resize(mask,mask,{130,130},0,0,cv::INTER_NEAREST);
        int x=25+(i%6)*194,y=60+(i/6)*164;
        mask.copyTo(gallery(cv::Rect(x,y,130,130)));label(gallery,std::to_string(i*15)+" deg",{x,y+150},.6);
    }
    cv::imwrite((out/"orientation_template_gallery.png").string(),gallery);
    cv::Mat zero(600,600,CV_8UC3,dark);auto mask=load("data/NewMuBan(reordered)/0_degrees.png");
    cv::resize(mask,mask,{450,450},0,0,cv::INTER_NEAREST);mask.copyTo(zero(cv::Rect(65,65,450,450)));
    auto pts=corners("data/NewMuBan(reordered)/0_degrees.txt");
    numbered(zero,pts,{-65.f/9,-65.f/9},9,c0);
    label(zero,"0-degree mask + .txt corner indices",{25,30},.72);
    label(zero,"Original mask 50 x 50 px; display x9",{25,558},.65);
    cv::imwrite((out/"bc_template_zero_indices.png").string(),zero);
}
void stereo(int frame,const fs::path& out,int region=-1) {
    std::ostringstream n;n<<std::setw(6)<<std::setfill('0')<<frame;
    auto l=load("data/stereo_crop/left_"+n.str()+".png"),r=load("data/stereo_crop/right_"+n.str()+".png");
    std::ostringstream directory;directory<<"output/document_visuals_20261014/frame"<<std::setw(3)<<std::setfill('0')<<frame;
    auto ps=pairs(fs::path(directory.str())/"disparity_f1.csv");
    if(region>=0)ps.erase(std::remove_if(ps.begin(),ps.end(),[&](const Pair& p){return p.region!=region;}),ps.end());
    std::vector<cv::Point2f> all;for(const auto& p:ps){all.push_back(p.left);all.push_back(p.right);}
    auto box=cv::boundingRect(all);int margin=region>=0?20:(frame==125?25:50);
    box=cv::Rect(box.x-margin,box.y-margin,box.width+margin*2,box.height+margin*2)&cv::Rect(0,0,std::min(l.cols,r.cols),std::min(l.rows,r.rows));
    double scale=std::min(8.,700./box.width);int w=std::lround(box.width*scale),h=std::lround(box.height*scale);
    cv::Mat a,b;cv::resize(l(box),a,{w,h},0,0,cv::INTER_NEAREST);cv::resize(r(box),b,{w,h},0,0,cv::INTER_NEAREST);
    cv::Mat panel(h+105,w*2+45,CV_8UC3,dark);a.copyTo(panel(cv::Rect(15,70,w,h)));b.copyTo(panel(cv::Rect(w+30,70,w,h)));
    label(panel,"LOCAL Docker debug | data frame "+std::to_string(frame)+(region>=0?" | class"+std::to_string(region):"")+" | measured LK | disparity only",{15,26},.63);
    label(panel,"Left",{15,55},.6);label(panel,"Right",{w+30,55},.6);
    for(const auto& p:ps) {
        cv::Point2f x=(p.left-cv::Point2f(box.x,box.y))*scale+cv::Point2f(15,70);
        cv::Point2f y=(p.right-cv::Point2f(box.x,box.y))*scale+cv::Point2f(w+30,70);
        const cv::Scalar palette[]={ {80,80,255},{80,255,80},{255,140,80},{80,235,245},
                                    {220,80,230},{230,230,80},{80,160,255},{180,255,180},{255,190,180},{220,220,255} };
        auto col=frame==1?(p.region==0?c0:c1):palette[p.id%10];
        cv::line(panel,x,y,col,1,cv::LINE_AA);
        cv::circle(panel,x,6,cv::Scalar(0,0,0),3);cv::circle(panel,y,6,cv::Scalar(0,0,0),3);
        cv::circle(panel,x,4,col,2);cv::circle(panel,y,4,col,2);
        if(frame==1&&region<0&&p.region==1) {
            if(p.id==10){label(panel,"class1",x+cv::Point2f(25,-30),.5,col);label(panel,"class1",y+cv::Point2f(25,-30),.5,col);}
            continue;
        }
        std::string id=frame==1?(std::to_string(p.region)+":"+std::to_string(p.id%10)):std::to_string(p.id);
        cv::putText(panel,id,x+cv::Point2f(5,-7),cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,0,0),3,cv::LINE_AA);
        cv::putText(panel,id,y+cv::Point2f(5,-7),cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,0,0),3,cv::LINE_AA);
        label(panel,id,x+cv::Point2f(5,-7),.5,col);label(panel,id,y+cv::Point2f(5,-7),.5,col);
    }
    std::ostringstream note;note<<"Same full-image crop: ("<<box.x<<","<<box.y<<","<<box.width<<","<<box.height<<") | display x"<<std::fixed<<std::setprecision(2)<<scale;
    label(panel,note.str(),{15,h+94},.56);
    cv::imwrite((out/("stereo_frame"+std::to_string(frame)+(region>=0?"_class"+std::to_string(region):"")+"_detail.png")).string(),panel);
}
int main(int argc,char** argv) {
    try {
        fs::path out=argc>1?argv[1]:"docs/assets/algorithm_v1_5";fs::create_directories(out);
        templates(out);for(int f:{1,29,125})stereo(f,out);stereo(1,out,1);
        std::cout<<"Rendered template figures and 3 measured stereo details\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
