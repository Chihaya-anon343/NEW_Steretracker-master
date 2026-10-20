#include "pose/GPnPSolver.hpp"
#include "common/GeometryUtils.hpp"
#include <unsupported/Eigen/NonLinearOptimization>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace gpnp {
namespace {
bool validRotation(const Eigen::Matrix3d& r) {
    return r.allFinite() && (r.transpose()*r-Eigen::Matrix3d::Identity()).norm() <= 1e-3 &&
           std::abs(r.determinant()-1.0) <= 1e-3;
}
bool finitePoint(const cv::Point2f& p) { return std::isfinite(p.x) && std::isfinite(p.y); }
const char* initName(PoseInitializationSource s) {
    switch(s) {
        case PoseInitializationSource::InitialPnP: return "InitialPnP";
        case PoseInitializationSource::DepthGuess: return "DepthGuess";
        case PoseInitializationSource::PreviousPose: return "PreviousPose";
        case PoseInitializationSource::StereoTriangulation: return "StereoTriangulation";
        case PoseInitializationSource::PlanarIppe: return "PlanarIppe";
        default: return "Provided";
    }
}
const char* stopName(int s) {
    switch(s) {
        case 1: return "relative_cost_tolerance";
        case 2: return "relative_step_tolerance";
        case 3: return "cost_and_step_tolerance";
        case 4: return "gradient_tolerance";
        case 5: return "function_budget_exhausted";
        case 6: return "cost_precision_limit";
        case 7: return "step_precision_limit";
        case 8: return "gradient_precision_limit";
        case 9: return "evaluation_aborted";
        case 0: return "improper_input_parameters";
        default: return "unexpected_optimizer_state";
    }
}
// SO(3) exponential with a Taylor branch at zero; no quaternion scale degree of freedom.
bool rotationIncrement(const Eigen::Vector3d& v, Eigen::Matrix3d& r) {
    const double theta=v.norm();
    if(!v.allFinite() || !std::isfinite(theta))return false;
    Eigen::Matrix3d k;
    k << 0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0;
    double a,b;
    if(theta<1e-6) {
        const double t2=theta*theta;
        a=1-t2/6+t2*t2/120;
        b=.5-t2/24+t2*t2/720;
    } else {
        a=std::sin(theta)/theta;
        b=(1-std::cos(theta))/(theta*theta);
    }
    r=Eigen::Matrix3d::Identity()+a*k+b*k*k;
    return validRotation(r);
}
}

struct GPNPCostFunctor {
    typedef double Scalar;
    typedef Eigen::VectorXd InputType;
    typedef Eigen::VectorXd ValueType;
    typedef Eigen::MatrixXd JacobianType;
    enum {InputsAtCompileTime=Eigen::Dynamic, ValuesAtCompileTime=Eigen::Dynamic};
    const Eigen::MatrixX3d& points;
    const Eigen::MatrixX3d& left_dirs;
    const Eigen::MatrixX3d& right_dirs;
    const Eigen::Vector3d right_origin;
    const Eigen::Matrix3d initial_rotation;
    const Eigen::Vector3d initial_translation;
    mutable bool invalid_evaluation{false};
    int inputs()const{return 6;}
    int values()const{return 6*static_cast<int>(points.rows())+3;}
    int operator()(const InputType& x,ValueType& f)const {
        Eigen::Matrix3d delta;
        if(x.size()!=6 || !x.allFinite() || !rotationIncrement(x.head<3>(),delta)) {
            invalid_evaluation=true; return -1; // Never return zero residual for invalid parameters.
        }
        const Eigen::Matrix3d r=delta*initial_rotation;
        const Eigen::Vector3d t=x.tail<3>();
        f.resize(values());
        for(int i=0;i<points.rows();++i) {
            const Eigen::Vector3d p=r*points.row(i).transpose()+t;
            f.segment<3>(6*i)=p.cross(left_dirs.row(i).transpose());
            f.segment<3>(6*i+3)=(p-right_origin).cross(right_dirs.row(i).transpose());
        }
        f.tail<3>()=(t-initial_translation).cwiseProduct(Eigen::Vector3d(.5,.5,2));
        if(!f.allFinite()){invalid_evaluation=true;return -1;}
        return 0;
    }
    int df(const InputType& x,JacobianType& j)const {
        j.resize(values(),inputs());
        InputType xp=x,xm=x;
        ValueType fp,fm;
        for(int c=0;c<inputs();++c) {
            const double h=std::sqrt(std::numeric_limits<double>::epsilon())*std::max(1.0,std::abs(x(c)));
            xp(c)=x(c)+h;xm(c)=x(c)-h;
            if((*this)(xp,fp)<0 || (*this)(xm,fm)<0)return -1;
            j.col(c)=(fp-fm)/(2*h);
            xp(c)=xm(c)=x(c);
        }
        if(!j.allFinite()){invalid_evaluation=true;return -1;}
        return 0;
    }
};

GPnPSolver::GPnPSolver(const StereoCameraParams& params,int min_pts)
    :params_(params),gpnp_min_pts_(min_pts){}
GPnPSolver::~GPnPSolver()=default;

PoseEstimate GPnPSolver::solve(PipelineResult& result,const std::vector<Eigen::Vector3d>& model,
                               const Eigen::Matrix3d* r_init,const Eigen::Vector3d* t_init,
                               double& timing,const GPnPOptions& options) {
    const auto start=std::chrono::steady_clock::now();
    monitor_=GPNPMonitor{};
    monitor_.gpnp_min_pts=gpnp_min_pts_;
    monitor_.opt_status=-2;monitor_.opt_message="not_started";
    monitor_.initial_source=(r_init || t_init)?initName(options.initial_source):"DepthGuess";
    PoseEstimate pose;
    auto finish=[&]() {
        timing=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        result.timing["gpnp"]=timing;monitor_.timing_ms=timing;
        monitor_.pose_accepted=pose.success;
        result.last_gpnp_monitor=monitor_;
        if(!pose.success)printMonitor();
        return pose;
    };
    auto fail=[&](const std::string& reason) {
        pose=PoseEstimate{};monitor_.failure_reason=reason;return finish();
    };
    if(gpnp_min_pts_<3 || options.max_function_evaluations<1 ||
       !std::isfinite(options.max_mean_reprojection_px) || options.max_mean_reprojection_px<=0 ||
       !std::isfinite(options.max_point_reprojection_px) || options.max_point_reprojection_px<=0)
        return fail("invalid_solver_options");
    if(!params_.K.allFinite() || !params_.K_inv.allFinite() || params_.K(0,0)<=0 || params_.K(1,1)<=0 ||
       (params_.K*params_.K_inv-Eigen::Matrix3d::Identity()).norm()>1e-6 ||
       (params_.K.row(2)-Eigen::RowVector3d(0,0,1)).norm()>1e-6 ||
       !validRotation(params_.R_rl) || !params_.t_rl.allFinite() || params_.t_rl.norm()<1e-9 ||
       !std::isfinite(params_.baseline) || params_.baseline<=0 ||
       !std::isfinite(params_.focal_length) || params_.focal_length<=0 ||
       std::abs(params_.baseline-params_.t_rl.norm())>1e-6*params_.baseline ||
       std::abs(params_.focal_length-params_.K(0,0))>1e-6*params_.focal_length)
        return fail("invalid_camera_calibration");
    if((r_init==nullptr)!=(t_init==nullptr))return fail("incomplete_initial_pose");
    if(r_init && (!validRotation(*r_init) || !t_init->allFinite()))return fail("invalid_initial_pose");
    monitor_.n_good_matches=static_cast<int>(result.good_matches.size());
    monitor_.n_idx_from_filtered=static_cast<int>(result.idx_from_filtered.size());
    monitor_.n_stereo_matched=static_cast<int>(result.pts_left_good.size());
    if(result.good_matches.empty())return fail("No template matches");
    if(result.idx_from_filtered.empty())return fail("No stereo-matched points");
    if(result.pts_left_good.size()!=result.pts_right_good.size() ||
       result.pts_left_good.size()!=result.idx_from_filtered.size())
        return fail("stereo_point_count_mismatch");
    if(result.pts_left_match.empty() || model.empty())return fail("empty_model_or_left_points");
    for(const auto& p:model)if(!p.allFinite())return fail("nonfinite_model_point");
    std::vector<int> lut(result.pts_left_match.size(),-1);
    std::vector<bool> used_model(model.size(),false),used_left(lut.size(),false);
    for(const auto& m:result.good_matches) {
        if(m.queryIdx<0 || m.queryIdx>=static_cast<int>(lut.size()) ||
           m.trainIdx<0 || m.trainIdx>=static_cast<int>(model.size()))
            return fail("invalid_template_match_index");
        if(lut[m.queryIdx]>=0 || used_model[m.trainIdx])return fail("duplicate_template_correspondence");
        if(!finitePoint(result.pts_left_match[m.queryIdx]))return fail("nonfinite_left_point");
        lut[m.queryIdx]=m.trainIdx;used_model[m.trainIdx]=true;
    }
    monitor_.n_query_to_train=monitor_.n_good_matches;
    std::vector<int> keep;
    for(int i=0;i<static_cast<int>(result.idx_from_filtered.size());++i) {
        const int q=result.idx_from_filtered[i];
        if(q<0 || q>=static_cast<int>(lut.size()))return fail("invalid_filtered_point_index");
        if(used_left[q])return fail("duplicate_filtered_point_index");
        used_left[q]=true;
        if(!finitePoint(result.pts_left_good[i]) || !finitePoint(result.pts_right_good[i]))
            return fail("nonfinite_stereo_point");
        if(lut[q]<0){++monitor_.n_missing_query;continue;}
        keep.push_back(i);
    }
    const int n=static_cast<int>(keep.size());
    monitor_.n_matched=monitor_.n_stereo_matched;monitor_.n_intersection=monitor_.n_pts=n;
    if(n<gpnp_min_pts_)return fail("Intersection points insufficient");
    Eigen::MatrixX3d points(n,3),dl(n,3),dr(n,3);
    for(int k=0;k<n;++k) {
        const int i=keep[k],j=lut[result.idx_from_filtered[i]];
        points.row(k)=model[j].transpose();
        const auto& l=result.pts_left_good[i];const auto& r=result.pts_right_good[i];
        Eigen::Vector3d a=params_.K_inv*Eigen::Vector3d(l.x,l.y,1);
        Eigen::Vector3d b=params_.R_rl*params_.K_inv*Eigen::Vector3d(r.x,r.y,1);
        if(!a.allFinite() || !b.allFinite() || a.norm()<1e-12 || b.norm()<1e-12)return fail("invalid_observation_ray");
        dl.row(k)=a.normalized().transpose();dr.row(k)=b.normalized().transpose();
    }
    // Planar points are valid; collinear or coincident physical points are not.
    int far=0;double span2=0,cross2=0;
    for(int k=1;k<n;++k) {
        const double d=(points.row(k)-points.row(0)).squaredNorm();
        if(d>span2){span2=d;far=k;}
    }
    const Eigen::Vector3d axis=(points.row(far)-points.row(0)).transpose();
    for(int k=1;k<n;++k)cross2=std::max(cross2,axis.cross((points.row(k)-points.row(0)).transpose()).squaredNorm());
    if(span2<1e-12 || cross2<1e-12*span2*span2)return fail("degenerate_model_geometry");
    auto spread=[&](const std::vector<cv::Point2f>& pixels) {
        const auto origin=pixels[keep[0]];cv::Point2d axis2(0,0);double extent=0,area=0;
        for(int i:keep) {
            cv::Point2d v(double(pixels[i].x)-origin.x,double(pixels[i].y)-origin.y);
            const double d=v.dot(v);if(d>extent){extent=d;axis2=v;}
        }
        for(int i:keep) {
            const double c=axis2.x*(double(pixels[i].y)-origin.y)-axis2.y*(double(pixels[i].x)-origin.x);
            area=std::max(area,c*c);
        }
        return extent>1e-12 && area>1e-12*extent*extent;
    };
    if(!spread(result.pts_left_good)||!spread(result.pts_right_good))return fail("degenerate_image_geometry");
    monitor_.n_rays=2*n;
    std::vector<double> disparities;
    for(int i:keep) {
        const double d=result.pts_left_good[i].x-result.pts_right_good[i].x;
        if(std::isfinite(d))disparities.push_back(std::abs(d));
    }
    const double median=computeMedian(disparities);
    const double depth=median>1?std::clamp(params_.focal_length*params_.baseline/median,50.0,5000.0):500.0;
    monitor_.depth_guess=depth;
    const Eigen::Matrix3d r0=r_init?*r_init:Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t0=t_init?*t_init:Eigen::Vector3d(0,0,depth);
    if(r_init)for(int k=0;k<n;++k) {
        const Eigen::Vector3d pl=r0*points.row(k).transpose()+t0;
        const Eigen::Vector3d pr=params_.R_rl.transpose()*(pl-params_.t_rl);
        if(!pl.allFinite()||!pr.allFinite()||pl.z()<=1e-6||pr.z()<=1e-6)return fail("nonpositive_initial_depth");
    }
    GPNPCostFunctor functor{points,dl,dr,params_.t_rl,r0,t0};
    Eigen::VectorXd x(6);x.head<3>().setZero();x.tail<3>()=t0;
    Eigen::VectorXd residual;
    if(functor(x,residual)<0)return fail("invalid_initial_residual");
    monitor_.opt_initial_cost=residual.squaredNorm();
    if(!std::isfinite(monitor_.opt_initial_cost))return fail("nonfinite_initial_cost");
    Eigen::LevenbergMarquardt<GPNPCostFunctor,double> lm(functor);
    lm.parameters.ftol=lm.parameters.xtol=lm.parameters.gtol=1e-8;
    lm.parameters.maxfev=options.max_function_evaluations;
    const int status=static_cast<int>(lm.minimize(x));
    monitor_.opt_status=status;monitor_.opt_message=stopName(status);monitor_.opt_nfev=lm.nfev;
    Eigen::Matrix3d delta;
    monitor_.candidate_valid=x.allFinite() && rotationIncrement(x.head<3>(),delta) && !functor.invalid_evaluation;
    if(!monitor_.candidate_valid)return fail("invalid_optimizer_candidate");
    pose.R=delta*r0;pose.t=x.tail<3>();pose.num_points=n;
    if(functor(x,residual)<0)return fail("nonfinite_final_residual");
    monitor_.opt_final_cost=residual.squaredNorm();
    monitor_.opt_cost_reduction=monitor_.opt_initial_cost-monitor_.opt_final_cost;
    Eigen::Quaterniond q(pose.R);q.normalize();if(q.w()<0)q.coeffs()=-q.coeffs();
    monitor_.q_opt={q.x(),q.y(),q.z(),q.w()};monitor_.t_opt={pose.t.x(),pose.t.y(),pose.t.z()};
    if(status<1 || status>4)return fail("optimizer_"+monitor_.opt_message);
    if(!std::isfinite(monitor_.opt_final_cost) ||
       monitor_.opt_final_cost>monitor_.opt_initial_cost+1e-9*(1+monitor_.opt_initial_cost))
        return fail("invalid_final_cost");
    monitor_.opt_success=true;
    auto geometry=[&](const Eigen::Matrix3d& r,const Eigen::Vector3d& t,bool stereo) {
        monitor_.mean_left_px=monitor_.mean_right_px=monitor_.max_reprojection_px=0;
        if(!validRotation(r) || !t.allFinite()){monitor_.geometry_reason="invalid_pose";return false;}
        for(int k=0;k<n;++k) {
            const int i=keep[k];
            const Eigen::Vector3d pl=r*points.row(k).transpose()+t;
            const Eigen::Vector3d pr=params_.R_rl.transpose()*(pl-params_.t_rl);
            if(!pl.allFinite() || pl.z()<=1e-6 || (stereo && (!pr.allFinite() || pr.z()<=1e-6))) {
                monitor_.geometry_reason="nonpositive_point_depth";return false;
            }
            const Eigen::Vector3d ul=params_.K*pl,ur=params_.K*pr;
            const auto& l=result.pts_left_good[i];const auto& rr=result.pts_right_good[i];
            const double el=std::hypot(ul.x()/ul.z()-l.x,ul.y()/ul.z()-l.y);
            const double er=stereo?std::hypot(ur.x()/ur.z()-rr.x,ur.y()/ur.z()-rr.y):0;
            if(!std::isfinite(el)||!std::isfinite(er)){monitor_.geometry_reason="nonfinite_reprojection";return false;}
            monitor_.mean_left_px+=el;monitor_.mean_right_px+=er;
            monitor_.max_reprojection_px=std::max({monitor_.max_reprojection_px,el,er});
        }
        monitor_.mean_left_px/=n;monitor_.mean_right_px/=n;
        if(monitor_.mean_left_px>options.max_mean_reprojection_px ||
           monitor_.mean_right_px>options.max_mean_reprojection_px ||
           monitor_.max_reprojection_px>options.max_point_reprojection_px) {
            monitor_.geometry_reason="reprojection_error";return false;
        }
        monitor_.geometry_reason="accepted";return true;
    };
    if(geometry(pose.R,pose.t,true)) {
        pose.success=true;pose.source=PoseSource::StereoGpnp;return finish();
    }
    const std::string stereo_failure=monitor_.geometry_reason;
    // Only a known InitialPnP result can be reported as a left-PnP fallback.
    if(r_init && options.initial_source==PoseInitializationSource::InitialPnP && geometry(*r_init,*t_init,false)) {
        pose.R=*r_init;pose.t=*t_init;pose.success=true;pose.source=PoseSource::LeftPnpFallback;
        monitor_.used_initial_pose_fallback=true;monitor_.geometry_reason="left_initial_fallback:"+stereo_failure;
        return finish();
    }
    monitor_.geometry_reason=stereo_failure;
    return fail(stereo_failure);
}

void GPnPSolver::printMonitor() const {
    const std::string sep(70, '-');
    std::cout << "\n" << sep << "\n";
    std::cout << "  ⚠ GPNP Failed | Reason: "
              << (monitor_.failure_reason.empty() ? "Unknown" : monitor_.failure_reason) << "\n";
    std::cout << sep << "\n\n  [Data Flow Statistics]\n";
    std::cout << "    good_matches                      : " << monitor_.n_good_matches << "\n";
    std::cout << "    query_to_train (mapping)           : " << monitor_.n_query_to_train << "\n";
    std::cout << "    idx_from_filtered                  : " << monitor_.n_idx_from_filtered << "\n";
    std::cout << "    n_stereo_matched                   : " << monitor_.n_stereo_matched << "\n";
    std::cout << "    n_matched                          : " << monitor_.n_matched << "\n";
    std::cout << "    n_intersection                     : " << monitor_.n_intersection << "\n";
    std::cout << "      ├─ missing queryIdx              : " << monitor_.n_missing_query << "\n";
    std::cout << "      └─ missing trainIdx              : " << monitor_.n_missing_train << "\n";
    std::cout << "    n_pts (points for PnP)             : " << monitor_.n_pts << "\n";
    std::cout << "    gpnp_min_pts                       : " << monitor_.gpnp_min_pts << "\n";
    std::cout << "\n  [Optimizer Status]\n";
    std::cout << "    success         : " << (monitor_.opt_success ? "true" : "false") << "\n";
    std::cout << "    stop reason     : " << monitor_.opt_message << "\n";
    std::cout << "    initialization  : " << monitor_.initial_source << "\n";
    std::cout << "    candidate valid : " << monitor_.candidate_valid << "\n";
    std::cout << "    pose accepted   : " << monitor_.pose_accepted << "\n";
    std::cout << "    geometry        : " << monitor_.geometry_reason << "\n";
    std::cout << "    status          : " << monitor_.opt_status << "\n";
    std::cout << "    nfev            : " << monitor_.opt_nfev << "\n";
    std::cout << "    initial cost    : " << monitor_.opt_initial_cost << "\n";
    std::cout << "    final cost      : " << monitor_.opt_final_cost << "\n";
    if (!monitor_.q_opt.empty() && !monitor_.t_opt.empty()) {
        std::cout << "\n  [Optimization Result]\n";
        std::cout << "    q  : [" << monitor_.q_opt[0] << ", " << monitor_.q_opt[1]
                  << ", " << monitor_.q_opt[2] << ", " << monitor_.q_opt[3] << "]\n";
        std::cout << "    t  : [" << monitor_.t_opt[0] << ", " << monitor_.t_opt[1]
                  << ", " << monitor_.t_opt[2] << "]\n";
    }
    if (!monitor_.exception.empty())
        std::cout << "\n  [Exception] " << monitor_.exception << "\n";
    std::cout << "\n  GPNP Time: " << monitor_.timing_ms << " ms\n";
    std::cout << sep << "\n" << std::endl;
}

} // namespace gpnp
