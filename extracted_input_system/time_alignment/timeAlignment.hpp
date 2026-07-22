#pragma once
#include <deque>
#include <vector>
#include <Eigen/Core>

struct IMUSample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double t;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyro;
    Eigen::Vector3d angle;
};

class TimeSyncUnit {
public:
    TimeSyncUnit(double radar_alpha=50.0, double radar_max_dt=0.2);

    void addIMU(double t,
                const Eigen::Vector3d& acc,
                const Eigen::Vector3d& gyro);

    void addRadar(double t, double height);

    std::vector<IMUSample> getIMUUntil(double t_cam);

    // 利用线性插值获取指定时刻的IMU数据
    bool getInterpolatedIMU(double t_cam, IMUSample& synced_imu);

    bool getSyncedRadarHeight(double t_cam, double& height_out);

private:
    std::deque<IMUSample> imu_buf_;
    std::deque<std::pair<double,double>> radar_buf_;

    double radar_alpha_;
    double radar_max_dt_;
};
