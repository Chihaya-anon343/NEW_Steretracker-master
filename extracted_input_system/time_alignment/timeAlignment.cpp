#include "timeAlignment.hpp"
#include <cmath>
#include <numeric>

TimeSyncUnit::TimeSyncUnit(double radar_alpha, double radar_max_dt)
    : radar_alpha_(radar_alpha), radar_max_dt_(radar_max_dt) {}

void TimeSyncUnit::addIMU(double t, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro) {
    
    imu_buf_.push_back({t, acc, gyro, Eigen::Vector3d::Zero()});
    
    // 清理旧数据，保留最近2秒内的IMU数据以防止内存泄漏
    while (!imu_buf_.empty() && (t - imu_buf_.front().t > 2.0)) {
        imu_buf_.pop_front();
    }
}

void TimeSyncUnit::addRadar(double t, double height) {
    radar_buf_.push_back({t, height});
    
    // 清楚旧数据
    while (!radar_buf_.empty()) {
        double t_old = radar_buf_.front().first;
        if ((t - t_old) > radar_max_dt_ * 2) {
            radar_buf_.pop_front();
        } else {
            break;
        }
    }
}

std::vector<IMUSample> TimeSyncUnit::getIMUUntil(double t_cam) {
    std::vector<IMUSample> data;
    

    while (imu_buf_.size() >= 2 && imu_buf_[1].t <= t_cam) {
        data.push_back(imu_buf_.front());
        imu_buf_.pop_front();
    }
    

    if (!imu_buf_.empty()) {
        
        const auto& sample = imu_buf_.front();
        double dt_tail = t_cam - sample.t;
        

        if (dt_tail > 0) {
            data.push_back(sample);
            imu_buf_.pop_front();
        }
    }
    
    return data;
}

bool TimeSyncUnit::getInterpolatedIMU(double t_cam, IMUSample& synced_imu) {
    if (imu_buf_.size() < 2) return false;

    // 寻找 t_cam 前后的两个 IMU 帧
    // imu_buf_ 是有序的，找到第一个 t >= t_cam 的位置
    auto it = std::upper_bound(imu_buf_.begin(), imu_buf_.end(), t_cam, 
        [](double val, const IMUSample& sample) {
            return val < sample.t;
        });

    // 如果所有数据都比 t_cam 小，或者所有数据都比 t_cam 大（不可能，因为上面的循环），则无法插值
    if (it == imu_buf_.end() || it == imu_buf_.begin()) {
        return false; 
    }

    const IMUSample& next = *it;
    const IMUSample& prev = *(--it);

    double dt = next.t - prev.t;
    if (dt <= 0) return false; // 异常情况

    double alpha = (t_cam - prev.t) / dt;

    synced_imu.t = t_cam;
    // 线性插值公式
    // d = d0 + alpha * (d1 - d0)
    synced_imu.acc = prev.acc + alpha * (next.acc - prev.acc);
    synced_imu.gyro = prev.gyro + alpha * (next.gyro - prev.gyro);
    synced_imu.angle = prev.angle + alpha * (next.angle - prev.angle);

    return true;
}

bool TimeSyncUnit::getSyncedRadarHeight(double t_cam, double& height_out) {
    if (radar_buf_.empty()) return false;
    
    std::vector<double> weights;
    std::vector<double> heights;
    
    for (const auto& item : radar_buf_) {
        double t_r = item.first;
        double h = item.second;
        
        double dt = std::abs(t_cam - t_r);
        if (dt > radar_max_dt_) continue;
        
        double w = std::exp(-radar_alpha_ * dt);
        weights.push_back(w);
        heights.push_back(h);
    }
    
    if (weights.empty()) return false;
    
    double sum_w = 0.0;
    double sum_wh = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum_w += weights[i];
        sum_wh += weights[i] * heights[i];
    }
    
    if (sum_w == 0.0) return false;
    
    height_out = sum_wh / sum_w;
    return true;
}
