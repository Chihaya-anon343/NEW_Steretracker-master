#include "can_socket.hpp"

#include <iostream>
#include <unistd.h>
#include <cerrno>
#include <cstring>

// Linux SocketCAN头文件
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <sys/select.h>
namespace nanhang
{
    CanSocket::CanSocket() : sockfd_(-1) {
        memset(error_msg_, 0, sizeof(error_msg_));
    }

    CanSocket::~CanSocket() {
        disconnect();
    }

    bool CanSocket::canConnect(const std::string& interface) {
        // 如果已经连接，先断开
        if (sockfd_ >= 0) {
            disconnect();
        }
        
        // 创建Socket
        sockfd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (sockfd_ < 0) {
            snprintf(error_msg_, sizeof(error_msg_), 
                    "Failed to create socket: %s", strerror(errno));
            return false;
        }
        
        interface_ = interface;
        
        // 获取接口索引
        struct ifreq ifr;
        memset(&ifr, 0, sizeof(ifr));
        strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);
        ifr.ifr_name[IFNAMSIZ - 1] = '\0';
        
        if (ioctl(sockfd_, SIOCGIFINDEX, &ifr) < 0) {
            snprintf(error_msg_, sizeof(error_msg_), 
                    "Failed to get interface index for %s: %s", 
                    interface.c_str(), strerror(errno));
            close(sockfd_);
            sockfd_ = -1;
            return false;
        }
        
        // 绑定Socket
        struct sockaddr_can addr;
        memset(&addr, 0, sizeof(addr));
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        
        if (bind(sockfd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            snprintf(error_msg_, sizeof(error_msg_), 
                    "Failed to bind socket to %s: %s", 
                    interface.c_str(), strerror(errno));
            close(sockfd_);
            sockfd_ = -1;
            return false;
        }
        
        std::cout << "Connected to CAN interface: " << interface << std::endl;
        return true;
    }

    void CanSocket::disconnect() {
        if (sockfd_ >= 0) {
            close(sockfd_);
            sockfd_ = -1;
            interface_.clear();
            std::cout << "Disconnected from CAN interface" << std::endl;
        }
    }

    bool CanSocket::sendFrame(const CanFrame& frame) {
        if (sockfd_ < 0) {
            snprintf(error_msg_, sizeof(error_msg_), "Not connected");
            return false;
        }
        
        // 转换为Linux CAN帧格式
        struct can_frame raw_frame;
        memset(&raw_frame, 0, sizeof(raw_frame));
        
        raw_frame.can_id = frame.id;
        raw_frame.can_dlc = frame.dlc;
        if (frame.dlc > 0) {
            memcpy(raw_frame.data, frame.data, frame.dlc);
        }
        
        // 发送数据
        int nbytes = write(sockfd_, &raw_frame, sizeof(struct can_frame));
        if (nbytes != sizeof(struct can_frame)) {
            snprintf(error_msg_, sizeof(error_msg_), 
                    "Failed to send frame: %s", strerror(errno));
            return false;
        }
        
        // 打印发送信息（可选）
        std::cout << "Sent: ID=0x" << std::hex << frame.id 
                << " DLC=" << std::dec << (int)frame.dlc;
        if (frame.dlc > 0) {
            std::cout << " Data:";
            for (int i = 0; i < frame.dlc; i++) {
                std::cout << " 0x" << std::hex << (int)frame.data[i];
            }
        }
        std::cout << std::endl;
        
        return true;
    }

    bool CanSocket::receiveFrame(CanFrame& frame, int timeout_ms)
    {
        if (sockfd_ < 0) {
            snprintf(error_msg_, sizeof(error_msg_), "Not connected");
            return false;
        }
        
        // 设置超时
        if (timeout_ms >= 0) {
            fd_set readfds;
            struct timeval tv;
            
            FD_ZERO(&readfds);
            FD_SET(sockfd_, &readfds);
            
            tv.tv_sec = timeout_ms / 1000;
            tv.tv_usec = (timeout_ms % 1000) * 1000;
            
            int ret = select(sockfd_ + 1, &readfds, NULL, NULL, &tv);
            if (ret <= 0) {
                if (ret == 0) {
                    snprintf(error_msg_, sizeof(error_msg_), "Receive timeout");
                } else {
                    snprintf(error_msg_, sizeof(error_msg_), 
                            "Select error: %s", strerror(errno));
                }
                return false;
            }
        }
        
        // 接收数据
        struct can_frame raw_frame;
        int nbytes = read(sockfd_, &raw_frame, sizeof(struct can_frame));
        if (nbytes <= 0) {
            snprintf(error_msg_, sizeof(error_msg_), 
                    "Failed to receive frame: %s", strerror(errno));
            return false;
        }
        
        // 转换为我们的格式
        frame.id = raw_frame.can_id;
        frame.dlc = raw_frame.can_dlc;
        if (frame.dlc > 0) {
            memcpy(frame.data, raw_frame.data, frame.dlc);
        }
        // // 打印接收信息（可选）
        // std::cout << "Received: ID=0x" << std::hex << frame.id 
        //         << " DLC=" << std::dec << (int)frame.dlc;
        // if (frame.dlc > 0) {
        //     std::cout << " Data:";
        //     for (int i = 0; i < frame.dlc; i++) {
        //         std::cout << " 0x" << std::hex << (int)frame.data[i];
        //     }
        // }
        // std::cout << std::endl;
        
        return true;
    }
    

}