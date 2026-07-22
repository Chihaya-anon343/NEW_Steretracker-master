#ifndef CAN_SOCKET_HPP
#define CAN_SOCKET_HPP

#include <string>
#include <cstdint>
#include <cstring>
// #include "comm_struct.h"
// #include "comm_function.h"


namespace nanhang
{

struct CanFrame {
    uint32_t id;                    // CAN标识符
    uint8_t dlc;                    // 数据长度码（0-8）
    uint8_t data[8];                // 数据
    
    CanFrame() : id(0), dlc(0) {
        memset(data, 0, sizeof(data));
    }
    
    CanFrame(uint32_t can_id, const uint8_t* data_ptr = nullptr, uint8_t len = 0) 
        : id(can_id), dlc(len > 8 ? 8 : len) {
        memset(data, 0, sizeof(data));
        if (data_ptr && dlc > 0) {
            memcpy(data, data_ptr, dlc);
        }
    }
};
struct Target{
    int  target_id;   
    double hight;
    double angle;
};

class CanSocket {
public:
    CanSocket();
    ~CanSocket();
    

    CanSocket(const CanSocket&) = delete;
    CanSocket& operator=(const CanSocket&) = delete;
    
    bool canConnect(const std::string& interface = "can0");

    void disconnect();
    
    bool isConnected() const { return sockfd_ >= 0; }

    bool sendFrame(const CanFrame& frame);
    
    bool receiveFrame(CanFrame& frame, int timeout_ms = -1);
    // void GetHightOut(double hight, serialdate &ret);

    const char* getLastError() const { return error_msg_; }
    
private:
    int sockfd_;                    // Socket文件描述符
    std::string interface_;         // 接口名称
    char error_msg_[256];           // 错误信息
};
}
#endif // CAN_SOCKET_HPP
