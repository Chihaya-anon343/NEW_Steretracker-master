// Lossless decoded-pixel cache for repeated experiments on the Windows bind.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    std::ifstream input(argv[1]);
    std::ofstream output(argv[3]);
    if (!input || !output) return 3;
    output << "source,cached,rows,cols,decoded_fnv1a\n";
    std::string source;
    int count = 0;
    while (std::getline(input, source)) {
        if (source.empty()) continue;
        const std::filesystem::path p(source);
        const auto destination = std::filesystem::path(argv[2]) / p.parent_path().filename() /
                                 (p.stem().string() + ".bmp");
        std::filesystem::create_directories(destination.parent_path());
        const cv::Mat image = cv::imread(source, cv::IMREAD_COLOR);
        if (image.empty() || !cv::imwrite(destination.string(), image)) return 4;
        const cv::Mat check = cv::imread(destination.string(), cv::IMREAD_COLOR);
        if (check.size() != image.size() || check.type() != image.type() ||
            cv::norm(check, image, cv::NORM_INF) != 0) return 5;
        uint64_t hash = 14695981039346656037ULL;
        for (int y = 0; y < image.rows; ++y)
            for (size_t x = 0; x < size_t(image.cols) * image.elemSize(); ++x)
                hash = (hash ^ image.ptr<unsigned char>(y)[x]) * 1099511628211ULL;
        output << source << ',' << destination.generic_string() << ',' << image.rows << ','
               << image.cols << ',' << std::hex << hash << std::dec << '\n';
        if (++count % 100 == 0) std::cout << "CACHED " << count << std::endl;
    }
    return input.eof() && output.good() ? 0 : 6;
}
