#include <opencv2/opencv.hpp>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <future>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

double cosine_similarity(const cv::Vec3d& vec_a, const cv::Vec3d& vec_b)
{
    double dot_product = 0;
    double length_a = 0;
    double length_b = 0;

    for (int i = 0; i < 3; ++i)
    {
        // dot the i'th elements
        dot_product += vec_a[i] * vec_b[i];
        
        // add the square of the i'th elements
        length_a += pow(vec_a[i], 2);
        length_b += pow(vec_b[i], 2);
    }

    // Ensure the lengths are not zero to avoid division by zero
    if (length_a == 0 || length_b == 0) {
        // cerr << "One or both vectors are zero length." << endl;
        return 0;
    }

    return dot_product / (sqrt(length_a) * sqrt(length_b));
}

double euclidean_distance_3D(const cv::Vec3d& vec_a, const cv::Vec3d& vec_b)
{
    double distance = 0;

    // loop over the first element of both vectors
    for (int i =0; i <3; ++i)
    {
        // sum the squared distance
        distance += pow((vec_a[i] - vec_b[i]), 2);
    }

    // take the square root of the summed differences
    distance = sqrt(distance);

    return distance;
}

cv::Vec3d inverse_color_rgb(const cv::Vec3d& input_color)
{
    // return inverse color of the original input space
    cv::Vec3d inverse_color;

    for (int i = 0; i<3; i++)
    {
        // find opposite color value
        inverse_color[i] = 255 - input_color[i];
    }

    return inverse_color;
}

cv::Vec3d inverse_color_lab(const cv::Vec3d& input_color)
{
    // Inverse color in LAB color space
    cv::Vec3d inverse_color;

    // Invert the L component (range 0 to 100)
    inverse_color[0] = 100 - input_color[0];

    // Invert the a component (range -128 to 127)
    inverse_color[1] = -input_color[1];

    // Invert the b component (range -128 to 127)
    inverse_color[2] = -input_color[2];

    return inverse_color;
}

// Asynchronous write function
future<void> async_write_image(const std::string& path, const cv::Mat& image) {
    return std::async(std::launch::async, [&]() {
        imwrite(path, image);
    });
}