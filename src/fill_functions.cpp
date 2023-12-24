#include <opencv2/opencv.hpp>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>

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