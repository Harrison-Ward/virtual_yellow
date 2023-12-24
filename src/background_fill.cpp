#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <cmath>
#include "fill_functions.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;

int main(int argc, char *argv[])
{
    // define the path where the background samples live
    string background_input_path = "../images/background";

    // define the path where the test images live
    string test_input_path = "../images/test";

    // define output path
    string output_path = "../images/test_output";

    // define color sample vector
    vector<Vec3d> avg_color_samples;

    // define input image vectors
    vector<Mat> input_images;

    // loop over the images in the images folder
    for (const auto &entry : fs::directory_iterator(background_input_path))
    {
        if (entry.path().extension() == ".png")
        {
            // read in the image
            Mat sample_image = imread(entry.path().string());

            // check if image loaded properly
            if (sample_image.empty())
            {
                cerr << "Error reading: " << entry.path().string() << endl;
                continue; // Skip this iteration
            }

            // store the image
            input_images.push_back(sample_image);

            // calculate the mean color value of the image
            Scalar avg_color_scalar(mean(sample_image));

            // scale the average color vector
            cv::Vec3d avg_color_vector(
                avg_color_scalar[0], // Blue
                avg_color_scalar[1], // Green
                avg_color_scalar[2]  // Red
            );

            // store the average vector
            avg_color_samples.push_back(avg_color_vector);
        }
    }

    // define a reference color vector
    Vec3d reference_vector(avg_color_samples[0]);

    // calculate pixel color similarity threshold
    double threshold;
    threshold = cosine_similarity(avg_color_samples[0], avg_color_samples[1]);

    // store output vectors
    vector<Mat> output_images;

    // keep track of image index
    int image_number = 0;

    // loop over images, leave dis-similar pixels black and keep similar pixels
    for (const auto &entry : fs::directory_iterator(test_input_path))
    {
        // read in the test image
        Mat image = imread(entry.path().string());

        // check that the image corectly loaded
        if (image.empty())
        {
            cerr << "Error reading: " << entry.path().string() << endl;
            continue; // Skip this iteration
        }

        // define blank output image of same shape and size as input image
        Mat output_image(image.rows, image.cols, CV_8UC3, Vec3b(0, 0, 0));

        // def an earlier stopping criteria to kill search early when no green pixels are found
        int pixel_count = (image.cols * image.rows);
        double momentum = 0.10 * pixel_count;

        // begin search from the bottom of the image 
        for (int y = image.rows; y > 0; --y)
        {
            if (momentum < 0)
            {
                continue;
            }

            for (int x = image.cols; x > 0; --x)
            {
                // take cosine similarity of this pixel and sample vector
                Vec3b pixel(image.at<Vec3b>(y, x));
                Vec3d pixel_d(pixel[0], pixel[1], pixel[2]);
                
                if (cosine_similarity(reference_vector, pixel_d) > threshold * 1)
                {
                    output_image.at<Vec3b>(y, x) = pixel;
                    momentum++;
                }
                
                else 
                {
                    momentum--; 
                }
            }
        }
        
        // write test image to file
        string image_index = to_string(image_number);
        string file_name("/test_file_" + image_index + ".png");
        string file_output_path(output_path + file_name);

        // write out file path
        cout << "Image " << image_number << ": " << entry.path().string() << endl;

        imwrite(file_output_path, output_image);
        image_number++;
    }
}