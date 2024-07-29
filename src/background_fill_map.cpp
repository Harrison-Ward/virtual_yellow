#include <boost/filesystem.hpp>
#include <cmath>
#include "fill_functions.h"
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

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

            // convert the image to LAB
            Mat sample_image_lab;
            cv::cvtColor(sample_image, sample_image_lab, cv::COLOR_BGR2Lab);

            // store the image
            input_images.push_back(sample_image_lab);

            // calculate the mean color value of the image
            Scalar avg_color_scalar(mean(sample_image_lab));

            // scale the average color vector
            cv::Vec3d avg_color_vector(
                avg_color_scalar[0], // L
                avg_color_scalar[1], // a
                avg_color_scalar[2]  // b
            );

            // store the average vector
            avg_color_samples.push_back(avg_color_vector);
        }
    }

    // define a reference color vector
    Vec3d reference_vector(avg_color_samples[0]);

    // define an inverse vector to the sample input vector
    Vec3d inverse_vector;
    inverse_vector = inverse_color_lab(avg_color_samples[0]);

    // calculate distance between the input and inverse vector
    double max_distance;
    max_distance = euclidean_distance_3D(avg_color_samples[0], inverse_vector);

    // calculate color similarity threshold
    double threshold = 0.95;
    double threshold_distance = (1 - threshold) * max_distance;

    // store output vectors
    vector<Mat> output_images;

    // keep track of image index
    int image_number = 0;

    // define kernel size for averaging
    int kernel_size = 10; // You can change this value as needed

    // loop over images, leave dissimilar pixels black and keep similar pixels
    for (const auto &entry : fs::directory_iterator(test_input_path))
    {
        auto start_total = chrono::high_resolution_clock::now();

        // read in the test image
        auto start_read = chrono::high_resolution_clock::now();
        Mat image = imread(entry.path().string());
        auto end_read = chrono::high_resolution_clock::now();

        // check that the image correctly loaded
        if (image.empty())
        {
            cerr << "Error reading: " << entry.path().string() << endl;
            continue; // Skip this iteration
        }

        // convert the image to LAB
        auto start_convert = chrono::high_resolution_clock::now();
        Mat image_lab;
        cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
        auto end_convert = chrono::high_resolution_clock::now();

        // apply average pooling
        auto start_blur = chrono::high_resolution_clock::now();
        Mat averaged_image;
        blur(image_lab, averaged_image, Size(kernel_size, kernel_size));
        auto end_blur = chrono::high_resolution_clock::now();

        // define blank mask image
        auto start_mask = chrono::high_resolution_clock::now();
        Mat mask = Mat::zeros(averaged_image.size(), CV_8UC1);

        parallel_for_(Range(0, averaged_image.rows), [&](const Range& range) {
            for (int y = range.start; y < range.end; ++y) {
                Vec3b* row = averaged_image.ptr<Vec3b>(y);  // Get row pointer for efficient access
                uchar* mask_row = mask.ptr<uchar>(y);  // Get row pointer for mask
                for (int x = 0; x < averaged_image.cols; ++x) {
                    Vec3b pixel = row[x];
                    Vec3d pixel_d(pixel[0], pixel[1], pixel[2]);

                    if (euclidean_distance_3D(reference_vector, pixel_d) > threshold_distance) {
                        mask_row[x] = 255; // within threshold
                    }
                }
            }
        });
        auto end_mask = chrono::high_resolution_clock::now();


        // resize the mask to the original image size
        auto start_resize = chrono::high_resolution_clock::now();
        Mat resized_mask;
        resize(mask, resized_mask, image.size(), 0, 0, INTER_NEAREST);
        auto end_resize = chrono::high_resolution_clock::now();

        // apply the mask to the original image
        auto start_apply_mask = chrono::high_resolution_clock::now();

        // Apply the mask to the original image
        Mat output_image;
        image.copyTo(output_image, resized_mask);
        auto end_apply_mask = chrono::high_resolution_clock::now();

        // Write test image to file asynchronously
        auto start_write = chrono::high_resolution_clock::now();
        string image_index = to_string(image_number);
        string file_name("/test_file_" + image_index + ".png");
        string file_output_path = output_path + file_name;
        auto write_future = async_write_image(file_output_path, output_image);
        auto end_write = chrono::high_resolution_clock::now();

        auto end_total = chrono::high_resolution_clock::now();

        image_number++;

        // Write out file path
        cout << "Image " << image_number << ": " << entry.path().string() << endl;

        // Display time for each step
        cout << "Read time: " << chrono::duration_cast<chrono::milliseconds>(end_read - start_read).count() << " ms" << endl;
        cout << "Convert time: " << chrono::duration_cast<chrono::milliseconds>(end_convert - start_convert).count() << " ms" << endl;
        cout << "Blur time: " << chrono::duration_cast<chrono::milliseconds>(end_blur - start_blur).count() << " ms" << endl;
        cout << "Mask creation time: " << chrono::duration_cast<chrono::milliseconds>(end_mask - start_mask).count() << " ms" << endl;
        cout << "Resize time: " << chrono::duration_cast<chrono::milliseconds>(end_resize - start_resize).count() << " ms" << endl;
        cout << "Apply mask time: " << chrono::duration_cast<chrono::milliseconds>(end_apply_mask - start_apply_mask).count() << " ms" << endl;
        cout << "Write time: " << chrono::duration_cast<chrono::milliseconds>(end_write - start_write).count() << " ms" << endl;
        cout << "Total processing time: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms" << endl;
        cout << file_output_path << ": generated in: " << chrono::duration_cast<chrono::milliseconds>(end_total - start_total).count() << " ms\n" << endl;

        // Ensure the previous write operation is completed
        write_future.get();
}
}