#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"

#include <random>
#include <iostream>
#include <algorithm>

#include "include/nf.hpp"


namespace utils = nf::utils;


struct Random
{
    std::random_device dev;
    std::mt19937 rng;
    std::uniform_int_distribution<int> dist;

    Random(int low, int high)
    {
        rng = std::mt19937(dev());
        dist = std::uniform_int_distribution<int>(low, high);
    }
    

    int gen() { return dist(rng); }
};

/*
 * Creates the target image using pieces of the background path. The given mask 
 * tells the program which parts to focus on
 */
auto circle_img(const char* target_path, const char* background_path,
        const char* mask_path, int step=3)
{
    auto mask = cv::imread(mask_path);
    auto target = cv::imread(target_path);
    auto background = cv::imread(background_path);

    // radius of circles to use
    const std::vector<int> r = { 200, 100, 20, 10, 5 };
    // how many circles to use
    const std::vector<int> iters = { 20, 50, 500, 500, 500 };

    float sx = 1.;
    float sy = 1.;
    auto down_mask = nf::resize(mask, sx, sy);
    auto down_target = nf::resize(target, sx, sy);
    auto down_background = nf::resize(background, sx, sy);

    auto img = down_background.clone();


    const auto get_targets = [&down_target](const cv::Mat& mask, auto condition){
        // get pixels inside mask
        std::vector<std::vector<int>> pixels_pos;
        std::vector<cv::Vec3b> target_pixels;
        for (size_t y = 0; y < mask.rows; y++)
        {
            for (size_t x = 0; x < mask.cols; x++)
            {
                if (condition(mask, y, x))
                {
                    std::vector<int> v = {static_cast<int>(x), static_cast<int>(y)};
                    pixels_pos.push_back(v);
                    target_pixels.push_back(down_target.at<cv::Vec3b>(y, x));
                }
            }
        }
        return std::make_pair(pixels_pos, target_pixels);
    };

    auto place_circles = [&img, &step, &down_background, &r, &sx, &iters](
        const std::vector<std::vector<int>>& pixels_pos,
        const std::vector<cv::Vec3b>& target_pixels, int start, int end){
        // calculate background means for pixels inside the image
        for (size_t k = start; k < end; k++)
        {
            int cr = r[k] * sx;

            std::vector<std::vector<int>> sub_mask_pos;
            std::vector<std::vector<int>> means;

            #pragma omp parallel for
            for (size_t x = cr; x < down_background.cols - cr; x+=step * (k+1))
            {
                for (size_t y = cr; y < down_background.rows - cr; y+=step*(k+1))
                {
                    auto mask = nf::circle_mask(down_background, x, y, cr);
                    auto mean = nf::img_mean(down_background, mask);

                    means.push_back(mean);

                    std::vector<int> v = {(int)x, (int)y};
                    sub_mask_pos.push_back(v);
                }
            }

            Random random(0, target_pixels.size());
            #pragma omp parallel for
            for (size_t i = 0; i < iters[k]; i++)
            {
                int idx = random.gen();
                cv::Vec3b target_pixel = target_pixels[idx];
                std::vector<int> pixel_pos = pixels_pos[idx]; 
                std::vector<int> mask_pixel_pos = {0, 0};

                int min_avg = 255 * 3;
                
                for (size_t j = 0; j < means.size(); j++)
                {
                    std::vector<int> mean = means[j];
                    // calc distance between avgs
                    int avg = std::abs(mean[0] - target_pixel[0]) + 
                        std::abs(mean[1] - target_pixel[1]) + 
                        std::abs(mean[2] - target_pixel[2]);

                    if (avg < min_avg)
                    {
                        min_avg = avg;
                        mask_pixel_pos = sub_mask_pos[j];
                    } 
                }

                int x = mask_pixel_pos[0];
                int y = mask_pixel_pos[1];

                cv::Mat min_avg_img = nf::circle_mask(down_background, x, y, cr);
                min_avg_img = nf::translate(min_avg_img, pixel_pos[0] - x, pixel_pos[1] - y);

                img = nf::overlay(min_avg_img, img, min_avg_img != cv::Vec3b(0,0,0));
            }

            std::cout << "Finished k=" << k << std::endl;
        }
    };

    // get pixels inside mask
    auto pixels = get_targets(down_mask, [](const cv::Mat& mask, int y, int x){
        return mask.at<cv::Vec3b>(y, x) != cv::Vec3b(0,0,0); });

    place_circles(pixels.first, pixels.second, 0, r.size() - 1);

    // get pixels inside mask AND edges
    auto edges_mask = nf::get_edges_color(down_target, 150, down_mask, 5);
    pixels = get_targets(edges_mask, [](const cv::Mat& mask, int y, int x){
        return mask.at<uchar>(y, x) != 0; });
    place_circles(pixels.first, pixels.second, r.size() - 1, r.size());

    auto t_size = down_target.size();
    auto b_size = down_background.size();
    if (t_size.width < b_size.width && t_size.height < b_size.height)
        img = img(cv::Range(0, t_size.width), cv::Range(0, t_size.height));
    
    return img;
}


int main(int argc, char** argv)
{
    const char* mask_path = utils::parse_option("-m", "imgs/john_mask.jpg", argc, argv);
    const char* target_path = utils::parse_option("-t", "imgs/john.jpg", argc, argv);
    const char* background_path = utils::parse_option("-b", "imgs/galaxy.jpg", argc, argv);

    auto img = circle_img(target_path, background_path, mask_path, 5);

    utils::time_it([&](){circle_img(target_path, background_path, mask_path, 5);});

    cv::imwrite("output.jpg", img);

    return 0;
}