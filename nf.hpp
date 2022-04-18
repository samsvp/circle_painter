#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <execution>

#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"


namespace nf
{

/*
 * Returns the given masked region
 */
cv::Mat circle_mask(const cv::Mat&  img, int c_x, int c_y, int r) 
{
    cv::Mat masked = img.clone();
    cv::Mat1b mask = cv::Mat::ones(img.size(), CV_8UC1);
    cv::circle(mask, cv::Point(c_x, c_y), r, cv::Scalar(0, 0, 0, 0), -1, 8, 0);
    masked.setTo(0, mask);
    return masked;
}


/*
 * Translates the image by the given offsets
 */
cv::Mat translate(const cv::Mat&  img, int offset_x, int offset_y)
{
    cv::Mat out = cv::Mat::zeros(img.size(), img.type());

    cv::Rect source = cv::Rect(cv::max(0,-offset_x), cv::max(0,-offset_y), 
        img.cols-abs(offset_x), img.rows-cv::abs(offset_y)
    );
    cv::Rect target = cv::Rect(cv::max(0,offset_x), cv::max(0,offset_y),
        img.cols-abs(offset_x),img.rows-cv::abs(offset_y)
    );

    img(source).copyTo(out(target));
    
    return out;
}


/*
 * Overlays the non black part of the foreground into the background
 */
cv::Mat overlay(const cv::Mat& foreground, const cv::Mat& background)
{
    cv::Mat out = cv::Mat::zeros(foreground.size(), foreground.type());
    
    for (size_t y = 0; y < foreground.cols; y++)
    {
        for (size_t x = 0; x < foreground.rows; x++)
        {
            out.at<cv::Vec3b>(x, y) = foreground.at<cv::Vec3b>(x, y) == cv::Vec3b(0,0,0) ?
                background.at<cv::Vec3b>(x, y) : foreground.at<cv::Vec3b>(x, y);
        }
    }
    
    return out;
}


/*
 * Overlays the foreground into the background using the given mask
 */
cv::Mat overlay(const cv::Mat& foreground, const cv::Mat& background, const cv::Mat& mask)
{
    cv::Mat out = cv::Mat::zeros(foreground.size(), foreground.type());

    for (size_t y = 0; y < foreground.cols * foreground.rows; y++)
    {
        out.at<cv::Vec3b>(y) = mask.at<cv::Vec3b>(y) == cv::Vec3b(0,0,0) ?
            background.at<cv::Vec3b>(y) : foreground.at<cv::Vec3b>(y);
    }
    
    return out;
}


std::vector<int> img_mean(const cv::Mat& img, const cv::Mat& mask)
{
    // extract each channel
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);

    // calc channel mean
    auto calc_mean = [&mask](cv::Mat img) {
        int s = 0;
        int n = 0;
        
        for (size_t i = 0; i < mask.cols * mask.rows; i++)
        {
            // apply masking
            if (mask.at<cv::Vec3b>(i) != cv::Vec3b(0,0,0))
            {
                s += img.at<uchar>(i);
                n++;
            }
                
        }
        return s / n;
    };

    std::vector<int> means(3);
    std::transform(std::execution::par_unseq, channels.begin(), 
        channels.end(), means.begin(), calc_mean);
    
    return means;
}


cv::Mat resize(const cv::Mat& img, float sx, float sy)
{
    cv::Mat out;
    cv::resize(img, out, cv::Size(), sx, sy);
    return out;
}


cv::Mat get_edges(const cv::Mat& gray, float thresh, int ksize=5)
{
    cv::Mat blur;
    cv::blur(gray, blur, cv::Size(15, 15));
    cv::Mat edges;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    cv::Sobel(blur, grad_x, CV_16S, 1, 0, ksize);
    cv::Sobel(blur, grad_y, CV_16S, 0, 1, ksize);
    // converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edges);

    edges.setTo(0, edges < thresh);
    edges.setTo(255, edges > thresh);
    return edges;
}   


cv::Mat get_edges(const cv::Mat& gray, float thresh, const cv::Mat& mask, int ksize=5)
{
    cv::Mat edges = nf::get_edges(gray, thresh, ksize);
    cv::Mat edges_mask;
    cv::bitwise_and(edges, mask, edges_mask);
    return edges_mask;
}   


cv::Mat get_edges_color(const cv::Mat& img, float thresh, int ksize=5)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    return nf::get_edges(gray, thresh, ksize);
}


cv::Mat get_edges_color(const cv::Mat& img, float thresh, const cv::Mat& mask, int ksize=5)
{
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gray_mask;
    cv::cvtColor(mask, gray_mask, cv::COLOR_BGR2GRAY);
    return nf::get_edges(gray, thresh, gray_mask, ksize);
}


namespace utils
{
const char* parse_option(const char* option, const char* deflt, 
    int argc, char**argv)
{
    for (int i=1; i<argc; i++)
    {
        if (std::strcmp(argv[i], option) == 0)
            return argv[i+1];
    }
    return deflt;
}


template<typename T>
T parse_option(const char* option, T deflt, 
    int argc, char**argv)
{
    for (int i=1; i<argc; i++)
    {
        if (std::strcmp(argv[i], option) == 0)
        {
            std::stringstream ss(argv[i+1]);
            T t;
            ss >> t;
            return t;
        }
    }
    return deflt;
}


template <typename F>
auto time_it(F f)
{
    using clock = std::chrono::system_clock;
    using sec = std::chrono::duration<double>;
    // for milliseconds, use using ms = std::chrono::duration<double, std::milli>;

    const auto before = clock::now();

    f();

    const sec duration = clock::now() - before;
    std::cout << "It took " << duration.count() << "s" << std::endl;
}

} // namespace utils


} // namespace nf