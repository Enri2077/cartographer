#include <chrono>
#include <iostream>
#include "cartographer/mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d.h"
#include <string>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>
#include <iomanip>

#include "cartographer/mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.pb.h"
#include "Eigen/Geometry"

//#include "absl/memory/memory.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/2d/tsdf_2d.h"
#include "cartographer/mapping/2d/tsdf_range_data_inserter_2d.h"
#include "cartographer/io/image.h"

using namespace cv;
using namespace std;
using namespace cartographer::mapping;

Vec3b colorMap(uint8_t in, string palette) {
    if (palette == "black,blue,green,red") {
        uint8_t intensity_min = 0;
        uint8_t intensity_1 = 85;
        uint8_t intensity_2 = 170;
        uint8_t intensity_max = 255;
        Vec3b color;
        if (in < intensity_1) {
            uint8_t start = intensity_min, end = intensity_1;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = x;  // B
            color[1] = 0;  // G
            color[2] = 0;  // R
            return color;
        } else if (in < intensity_2) {
            uint8_t start = intensity_1, end = intensity_2;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = 255 - x;  // B
            color[1] = x;  // G
            color[2] = 0;  // R
            return color;
        } else {
            uint8_t start = intensity_2, end = intensity_max;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = 0;  // B
            color[1] = 255 - x;  // G
            color[2] = x;  // R
            return color;
        }
    } else if (palette == "black,blue,green") {
        uint8_t intensity_min = 0;
        uint8_t intensity_1 = 128;
        uint8_t intensity_max = 255;
        Vec3b color;
        if (in < intensity_1) {
            uint8_t start = intensity_min, end = intensity_1;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = x;  // B
            color[1] = 0;  // G
            color[2] = 0;  // R
            return color;
        } else {
            uint8_t start = intensity_1, end = intensity_max;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = 255 - x;  // B
            color[1] = x;  // G
            color[2] = 0;  // R
            return color;
        }
    } else if (palette == "black,blue,yellow") {
        uint8_t intensity_min = 0;
        uint8_t intensity_1 = 128;
        uint8_t intensity_max = 255;
        Vec3b color;
        if (in < intensity_1) {
            uint8_t start = intensity_min, end = intensity_1;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = x;  // B
            color[1] = 0;  // G
            color[2] = 0;  // R
            return color;
        } else {
            uint8_t start = intensity_1, end = intensity_max;
            uint8_t x = intensity_max * (double) (in - start) / (end - start);
            color[0] = 255 - x;  // B
            color[1] = x;  // G
            color[2] = x;  // R
            return color;
        }
    } else {
        Vec3b color;
        color[0] = in;  // B
        color[1] = in;  // G
        color[2] = in;  // R
        return color;
    }
}


cv::RotatedRect getErrorEllipse(double chi_square_val, cv::Point2f mean, cv::Mat cov_mat) {
    // From https://gist.github.com/eroniki/2cff517bdd3f9d8051b5

    // Get the eigenvalues and eigenvectors
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(cov_mat, eigenvalues, eigenvectors);

    // Calculate the angle between the largest eigenvector and the x-axis
    double angle = atan2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(0, 0));

    // Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
    if (angle < 0)
        angle += 6.28318530718;

    // Convert to degrees instead of radians
    angle = 180 * angle / 3.14159265359;

    // Calculate the size of the minor and major axes
    double half_major_axis_size = chi_square_val * sqrt(eigenvalues.at<double>(0));
    double half_minor_axis_size = chi_square_val * sqrt(eigenvalues.at<double>(1));

    // Return the oriented ellipse
    // The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
    return cv::RotatedRect(mean, cv::Size2f(half_major_axis_size, half_minor_axis_size), -angle);

}

void insert_ranges_to_point_cloud(cartographer::sensor::PointCloud &cloud_out, vector<double> ranges_in, double angle_min, double angle_increment, double range_min, double range_max) {
    size_t n_pts = ranges_in.size();
    Eigen::ArrayXXf ranges(n_pts, 2);
    Eigen::ArrayXXf output(n_pts, 2);

    // Get the ranges into Eigen format
    for (size_t i = 0; i < n_pts; ++i) {
        ranges(i, 0) = (double) ranges_in[i];
        ranges(i, 1) = (double) ranges_in[i];
    }

    // Spherical->Cartesian projection
    Eigen::ArrayXXf co_sine_map_;
    co_sine_map_ = Eigen::ArrayXXf(n_pts, 2);
    for (size_t i = 0; i < n_pts; ++i) {
        co_sine_map_(i, 0) = cos(angle_min + (double) i * angle_increment);
        co_sine_map_(i, 1) = sin(angle_min + (double) i * angle_increment);
    }

    output = ranges * co_sine_map_;

    for (size_t i = 0; i < n_pts; ++i) {
        if (ranges_in[i] < range_max && ranges_in[i] >= range_min) {
            cloud_out.push_back({Eigen::Vector3f{output(i, 0), output(i, 1), 0.0}});
        }
    }
}


int main() {


//    //Covariance matrix of our data
//    cv::Mat cov_mat = (cv::Mat_<double>(2, 2) << 500, 0, 0, 100);
//
//    //The mean of our data
//    cv::Point2f mean(160, 120);
//
//    //Calculate the error ellipse for a 95% confidence interval
//    cv::RotatedRect ellipse = getErrorEllipse(2.4477, mean, cov_mat);
//
//    //Show the result
//    cv::Mat visualize_image(240, 320, CV_8UC1, cv::Scalar::all(0));
//    cv::ellipse(visualize_image, ellipse, cv::Scalar::all(255), 1);
//    cv::imshow("EllipseDemo", visualize_image);
//    cv::waitKey();
//    return 0;

    auto all_t_start = chrono::high_resolution_clock::now();
    string grid_source_ = "probability_grid_from_ranges"; // map, probability_grid_from_ranges, tsdf_from_ranges;
    bool apply_gaussian_blur = true;
    string palette_str = "black,blue,green,red"; // "black,blue,yellow";

    string point_clouds_source = "fake_corridor"; // olson, no_points, fake_corridor, run_scans
    double probability_grid_res_ = 0.05;


    // Scan
    auto make_point_clouds_t_start = chrono::high_resolution_clock::now();
    vector<cartographer::sensor::PointCloud> point_clouds;
    Mat grid_image_mat;
    int grid_image_w;
    int grid_image_h;
    double grid_image_res;
    double range_max_ = 0.0;

    if (point_clouds_source == "no_points") {
        cartographer::sensor::PointCloud point_cloud;
        point_clouds.push_back(point_cloud);
    }

    if (point_clouds_source == "fake_corridor") {
        cartographer::sensor::PointCloud point_cloud;
        auto left_wall = Eigen::Vector3f{-1.0, 0.0, 0.0};
        auto right_wall = Eigen::Vector3f{1.0, 0.0, 0.0};
        auto wall = Eigen::Vector3f{0.0, 30.0, 0.0};
        int num_wall_points = 30.0 / 0.05;
        for (int i = 0; i < num_wall_points; i++) {
            point_cloud.push_back({left_wall + wall / num_wall_points * i});
        }
        for (int i = 0; i < num_wall_points; i++) {
            if (i < num_wall_points / 2 - 15 || i > num_wall_points / 2 + 15) point_cloud.push_back({right_wall + wall / num_wall_points * i});
        }
        point_clouds.push_back(point_cloud);
    }

    if (point_clouds_source == "run_scans") {
//        std::ifstream infile("/home/enrico/ds/performance_modelling/output/test_slam/session_2020-10-20_13-21-55_595362_run_000000000/benchmark_data/scans.csv");  // airlab /scan
        std::ifstream infile("/home/enrico/ds/performance_modelling/output/test_slam/session_2020-10-27_21-56-48_330445_run_000000000/benchmark_data/scans_gt.csv");  // airlab /scan_gt
//        std::ifstream infile("/home/enrico/ds/performance_modelling/output/test_slam/session_2020-10-27_22-42-44_740120_run_000000000/benchmark_data/scans_gt.csv");  // 7A-B /scan_gt
//        std::ifstream infile("/home/enrico/ds/performance_modelling/output/test_slam/session_2020-10-28_09-24-03_450309_run_000000002/benchmark_data/scans_gt.csv");  // fr079 /scan_gt

        for (std::string line_str; getline(infile, line_str);) {
            std::stringstream ss(line_str);
            vector<string> values;
            for (string v; ss >> v;) {
                values.push_back(v);
                while (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
            }

            double /*t = stof(values[0]), */angle_min = stof(values[1]), /*angle_max = stof(values[2]), */angle_increment = stof(values[3]), range_min = stof(values[4]); //range_max = stof(values[5]);
            double range_max = 30.0;  // to ignore ranges higher than 30.0
            std::vector<double> ranges;
            for (size_t i = 6; i < values.size(); i++) ranges.push_back(stof(values[i]));

            cartographer::sensor::PointCloud point_cloud;
            insert_ranges_to_point_cloud(point_cloud, ranges, angle_min, angle_increment, range_min, range_max);
            point_clouds.push_back(point_cloud);
            if (range_max > range_max_) range_max_ = range_max;
        }
    }

    grid_image_mat = cv::imread("/home/enrico/Pictures/olson_scan.png", CV_8UC1);
    grid_image_w = grid_image_mat.cols;
    grid_image_h = grid_image_mat.rows;
    grid_image_res = 0.1; // meters per pixel
    if (point_clouds_source == "olson") {
        cartographer::sensor::PointCloud point_cloud;
        for (int i = 0; i < grid_image_w; i++) {
            float x = (i - grid_image_w / 2) * grid_image_res;
            for (int j = 0; j < grid_image_h; j++) {
                float y = (grid_image_h / 2 - j) * grid_image_res;
                if (grid_image_mat.at<uint8_t>(i, j) < 128) {
                    point_cloud.push_back({Eigen::Vector3f{x, y, 0.0}});
                }
            }
        }
        point_clouds.push_back(point_cloud);
    }

    auto make_point_clouds_t_end = chrono::high_resolution_clock::now();
    double make_point_clouds_elapsed_time_ms = chrono::duration<double, milli>(make_point_clouds_t_end - make_point_clouds_t_start).count();
    cout << "make_point_clouds_elapsed_time_ms " << make_point_clouds_elapsed_time_ms << endl;

    int point_cloud_count = 0;
    for (auto point_cloud : point_clouds) {
        auto make_grid_t_start = chrono::high_resolution_clock::now();

        // Probability grid (from scan itself)
        unique_ptr<Grid2D> grid_;
        ValueConversionTables conversion_tables_;
        unique_ptr<RangeDataInserterInterface> range_data_inserter_;
        if (grid_source_ == "tsdf_from_ranges") {
            grid_ = absl::make_unique<TSDF2D>(MapLimits(probability_grid_res_, Eigen::Vector2d(0.05, 0.25), CellLimits(6, 6)), 0.3, 1.0, &conversion_tables_);
            proto::TSDFRangeDataInserterOptions2D range_data_inserter_options;
            range_data_inserter_options.set_truncation_distance(0.15);
            range_data_inserter_options.set_maximum_weight(1.0);
            range_data_inserter_options.set_update_free_space(false);
            range_data_inserter_options.mutable_normal_estimation_options()->set_num_normal_samples(3);
            range_data_inserter_options.mutable_normal_estimation_options()->set_sample_radius(0.5);
            range_data_inserter_options.set_project_sdf_distance_to_scan_normal(true);
            range_data_inserter_options.set_update_weight_range_exponent(0);
            range_data_inserter_options.set_update_weight_angle_scan_normal_to_ray_kernel_bandwidth(0.5);
            range_data_inserter_options.set_update_weight_distance_cell_to_hit_kernel_bandwidth(0.5);
            range_data_inserter_ = absl::make_unique<TSDFRangeDataInserter2D>(range_data_inserter_options);
            range_data_inserter_->Insert(cartographer::sensor::RangeData{Eigen::Vector3f::Zero(), point_cloud, {}}, grid_.get());
            grid_->FinishUpdate();
        } else if (grid_source_ == "probability_grid_from_ranges") {
            if (point_clouds_source == "run_scans") {
                grid_ = absl::make_unique<ProbabilityGrid>(MapLimits(probability_grid_res_, Eigen::Vector2d(range_max_, range_max_), CellLimits(2 * range_max_ / probability_grid_res_, 2 * range_max_ / probability_grid_res_)), &conversion_tables_);
            } else {
                grid_ = absl::make_unique<ProbabilityGrid>(MapLimits(probability_grid_res_, Eigen::Vector2d(0.05, 0.25), CellLimits(6, 6)), &conversion_tables_);
            }
            proto::ProbabilityGridRangeDataInserterOptions2D range_data_inserter_options;
            range_data_inserter_options.set_hit_probability(1.0);
            range_data_inserter_options.set_miss_probability(0.0);
            range_data_inserter_options.set_insert_free_space(false);
            range_data_inserter_ = absl::make_unique<ProbabilityGridRangeDataInserter2D>(range_data_inserter_options);
            range_data_inserter_->Insert(cartographer::sensor::RangeData{Eigen::Vector3f::Zero(), point_cloud, {}}, grid_.get());
            grid_->FinishUpdate();
        } else if (grid_source_ == "map" and point_clouds_source == "olson") {
            grid_ = absl::make_unique<ProbabilityGrid>(MapLimits(grid_image_res, Eigen::Vector2d(grid_image_res * grid_image_w, grid_image_res * grid_image_h) * 0.5, CellLimits(grid_image_w, grid_image_h)), &conversion_tables_);
            ProbabilityGrid *const probability_grid = static_cast<ProbabilityGrid *>(grid_.get());

            if (apply_gaussian_blur) {

                Mat gaussian_blur_grid_image_mat;
                cv::GaussianBlur(grid_image_mat, gaussian_blur_grid_image_mat, cv::Size(), 4.0, 4.0);

                double max_p_gaussian = 0.0;
                for (int i = 0; i < grid_image_w; i++) {
                    for (int j = 0; j < grid_image_h; j++) {
                        double p_gaussian = 1.0 - (double) gaussian_blur_grid_image_mat.at<uint8_t>(grid_image_h - j - 1, i) / 255;
                        if (p_gaussian > max_p_gaussian) max_p_gaussian = p_gaussian;
                    }
                }

                for (int i = 0; i < grid_image_w; i++) {
                    for (int j = 0; j < grid_image_h; j++) {
                        double p = 1.0 - (double) grid_image_mat.at<uint8_t>(grid_image_h - j - 1, i) / 255;
                        double p_gaussian = 1.0 - (double) gaussian_blur_grid_image_mat.at<uint8_t>(grid_image_h - j - 1, i) / 255;
                        probability_grid->SetProbability(Eigen::Array2i(i, j), max(p, p_gaussian / max_p_gaussian));
                    }
                }

            } else {
                for (int i = 0; i < grid_image_w; i++) {
                    for (int j = 0; j < grid_image_h; j++) {
                        double p = 1.0 - (double) grid_image_mat.at<uint8_t>(grid_image_h - j - 1, i) / 255;
                        probability_grid->SetProbability(Eigen::Array2i(i, j), p);
                    }
                }
            }

            probability_grid->FinishUpdate();
        } else {
            cerr << "Unknown grid_source" << endl;
        }

        auto make_grid_t_end = chrono::high_resolution_clock::now();
        double make_grid_elapsed_time_ms = chrono::duration<double, milli>(make_grid_t_end - make_grid_t_start).count();
        cout << "make_grid_elapsed_time_ms " << make_grid_elapsed_time_ms << endl;

        // Compute the scores and covariance
        auto compute_covariance_t_start = chrono::high_resolution_clock::now();

        auto real_time_correlative_scan_matcher_options_ = scan_matching::proto::RealTimeCorrelativeScanMatcherOptions();
        real_time_correlative_scan_matcher_options_.set_angular_search_window(0.196);  // 0.392: 45° total, 0.785: 90° total,
        real_time_correlative_scan_matcher_options_.set_linear_search_window(1.0);
        real_time_correlative_scan_matcher_options_.set_rotation_delta_cost_weight(0.0);
        real_time_correlative_scan_matcher_options_.set_translation_delta_cost_weight(0.0);

        double score_resolution = 0.05;
        int num_angular_perturbations = 5;
        int num_linear_perturbations = std::ceil(real_time_correlative_scan_matcher_options_.linear_search_window() / score_resolution);
        double angular_perturbation_step_size = real_time_correlative_scan_matcher_options_.angular_search_window() / num_angular_perturbations;
        scan_matching::SearchParameters search_parameters(num_linear_perturbations, num_angular_perturbations, angular_perturbation_step_size, score_resolution);

        unique_ptr<scan_matching::RealTimeCorrelativeScanMatcher2D> real_time_correlative_scan_matcher_;
        real_time_correlative_scan_matcher_ = absl::make_unique<scan_matching::RealTimeCorrelativeScanMatcher2D>(real_time_correlative_scan_matcher_options_);
        Eigen::Matrix3d covariance;
        vector<scan_matching::Candidate2D> candidates = real_time_correlative_scan_matcher_->ComputeCovariance(point_cloud, *grid_, search_parameters, covariance);
        cout << "covariance: " << endl << covariance << endl;

        auto compute_covariance_t_end = chrono::high_resolution_clock::now();
        double compute_covariance_elapsed_time_ms = chrono::duration<double, milli>(compute_covariance_t_end - compute_covariance_t_start).count();
        cout << "compute_covariance_elapsed_time_ms " << compute_covariance_elapsed_time_ms << endl;

        auto make_images_t_start = chrono::high_resolution_clock::now();

        double max_x = 0.0, max_y = 0.0;
        for (auto candidate : candidates) {
            if (candidate.x > max_x) max_x = candidate.x;
            if (candidate.y > max_y) max_y = candidate.y;
        }

        // Make images of the score for each theta
        map<double, Mat> theta_to_scores_mat = map<double, Mat>();
        int image_size = 1 + 2 * num_linear_perturbations;
        for (auto candidate : candidates) {
            double theta = candidate.orientation;
            int i = image_size - num_linear_perturbations - candidate.y_index_offset - 1;
            int j = image_size - num_linear_perturbations - candidate.x_index_offset - 1;

            if (!theta_to_scores_mat.count(theta)) theta_to_scores_mat[theta] = Mat(image_size, image_size, CV_8UC3);

            uint8_t intensity = (uint8_t) (candidate.score * 255);
            theta_to_scores_mat[theta].at<Vec3b>(image_size - j - 1, i) = colorMap(intensity, palette_str);

            if (candidate.x == max_x && candidate.y == 0.0) {
                theta_to_scores_mat[theta].at<Vec3b>(image_size - j - 1, i) = Vec3b(0, 0, 255);  // red: x axis
            }
            if (candidate.y == max_y && candidate.x == 0.0) {
                theta_to_scores_mat[theta].at<Vec3b>(image_size - j - 1, i) = Vec3b(0, 255, 255);  // yellow: y axis
            }
        }

        // Make image of the probability grid
        int probability_grid_mat_w = grid_->limits().cell_limits().num_x_cells;
        int probability_grid_mat_h = grid_->limits().cell_limits().num_y_cells;
        Mat probability_grid_mat = Mat(probability_grid_mat_h, probability_grid_mat_w, CV_8UC3);

        auto probability_grid = static_cast<const ProbabilityGrid &>(*grid_);
        for (int i = 0; i < probability_grid_mat_w; i++) {
            for (int j = 0; j < probability_grid_mat_h; j++) {
                float probability = probability_grid.GetProbability(Eigen::Array2i(probability_grid_mat_h - j - 1, probability_grid_mat_w - i - 1));
                Vec3b color = Vec3b(255 * probability, 255 * probability, 255 * probability);
                probability_grid_mat.at<Vec3b>(probability_grid_mat_h - j - 1, i) = color;
            }
        }

        // Covariance matrix of our data
        Mat cov_mat = (Mat_<double>(2, 2) << covariance(0, 0)/grid_->limits().resolution(), covariance(0, 1)/grid_->limits().resolution(), covariance(1, 0)/grid_->limits().resolution(), covariance(1, 1)/grid_->limits().resolution());

        // The mean of our data
        Point2f mean(probability_grid_mat_w/2, probability_grid_mat_h/2);  // TODO: maybe switch x, y

        // Calculate the error ellipse for a 95% confidence interval
        RotatedRect confidence_interval_ellipse = getErrorEllipse(2.4477, mean, cov_mat);

        // Show the result
//        Mat visualize_image(240, 320, CV_8UC1, Scalar::all(0));
        ellipse(probability_grid_mat, confidence_interval_ellipse, Scalar::all(255), 1);
//        imshow("EllipseDemo", visualize_image);
//        waitKey();

        auto make_images_t_end = chrono::high_resolution_clock::now();
        double make_images_elapsed_time_ms = chrono::duration<double, milli>(make_images_t_end - make_images_t_start).count();
        cout << "make_images_elapsed_time_ms " << make_images_elapsed_time_ms << endl;

        // Save images of the score
        auto save_images_t_start = chrono::high_resolution_clock::now();
        vector<Mat> tiled_scores_mat_vec;
//        int image_theta_count = 0;
        for (auto const &theta_scores_mat : theta_to_scores_mat) {
            Mat scores_mat_with_border = Mat(theta_scores_mat.second.rows + 2, theta_scores_mat.second.cols + 2, CV_8UC3);
            scores_mat_with_border(cv::Rect_<int>(0, 0, scores_mat_with_border.cols, scores_mat_with_border.rows)) = Vec3b(20, 20, 20);
            theta_scores_mat.second.copyTo(scores_mat_with_border(cv::Rect_<int>(1, 1, theta_scores_mat.second.cols, theta_scores_mat.second.rows)));
            tiled_scores_mat_vec.push_back(scores_mat_with_border);

//            cv::resize(scores_mat_with_border, scores_mat_with_border, cv::Size(), 10.0, 10.0, cv::INTER_NEAREST);
//            double theta = theta_scores_mat.first;
//            ostringstream filename_ss;
//            filename_ss << "/home/enrico/tmp/gs/candidates_score_" << image_theta_count++ << "_theta_" << theta << ".png";
//            bool result = false;
//            try {
//                result = imwrite(filename_ss.str(), scores_mat_with_border, compression_params);
//            } catch (const cv::Exception &ex) {
//                fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//            }
//            if (!result) {
//                cerr << "Failed to write image" << endl;
//            }
        }

        Mat tiled_scores_mat;
        cv::hconcat(tiled_scores_mat_vec, tiled_scores_mat);
//        try {
//            ostringstream filename_tiled_scores_mat_ss;
//            filename_tiled_scores_mat_ss << "/home/enrico/tmp/gs/candidates_score_tiled_" << point_cloud_count << ".png";
//            imwrite(filename_tiled_scores_mat_ss.str(), tiled_scores_mat, compression_params);
//        } catch (const cv::Exception &ex) {
//            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        }
//
//        // Save image of the probability grid
//        ostringstream probability_grid_filename_ss;
//        probability_grid_filename_ss << "/home/enrico/tmp/gs/probability_grid_" << point_cloud_count << ".png";
//        bool probability_grid_result = false;
//        try {
//            probability_grid_result = imwrite(probability_grid_filename_ss.str(), probability_grid_mat, compression_params);
//        } catch (const cv::Exception &ex) {
//            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
//        }
//        if (!probability_grid_result) {
//            cerr << "Failed to write image" << endl;
//        }

        cv::resize(tiled_scores_mat, tiled_scores_mat, cv::Size(), 3, 3, cv::INTER_NEAREST);

        Mat tiled_all_mat = Mat(probability_grid_mat.rows + tiled_scores_mat.rows, max(probability_grid_mat.cols, tiled_scores_mat.cols), CV_8UC3);
        tiled_all_mat(cv::Rect_<int>(0, 0, tiled_all_mat.cols, tiled_all_mat.rows)) = Vec3b();
        probability_grid_mat.copyTo(tiled_all_mat(cv::Rect_<int>(tiled_all_mat.cols / 2 - probability_grid_mat.cols / 2, 0, probability_grid_mat.cols, probability_grid_mat.rows)));
        tiled_scores_mat.copyTo(tiled_all_mat(cv::Rect_<int>(tiled_all_mat.cols / 2 - tiled_scores_mat.cols / 2, probability_grid_mat.rows, tiled_scores_mat.cols, tiled_scores_mat.rows)));

        try {
            vector<int> compression_params;
            compression_params.push_back(IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            ostringstream filename_tiled_all_ss;
            filename_tiled_all_ss << "/home/enrico/tmp/gs/tiled_all_" << setfill('0') << setw(5) << point_cloud_count << ".png";
            imwrite(filename_tiled_all_ss.str(), tiled_all_mat, compression_params);
        } catch (const cv::Exception &ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        }

        auto save_images_t_end = chrono::high_resolution_clock::now();
        double save_images_elapsed_time_ms = chrono::duration<double, milli>(save_images_t_end - save_images_t_start).count();
        cout << "save_images_elapsed_time_ms " << save_images_elapsed_time_ms << endl;

        point_cloud_count++;
        cout << point_cloud_count << "/" << point_clouds.size() << endl << endl;
    }

    vector<int> palette_compression_params;
    palette_compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    palette_compression_params.push_back(9);
    int w = 256, h = 50;
    Mat palette_mat = Mat(h, w, CV_8UC3);
    for (int i = 0; i < w; i++) {
        uint8_t intensity = i;
        Vec3b color = colorMap(intensity, palette_str);
        for (int j = 0; j < h; j++) {
            palette_mat.at<Vec3b>(j, i) = color;
        }
    }
    try {
        imwrite("/home/enrico/tmp/gs/palette.png", palette_mat, palette_compression_params);
    } catch (const cv::Exception &ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }

    auto all_t_end = chrono::high_resolution_clock::now();
    double all_elapsed_time_ms = chrono::duration<double, milli>(all_t_end - all_t_start).count();
    cout << "all_elapsed_time_ms " << all_elapsed_time_ms << endl;

    // make video from tiled images with `ffmpeg -r 10 -i tiled_all_%05d.png -vcodec libx265 -crf 10 out.mp4`

    // TODO: compute
    //  - grid from scan_gt
    //  - keep translation/rotation_delta_cost_weight?
    //  - manually set linear and rotation step
    //  - covariance
    //  - covariance sum or something to get an overall measure of similarity (max axis for translation?)
    //  - eigen vectors
    //  - translation_covariance_axis·direction_of_translation
    //  - rotation_covariance·rotation
    return 0;
}
