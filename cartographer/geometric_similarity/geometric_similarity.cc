#include <chrono>
#include <iostream>
#include "cartographer/mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d.h"

#include "cartographer/mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.pb.h"
#include "Eigen/Geometry"

#include "absl/memory/memory.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/mapping/2d/tsdf_2d.h"
#include "cartographer/mapping/2d/tsdf_range_data_inserter_2d.h"
#include "cartographer/io/image.h"
#include <opencv2/imgcodecs.hpp>
#include <cv.hpp>

using namespace cv;
using namespace std;
using namespace cartographer::mapping;

int main() {

    bool use_tsdf = true;

    auto make_grid_t_start = chrono::high_resolution_clock::now();

    // Scan
    cartographer::sensor::PointCloud point_cloud_;
    auto left_wall = Eigen::Vector3f{-1.0, 0.0, 0.0};
    auto right_wall = Eigen::Vector3f{1.0, 0.0, 0.0};
    auto wall = Eigen::Vector3f{0.0, 1.0, 0.0};
    int num_wall_points = 100;
    for (int i = 0; i < num_wall_points; i++) {
        point_cloud_.push_back({left_wall + wall / num_wall_points * i});
        point_cloud_.push_back({right_wall + wall / num_wall_points * i});
    }

    // Probability grid (from scan itself)
    unique_ptr<Grid2D> grid_;
    ValueConversionTables conversion_tables_;
    unique_ptr<RangeDataInserterInterface> range_data_inserter_;
    if (use_tsdf) {
        grid_ = absl::make_unique<TSDF2D>(MapLimits(0.05, Eigen::Vector2d(0.3, 0.5), CellLimits(20, 20)), 0.3, 1.0, &conversion_tables_);
        proto::TSDFRangeDataInserterOptions2D range_data_inserter_options;
        range_data_inserter_options.set_truncation_distance(0.15);
        range_data_inserter_options.set_maximum_weight(1.0);
        range_data_inserter_options.set_update_free_space(false);
        range_data_inserter_options.mutable_normal_estimation_options()->set_num_normal_samples(4);
        range_data_inserter_options.mutable_normal_estimation_options()->set_sample_radius(0.5);
        range_data_inserter_options.set_project_sdf_distance_to_scan_normal(false);
        range_data_inserter_options.set_update_weight_range_exponent(2);
        range_data_inserter_options.set_update_weight_angle_scan_normal_to_ray_kernel_bandwidth(0.5);
        range_data_inserter_options.set_update_weight_distance_cell_to_hit_kernel_bandwidth(0.5);
        range_data_inserter_ = absl::make_unique<TSDFRangeDataInserter2D>(range_data_inserter_options);
        range_data_inserter_->Insert(cartographer::sensor::RangeData{Eigen::Vector3f::Zero(), point_cloud_, {}}, grid_.get());
        grid_->FinishUpdate();
    } else {
        grid_ = absl::make_unique<ProbabilityGrid>(MapLimits(0.05, Eigen::Vector2d(0.05, 0.25), CellLimits(6, 6)), &conversion_tables_);
        proto::ProbabilityGridRangeDataInserterOptions2D range_data_inserter_options;
        range_data_inserter_options.set_hit_probability(0.7);
        range_data_inserter_options.set_miss_probability(0.4);
        range_data_inserter_options.set_insert_free_space(true);
        range_data_inserter_ = absl::make_unique<ProbabilityGridRangeDataInserter2D>(range_data_inserter_options);
        range_data_inserter_->Insert(cartographer::sensor::RangeData{Eigen::Vector3f::Zero(), point_cloud_, {}}, grid_.get());
        grid_->FinishUpdate();
    }

    auto make_grid_t_end = chrono::high_resolution_clock::now();
    double make_grid_elapsed_time_ms = chrono::duration<double, milli>(make_grid_t_end - make_grid_t_start).count();
    cout << "make_grid_elapsed_time_ms " << make_grid_elapsed_time_ms << endl;

    // Compute the scores and covariance
    auto compute_covariance_t_start = chrono::high_resolution_clock::now();

    auto real_time_correlative_scan_matcher_options_ = scan_matching::proto::RealTimeCorrelativeScanMatcherOptions();
    real_time_correlative_scan_matcher_options_.set_angular_search_window(0.392);
    real_time_correlative_scan_matcher_options_.set_linear_search_window(2.0);
    real_time_correlative_scan_matcher_options_.set_rotation_delta_cost_weight(0.5);
    real_time_correlative_scan_matcher_options_.set_translation_delta_cost_weight(0.5);

    unique_ptr<scan_matching::RealTimeCorrelativeScanMatcher2D> real_time_correlative_scan_matcher_;
    real_time_correlative_scan_matcher_ = absl::make_unique<scan_matching::RealTimeCorrelativeScanMatcher2D>(real_time_correlative_scan_matcher_options_);
    vector<scan_matching::Candidate2D> candidates = real_time_correlative_scan_matcher_->ComputeCovariance(point_cloud_, *grid_);
    const scan_matching::Candidate2D &best_candidate = *std::max_element(candidates.begin(), candidates.end());
    double best_score = best_candidate.score;

    auto compute_covariance_t_end = chrono::high_resolution_clock::now();
    double compute_covariance_elapsed_time_ms = chrono::duration<double, milli>(compute_covariance_t_end - compute_covariance_t_start).count();
    cout << "compute_covariance_elapsed_time_ms " << compute_covariance_elapsed_time_ms << endl;

    // Make images of the score for each theta
    auto make_images_t_start = chrono::high_resolution_clock::now();

    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    map<double, Mat> theta_to_mat;
    const int num_linear_perturbations = ceil(real_time_correlative_scan_matcher_options_.linear_search_window() / grid_->limits().resolution());
    int image_size = 1 + 2 * num_linear_perturbations;
    for (auto candidate : candidates) {
        double theta = candidate.orientation;
        int x_image_index = num_linear_perturbations + candidate.x_index_offset;
        int y_image_index = num_linear_perturbations + candidate.y_index_offset;

        if (!theta_to_mat.count(theta)) theta_to_mat[theta] = Mat(image_size, image_size, CV_8UC3);

        Vec3b color;
        if (candidate.score == best_score) {
            color[0] = 0;
            color[1] = 0;
            color[2] = 255;
            cout << "best_score: x: " << x_image_index << " y: " << y_image_index << " theta: " << theta << endl;
        } else {
            uint8_t intensity = (uint8_t) (candidate.score / best_score * 255);  // TODD Should be divided by best_score in covariance too?
            color[0] = intensity;
            color[1] = intensity;
            color[2] = intensity;
        }
        theta_to_mat[theta].at<Vec3b>(x_image_index, y_image_index) = color;
    }

    // Make image of the probability grid
    Mat probability_grid_mat = Mat(grid_->limits().cell_limits().num_x_cells, grid_->limits().cell_limits().num_y_cells, CV_8UC1);

    auto probability_grid = static_cast<const ProbabilityGrid &>(*grid_);
    for (int i = 0; i < grid_->limits().cell_limits().num_x_cells; i++) {
        for (int j = 0; j < grid_->limits().cell_limits().num_y_cells; j++) {
            float probability = probability_grid.GetProbability(Eigen::Array2i(i, j));
            char intensity = (char) (probability * 255);
            probability_grid_mat.at<char>(i, j) = intensity;
        }
    }
    auto make_images_t_end = chrono::high_resolution_clock::now();
    double make_images_elapsed_time_ms = chrono::duration<double, milli>(make_images_t_end - make_images_t_start).count();
    cout << "make_images_elapsed_time_ms " << make_images_elapsed_time_ms << endl;

    // Save images of the score
    auto save_images_t_start = chrono::high_resolution_clock::now();
    for (auto const &theta_mat : theta_to_mat) {
        double theta = theta_mat.first;
        Mat mat = theta_mat.second;
        Mat upscaled_mat;
        cv::resize(mat, upscaled_mat, cv::Size(), 10.0, 10.0, cv::INTER_NEAREST);
        ostringstream filename_ss;
        filename_ss << "/home/enrico/tmp/gs/candidates_score_" << (int) (1000 * (theta + real_time_correlative_scan_matcher_options_.angular_search_window())) << "theta_" << theta << ".png";
        bool result = false;
        try {
            result = imwrite(filename_ss.str(), upscaled_mat, compression_params);
        } catch (const cv::Exception &ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        }
        if (!result) {
            cerr << "Failed to write image" << endl;
        }
    }

    // Save image of the probability grid
    ostringstream probability_grid_filename_ss;
    probability_grid_filename_ss << "/home/enrico/tmp/gs/probability_grid_.png";
    bool probability_grid_result = false;
    try {
        Mat upscaled_probability_grid_mat;
        cv::resize(probability_grid_mat, upscaled_probability_grid_mat, cv::Size(), 10.0, 10.0, cv::INTER_NEAREST);
        probability_grid_result = imwrite(probability_grid_filename_ss.str(), upscaled_probability_grid_mat, compression_params);
    } catch (const cv::Exception &ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    if (!probability_grid_result) {
        cerr << "Failed to write image" << endl;
    }

    auto save_images_t_end = chrono::high_resolution_clock::now();
    double save_images_elapsed_time_ms = chrono::duration<double, milli>(save_images_t_end - save_images_t_start).count();
    cout << "save_images_elapsed_time_ms " << save_images_elapsed_time_ms << endl;

    // TODO: compute
    //  - grid from scan_gt
    //  - covariance
    //  - covariance sum or something to get an overall measure of similarity (max axis for translation?)
    //  - translation_covariance_axis·direction_of_translation
    //  - rotation_covariance·rotation
    return 0;
}
