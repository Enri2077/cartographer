#include <chrono>
#include <iostream>
#include "cartographer/mapping/internal/2d/scan_matching/real_time_correlative_scan_matcher_2d.h"

#include "cartographer/mapping/proto/scan_matching/real_time_correlative_scan_matcher_options.pb.h"
#include "Eigen/Geometry"

#include "absl/memory/memory.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/probability_grid_range_data_inserter_2d.h"
#include "cartographer/sensor/point_cloud.h"
#include "cartographer/mapping/2d/grid_2d.h"
#include "cartographer/io/image.h"
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

int main() {

    cartographer::mapping::ValueConversionTables conversion_tables_;
    std::unique_ptr<cartographer::mapping::RangeDataInserterInterface> range_data_inserter_;
    std::unique_ptr<cartographer::mapping::scan_matching::RealTimeCorrelativeScanMatcher2D> real_time_correlative_scan_matcher_;

    // Scan
    cartographer::sensor::PointCloud point_cloud_;
//    point_cloud_.push_back({Eigen::Vector3f{0.025f, 0.175f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.025f, 0.175f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.075f, 0.175f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.125f, 0.175f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.125f, 0.125f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.125f, 0.075f, 0.f}});
//    point_cloud_.push_back({Eigen::Vector3f{-0.125f, 0.025f, 0.f}});
    auto left_wall = Eigen::Vector3f{-1.0, 0.0, 0.0};
    auto right_wall = Eigen::Vector3f{1.0, 0.0, 0.0};
    auto wall = Eigen::Vector3f{0.0, 1.0, 0.0};

    for(int i = 0; i < 10; i++){
        point_cloud_.push_back({left_wall + wall*i});
        point_cloud_.push_back({right_wall + wall*i});
        cout << "left: " << left_wall + wall*i << endl;
        cout << "right: " << right_wall + wall*i << endl;
    }

    // Probability grid (from scan TODO get it raytraced scan)
    std::unique_ptr<cartographer::mapping::Grid2D> grid_;
    grid_ = absl::make_unique<cartographer::mapping::ProbabilityGrid>(cartographer::mapping::MapLimits(0.05, Eigen::Vector2d(0.05, 0.25), cartographer::mapping::CellLimits(6, 6)), &conversion_tables_);

    cartographer::mapping::proto::ProbabilityGridRangeDataInserterOptions2D range_data_inserter_options;
    range_data_inserter_options.set_hit_probability(0.7);
    range_data_inserter_options.set_miss_probability(0.4);
    range_data_inserter_options.set_insert_free_space(true);
    range_data_inserter_ = absl::make_unique<cartographer::mapping::ProbabilityGridRangeDataInserter2D>(range_data_inserter_options);
    range_data_inserter_->Insert(cartographer::sensor::RangeData{Eigen::Vector3f::Zero(), point_cloud_, {}}, grid_.get());
    grid_->FinishUpdate();

    auto compute_covariance_t_start = std::chrono::high_resolution_clock::now();

    auto real_time_correlative_scan_matcher_options_ = cartographer::mapping::scan_matching::proto::RealTimeCorrelativeScanMatcherOptions();
    real_time_correlative_scan_matcher_options_.set_angular_search_window(0.5);
    real_time_correlative_scan_matcher_options_.set_linear_search_window(1.0);
    real_time_correlative_scan_matcher_options_.set_rotation_delta_cost_weight(0.0);
    real_time_correlative_scan_matcher_options_.set_translation_delta_cost_weight(0.0);
    real_time_correlative_scan_matcher_ = absl::make_unique<cartographer::mapping::scan_matching::RealTimeCorrelativeScanMatcher2D>(real_time_correlative_scan_matcher_options_);
    std::vector<cartographer::mapping::scan_matching::Candidate2D> candidates = real_time_correlative_scan_matcher_->ComputeCovariance(point_cloud_, *grid_);

    // Make images of the score for each theta
    map<double, Mat> theta_to_mat;

    const int num_linear_perturbations = std::ceil(real_time_correlative_scan_matcher_options_.linear_search_window() / grid_->limits().resolution());
    int image_size = 1 + 2 * num_linear_perturbations;
    std::vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    for (auto candidate : candidates) {
        double theta = candidate.orientation;
        int x_image_index = num_linear_perturbations + candidate.x_index_offset;
        int y_image_index = num_linear_perturbations + candidate.y_index_offset;

        if (!theta_to_mat.count(theta)) theta_to_mat[theta] = Mat(image_size, image_size, CV_8UC1);

        char intensity = (char) (candidate.score * 255);
        theta_to_mat[theta].at<char>(x_image_index, y_image_index) = intensity;
    }

    auto compute_covariance_t_end = std::chrono::high_resolution_clock::now();
    double compute_covariance_elapsed_time_ms = std::chrono::duration<double, std::milli>(compute_covariance_t_end - compute_covariance_t_start).count();
    cout << "compute_covariance_elapsed_time_ms " << compute_covariance_elapsed_time_ms << endl;

    auto save_images_t_start = std::chrono::high_resolution_clock::now();

    for (auto const &theta_mat : theta_to_mat) {
        std::ostringstream filename_ss;
        filename_ss << "/home/enrico/tmp/gs/candidates_score_" << (int)(1000*(theta_mat.first + real_time_correlative_scan_matcher_options_.angular_search_window())) << "theta_" << theta_mat.first << ".png";
        bool result = false;
        try {
            result = imwrite(filename_ss.str(), theta_mat.second, compression_params);
        } catch (const cv::Exception &ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        }
        if (!result) {
            cerr << "Failed to write image" << endl;
        }
    }

    auto save_images_t_end = std::chrono::high_resolution_clock::now();
    double save_images_elapsed_time_ms = std::chrono::duration<double, std::milli>(save_images_t_end - save_images_t_start).count();
    cout << "save_images_elapsed_time_ms " << save_images_elapsed_time_ms << endl;

    return 0;
}
