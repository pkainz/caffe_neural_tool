/*
 * image_preprocessor.hpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Fabian Tschopp
 */

#ifndef IMAGE_PROCESSOR_HPP_
#define IMAGE_PROCESSOR_HPP_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <functional>
#include "caffe_neural_tool.hpp"

namespace caffe_neural {

class ImageProcessor {
 public:
  ImageProcessor(int patch_size, int nr_labels);
  void SubmitRawImage(cv::Mat input, int img_id);
  void ClearImages();
  void SubmitImage(cv::Mat raw, int img_id, std::vector<cv::Mat> labels);
  int Init();
  void SetBorderParams(bool apply, int border_size);
  void SetClaheParams(bool apply, float clip_limit);
  void SetBlurParams(bool apply, float mu, float std, int blur_size);
  void SetCropParams(int image_crop, int label_crop);
  void rotate(cv::Mat& src, double angle, cv::Mat& dst, cv::InterpolationFlags interpolation);
  void SetNormalizationParams(bool apply);

  void SetRotationParams(bool apply);
  void SetPatchMirrorParams(bool apply);
  void SetIntShiftParams(bool apply, bool use_hsv, int range);
  void SetScaleParams(bool apply, float range, int instances);
  void scale_keep_size(cv::Mat& src, double scale, cv::Mat& dst, cv::InterpolationFlags interpolation);

  void SetLabelHistEqParams(bool apply, bool patch_prior, bool mask_prob,
                            std::vector<float> label_boost);

  long BinarySearchPatch(double offset);

  void SetLabelConsolidateParams(bool apply, std::vector<int> labels);

  std::vector<cv::Mat>& raw_images();
  std::vector<cv::Mat>& label_images();
  std::vector<int>& image_number();

 protected:

  std::vector<cv::Mat> raw_images_;   // single raw images
  std::vector<cv::Mat> label_images_; // single 8-bit label images
  std::vector<std::vector<cv::Mat>> label_stack_; //
  std::vector<int> image_number_;
  std::vector<unsigned long> cum_sum_;

  // General parameters
  std::vector<int> image_size_x_;
  std::vector<int> image_size_y_;
  int patch_size_;
  int nr_labels_;
  std::function<double()> offset_selector_;
  std::vector<int> off_size_x_;
  std::vector<int> off_size_y_;

  // Normalization parameters
  bool apply_normalization_ = false;

  // Final crop subtraction parameters
  int image_crop_ = 0;
  int label_crop_ = 0;

  // Border parameters
  bool apply_border_reflect_ = false;
  int border_size_;  // = 0.5 * padding in prototxt

  // CLAHE parameters
  bool apply_clahe_ = false;
  cv::Ptr<cv::CLAHE> clahe_;

  // Blur parameters
  bool apply_blur_ = false;
  float blur_mean_;
  float blur_std_;
  int blur_size_;
  std::function<float()> blur_random_selector_;

  // Simple rotation parameters
  bool apply_rotation_ = false;
  bool random_rotation_ = false;
  std::function<unsigned int()> rotation_rand_;
  std::function<unsigned int()> rotation_rand_angle_;

  // Patch mirroring
  bool apply_patch_mirroring_ = false;
  std::function<unsigned int()> patch_mirror_rand_;

  // intensity shifting
  bool apply_intensity_shift_ = false;
  bool use_hsv_ = false;
  int intensity_shift_range_;
  std::function<int()> random_intrange_selector_;

  // scale parameters (default is no scaling)
  bool apply_scaling_ = false;
  float scale_range_;
  int scaled_instances_;
  std::function<double()> random_upscale_selector_;
  std::function<double()> random_downscale_selector_;

  // Label histrogram equalization
  bool apply_label_hist_eq_ = false;
  bool apply_label_patch_prior_ = false;
  bool apply_label_pixel_mask_ = false;
  std::vector<double> label_running_probability_;
  std::vector<float> label_mask_probability_;
  std::vector<std::function<float()>> label_mask_prob_rand_;
  std::function<double()> label_patch_prior_rand_;
  std::vector<float> label_boost_;

  // Label consolidation
  bool label_consolidate_ = false;
  std::vector<int> label_consolidate_labels_;

  // Patch sequence index
  int sequence_index_;
};

class ProcessImageProcessor : public ImageProcessor {
 public:
  ProcessImageProcessor(int patch_size, int nr_labels);
 protected:
};

class TrainImageProcessor : public ImageProcessor {
 public:
  TrainImageProcessor(int patch_size, int nr_labels);
  std::vector<cv::Mat> DrawPatchRandom();
 protected:
};

}

#endif /* IMAGE_PROCESSOR_HPP_ */
