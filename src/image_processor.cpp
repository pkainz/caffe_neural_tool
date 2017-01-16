/*
 * image_preprocessor.cpp
 *
 *  Created on: Apr 3, 2015
 *      Author: Fabian Tschopp
 */

#include "image_processor.hpp"
#include <glog/logging.h>

#include <omp.h>
#include <iostream>
#include <set>
#include "utils.hpp"

namespace caffe_neural {

ImageProcessor::ImageProcessor(int patch_size, int nr_labels)
    : patch_size_(patch_size),
      nr_labels_(nr_labels) {

}

std::vector<cv::Mat>& ImageProcessor::raw_images() {
  return raw_images_;
}

std::vector<cv::Mat>& ImageProcessor::label_images() {
  return label_images_;
}

std::vector<int>& ImageProcessor::image_number() {
  return image_number_;
}

void ImageProcessor::SetCropParams(int image_crop, int label_crop) {
  image_crop_ = image_crop;
  label_crop_ = label_crop;
}

void ImageProcessor::SetNormalizationParams(bool apply) {
  apply_normalization_ = apply;
}

void ImageProcessor::SetIntShiftParams(bool apply, bool use_hsv, int range){
  apply_intensity_shift_ = apply;
  use_hsv_ = use_hsv;
  intensity_shift_range_ = range;
  random_intrange_selector_ = GetRandomUniform(-range, range);
}

void ImageProcessor::SetScaleParams(bool apply, float range, int instances){
    apply_scaling_ = apply;
    scale_range_ = range;
    scaled_instances_ = instances;
    random_upscale_selector_ = GetRandomUniform<double>(1.f, 1.f + range);
    random_downscale_selector_ = GetRandomUniform<double>(1.f - range, 1.f);
}

void ImageProcessor::ClearImages() {
  raw_images_.clear();
  label_images_.clear();
  image_number_.clear();

  // overall statistics
  image_size_x_.clear();
  image_size_y_.clear();
  off_size_x_.clear();
  off_size_y_.clear();
}

void ImageProcessor::SetLabelConsolidateParams(bool apply, std::vector<int> labels) {
  label_consolidate_ = apply;
  label_consolidate_labels_ = labels;
}

void ImageProcessor::SubmitImage(cv::Mat raw, int img_id,
                                 std::vector<cv::Mat> labels) {

  // create additional 'scaled_instances_' versions of the original image
  for (unsigned int n=0; n <= scaled_instances_; ++n){
//      std::cout << n << std::endl;
//      std::cout << scaled_instances_ << std::endl;
//      std::cout << apply_scaling_ << std::endl;

      std::vector<cv::Mat> rawsplit;
      cv::split(raw, rawsplit);

      if (apply_clahe_) {
        for (unsigned int i = 0; i < rawsplit.size(); ++i) {
          cv::Mat dst;
          clahe_->apply(rawsplit[i], dst);
          rawsplit[i] = dst;
        }
      }

      cv::Mat src;
      cv::merge(rawsplit, src);
      src.convertTo(src, CV_32FC(3), 1.0 / 255.0);

      if (apply_normalization_) {
        cv::Mat dst;
        cv::normalize(src, dst, -1.0, 1.0, cv::NORM_MINMAX);
        src = dst;
      }

      // scale DOWN the input/label image (once) and crop or extend the border
      if (n > 0 && apply_scaling_) {
          cv::Mat dst;
          std::vector<cv::Mat> dst_labels;

          // draw random downscale factor
          double scale_ = random_downscale_selector_();
          //std::cout << scale_ << std::endl;

          // ensure that the size of dst is still large enough to fit the network input layer
          int orig_width = src.cols;
          int orig_height = src.rows;

          // compute lower boundaries for dst images
          int dst_width = std::max((int) (scale_ * orig_width), patch_size_ + 2 * border_size_);
          int dst_height = std::max((int) (scale_ * orig_height), patch_size_ + 2 * border_size_);

          cv::resize(src, dst, cv::Size(dst_width, dst_height), cv::INTER_LINEAR);
          src = dst;

          for (unsigned int i = 0; i < labels.size(); ++i) {
              cv::Mat dst_label;
              cv::resize(labels[i], dst_label, cv::Size(dst_width, dst_height), cv::INTER_NEAREST);
              dst_labels.push_back(dst_label);
          }
          labels = dst_labels;
      }

      if (apply_border_reflect_) {
        cv::Mat dst;
        cv::copyMakeBorder(src, dst, border_size_, border_size_, border_size_,
                           border_size_, IPL_BORDER_REFLECT, cv::Scalar::all(0.0));
        src = dst;
      }

      raw_images_.push_back(src);
      image_number_.push_back(img_id);
      label_stack_.push_back(labels);
  }

  //std::cout << "Number of available images: " << raw_images_.size() << std::endl;

}

int ImageProcessor::Init() {

  if (label_stack_[0].size() > 1) {
    for (unsigned int j = 0; j < label_stack_.size(); ++j) {

      cv::Mat dst_label(label_stack_[j][0].rows, label_stack_[j][0].cols,
      CV_32FC1);
      dst_label.setTo(cv::Scalar(0.0));

      for (unsigned int i = 0; i < label_stack_[j].size(); ++i) {
#pragma omp parallel for
        for (int y = 0; y < label_stack_[j][i].rows; ++y) {
          for (int x = 0; x < label_stack_[j][i].cols; ++x) {
            // Multiple images with 1 label defined per image
            int ks = label_stack_[j][i].at<unsigned char>(y, x);
            if (ks > 0) {
              (dst_label.at<float>(y, x)) = i;
            }
          }
        }
      }
      label_images_.push_back(dst_label);
    }
  } else {

    std::set<int> labelset;

    for (unsigned int j = 0; j < label_stack_.size(); ++j) {// for all images
      for (int y = 0; y < label_stack_[j][0].rows; ++y) {   // for all rows
        for (int x = 0; x < label_stack_[j][0].cols; ++x) { // for all cols
          int ks = label_stack_[j][0].at<unsigned char>(y, x);
          // Label not yet registered
          if (labelset.find(ks) == labelset.end()) {
            labelset.insert(ks);
          }
        }
      }
    }
    // convert labels to integers starting at 0
    for (unsigned int j = 0; j < label_stack_.size(); ++j) { // for each image
      cv::Mat dst_label(label_stack_[j][0].rows, label_stack_[j][0].cols, CV_32FC1);
      dst_label.setTo(cv::Scalar(0.0));
#pragma omp parallel for
      for (int y = 0; y < label_stack_[j][0].rows; ++y) {    // for each row
        for (int x = 0; x < label_stack_[j][0].cols; ++x) {  // for each col
          // Single image with many labels defined per image
          int ks = label_stack_[j][0].at<unsigned char>(y, x);
          (dst_label.at<float>(y, x)) = (float) std::distance(
              labelset.begin(), labelset.find(ks));
        }
      }
      label_images_.push_back(dst_label);
    }
  }

  label_stack_.clear();

  if (raw_images_.size() == 0 || label_images_.size() == 0
      || raw_images_.size() != label_images_.size()) {
    return -1;
  }


  // compute the cumulative sum of pixels after each image
  unsigned long cumsum = 0;
  cum_sum_.clear();
  cum_sum_.push_back(0);
  for (unsigned int k = 0; k < label_images_.size(); ++k) {

      // record rows/cols for label images
      image_size_x_.push_back(label_images_[k].cols - 2 * border_size_);
      image_size_y_.push_back(label_images_[k].rows - 2 * border_size_);

      // record off_sizes for each image
      off_size_x_.push_back((image_size_x_[k] - patch_size_) + 1);
      off_size_y_.push_back((image_size_y_[k] - patch_size_) + 1);

      // check that x and y are larger than 0
      assert(off_size_x_[k] > 0);
      assert(off_size_y_[k] > 0);

      // compute the cumulative sum of pixels within the off_size_
      cumsum += (off_size_x_[k] * off_size_y_[k]);
      cum_sum_.push_back(cumsum);
  }

  // precomputes the image size from the first image in the list
//  image_size_x_ = raw_images_[0].cols - 2 * border_size_;
//  image_size_y_ = raw_images_[0].rows - 2 * border_size_;

//  int off_size_x = (image_size_x_ - patch_size_) + 1;
//  int off_size_y = (image_size_y_ - patch_size_) + 1;

  // create a function that selects an offset location in the image
//  offset_selector_ = GetRandomUniform<double>(
//      0, label_images_.size() * off_size_x * off_size_y);

  offset_selector_ = GetRandomUniform<double>(0, cumsum);

  if (apply_label_hist_eq_) {
    // compute label statistics
    std::vector<long> label_count(nr_labels_);
    std::vector<double> label_freq(nr_labels_);

    long total_count = 0;
    for (unsigned int k = 0; k < label_images_.size(); ++k) {
      cv::Mat label_image = label_images_[k];
      for (int y = 0; y < image_size_y_[k]; ++y) {
        for (int x = 0; x < image_size_x_[k]; ++x) {
          // Label counting should be biased towards the borders, as less batches cover those parts
          long mult = std::min(std::min(x, image_size_x_[k] - x), patch_size_)
              * std::min(std::min(y, image_size_y_[k] - y), patch_size_);
          label_count[label_image.at<float>(y, x)] += mult;
          total_count += mult;
        }
      }
    }

    for (int l = 0; l < nr_labels_; ++l) {
      label_freq[l] = (double) (label_count[l]) / (double) (total_count);
      LOG(INFO) << "Label " << l << ": " << label_freq[l];
    }

    if (apply_label_patch_prior_) {

      std::vector<double> weighted_label_count(nr_labels_);

//      label_running_probability_.resize(
//          label_images_.size() * off_size_x * off_size_y);

      label_running_probability_.resize(cumsum);


      // Loop over all images
//#pragma omp parallel for
      for (unsigned int k = 0; k < label_images_.size(); ++k) {
        cv::Mat label_image = label_images_[k];
        std::vector<long> patch_label_count(nr_labels_);
        // Loop over all possible patches
        for (int y = 0; y < off_size_y_[k]; ++y) {
          for (int x = 0; x < off_size_x_[k]; ++x) {

            if(x == 0) {
              // Fully compute the patches at the beginning of the row
              for(int l = 0; l < nr_labels_; ++l) {
                patch_label_count[l] = 0;
              }
              for (int py = y; py < y + patch_size_; ++py) {
                for (int px = x; px < x + patch_size_; ++px) {
                  patch_label_count[label_image.at<float>(py, px)]++;
                }
              }
            } else {
              // Only compute difference for further patches in the row (more efficient)
              for (int py = y; py < y + patch_size_; ++py) {
                patch_label_count[label_image.at<float>(py, x - 1)]--;
                patch_label_count[label_image.at<float>(py, x + patch_size_ - 1)]++;
              }
            }

            // Compute the weight of the patch
            double patch_weight = 0;
            for (int l = 0; l < nr_labels_; ++l) {
              patch_weight += (((double) (patch_label_count[l]))
                  / ((double) (patch_size_ * patch_size_))) / (label_freq[l]);
            }
            for (int l = 0; l < nr_labels_; ++l) {
              weighted_label_count[l] += patch_weight * patch_label_count[l];
            }
//            label_running_probability_[k * off_size_x * off_size_y
//                + y * off_size_x + x] = patch_weight;
            unsigned int idx__ = cum_sum_[k] + y * off_size_x_[k] + x;
            //std::cout << idx__ << std::endl;
            label_running_probability_[idx__] = patch_weight;
          }
        }
      }

      //std::cout << "check" << std::endl;

      for (unsigned int k = 1; k < label_running_probability_.size(); ++k) {
        label_running_probability_[k] += label_running_probability_[k - 1];
      }

      double freq_divisor = 0;
      for (int l = 0; l < nr_labels_; ++l) {
        freq_divisor += weighted_label_count[l];
      }
      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] = weighted_label_count[l] / freq_divisor;
        LOG(INFO) << "Label " << l << ": " << label_freq[l];
      }

      double end__ = label_running_probability_[label_running_probability_.size() - 1];
      //std::cout << end__ << std::endl;

      offset_selector_ = GetRandomUniform<double>(0, end__);
    }

    if (apply_label_pixel_mask_) {

      double boost_divisor = 0;

      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] *= 1.0 / label_boost_[l];
        boost_divisor += label_freq[l];
      }

      for (int l = 0; l < nr_labels_; ++l) {
        label_freq[l] *= 1.0 / boost_divisor;
      }

      label_mask_probability_.resize(nr_labels_);
      float mask_divisor = 0;
      for (int l = 0; l < nr_labels_; ++l) {
        label_mask_probability_[l] = 1.0 / label_freq[l];
        mask_divisor = std::max(mask_divisor, label_mask_probability_[l]);
      }
      for (int l = 0; l < nr_labels_; ++l) {
        label_mask_probability_[l] /= mask_divisor;
        LOG(INFO) << "Label " << l << ", mask probability: "
                  << label_mask_probability_[l];
      }
    }
  }

  return 0;
}

void ImageProcessor::SetBlurParams(bool apply, float mean, float std,
                                   int blur_size) {
  apply_blur_ = apply;
  blur_mean_ = mean;
  blur_std_ = std;
  blur_size_ = blur_size;
  blur_random_selector_ = GetRandomNormal<float>(mean, std);
}

void ImageProcessor::SetBorderParams(bool apply, int border_size) {
  apply_border_reflect_ = apply;
  border_size_ = border_size;
}

void ImageProcessor::SetClaheParams(bool apply, float clip_limit) {
  apply_clahe_ = apply;
  clahe_ = cv::createCLAHE();
  clahe_->setClipLimit(clip_limit);
}

void ImageProcessor::SetRotationParams(bool apply) {
  apply_rotation_ = apply;
  //random_rotation_ = random;
  rotation_rand_angle_ = GetRandomOffset(0, 359);
  rotation_rand_ = GetRandomOffset(0, 3);
}

void ImageProcessor::SetPatchMirrorParams(bool apply) {
  apply_patch_mirroring_ = apply;
  patch_mirror_rand_ = GetRandomOffset(0, 2);
}

void ImageProcessor::rotate(cv::Mat& src, double angle, cv::Mat& dst, cv::InterpolationFlags interpolation){
    int len = std::max(src.cols, src.cols);
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt,angle,1.0);

    cv::warpAffine(src, dst, r, cv::Size(len,len), interpolation, cv::BORDER_REFLECT_101);
}

void ImageProcessor::scale_keep_size(cv::Mat& src, double scale, cv::Mat& dst, cv::InterpolationFlags interpolation){
    int len = std::max(src.cols, src.cols);
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, 0, scale);

    cv::warpAffine(src, dst, r, cv::Size(len,len), interpolation, cv::BORDER_REFLECT_101);
}

void ImageProcessor::SetLabelHistEqParams(bool apply, bool patch_prior,
                                          bool mask_prob,
                                          std::vector<float> label_boost) {
  apply_label_hist_eq_ = apply;
  apply_label_patch_prior_ = patch_prior;
  apply_label_pixel_mask_ = mask_prob;
  label_mask_prob_rand_.resize(omp_get_max_threads());
  for (int i = 0; i < omp_get_max_threads(); ++i) {
    label_mask_prob_rand_[i] = GetRandomUniform<float>(0.0, 1.0);
  }
  label_boost_ = label_boost;
}

long ImageProcessor::BinarySearchPatch(double offset) {
  long mid, left = 0;
  long right = label_running_probability_.size();
  while (left < right) {
    mid = left + (right - left) / 2;
    if (offset > label_running_probability_[mid]) {
      left = mid + 1;
    } else if (offset < label_running_probability_[mid]) {
      right = mid;
    } else {
      return left;
    }
  }
  return left;
}

ProcessImageProcessor::ProcessImageProcessor(int patch_size, int nr_labels)
    : ImageProcessor(patch_size, nr_labels) {
}

TrainImageProcessor::TrainImageProcessor(int patch_size, int nr_labels)
    : ImageProcessor(patch_size, nr_labels) {
}

std::vector<cv::Mat> TrainImageProcessor::DrawPatchRandom() {

  // chooses the linear position in the set of all possible patches
  double offset = offset_selector_();

  // absolute ID of the patch location within ALL possible patches
  long abs_id = 0;

  if (apply_label_hist_eq_ && apply_label_patch_prior_) {
    abs_id = BinarySearchPatch(offset);
  } else {
    abs_id = (long) offset;
  }

  // compute the image ID
  // by looping over cum_sum_ and checking where the abs_id is larger
  int img_id = -1;
  for (int k = 0; k < cum_sum_.size(); ++k){
    if (cum_sum_[k] > abs_id){
        // found the image
        img_id = k-1;
        break;
    }
  }
  assert(img_id >= 0);

//  int off_size_x = off_size_x_[img_id];
//  int off_size_y = off_size_y_[img_id];
  // TODO: compute the (random, top-left) location of the patch
//  auto y_gen = GetRandomUniform(0,off_size_y);//(off_size_x * off_size_y) / off_size_x;
//  auto x_gen = GetRandomUniform(0,off_size_x);//abs_id - ((img_id * off_size_x * off_size_y) + (yoff * off_size_x));

//  int yoff = y_gen();
//  int xoff = x_gen();
//  int off_size_x = (image_size_x_ - patch_size_) + 1;
//  int off_size_y = (image_size_y_ - patch_size_) + 1;
//  int img_id = abs_id / (off_size_x * off_size_y);

  int yoff = (abs_id - cum_sum_[img_id]) / off_size_x_[img_id]; //(abs_id -(off_size_x * off_size_y)) / off_size_x;
  int xoff = abs_id - (cum_sum_[img_id] + (yoff * off_size_x_[img_id])); //abs_id - ((img_id * off_size_x * off_size_y) + (yoff * off_size_x));

  cv::Mat &full_image = raw_images_[img_id];
  cv::Mat &full_label = label_images_[img_id];

  int actual_patch_size = patch_size_ + 2 * border_size_;
  int actual_label_size = patch_size_;

  cv::Rect roi_patch(xoff, yoff, actual_patch_size, actual_patch_size);
  cv::Rect roi_label(xoff, yoff, actual_label_size, actual_label_size);

  // Deep copy so that the original image in storage doesn't get messed up
  cv::Mat patch = full_image(roi_patch).clone();
  cv::Mat label = full_label(roi_label).clone();

  if (apply_scaling_){

      // make an upscaled version of the (downscaled) image patch
      cv::Mat scaled_patch;
      cv::Mat scaled_label;

      float scale_ = random_upscale_selector_();
      //std::cout << scale_ << std::endl;

      scale_keep_size(patch, scale_, scaled_patch, cv::INTER_LINEAR);
      scale_keep_size(label, scale_, scaled_label, cv::INTER_NEAREST);

      patch = scaled_patch;
      label = scaled_label;
  }


  if (apply_intensity_shift_) {

      // convert img patch to hsv
      cv::Mat patch_hsv;
      cv::cvtColor(patch, patch_hsv, cv::COLOR_BGR2HSV);

      // apply intensity shift in hsv
      std::vector<cv::Mat> channels;
      cv::split(patch_hsv,channels);

      double range_ = (double) random_intrange_selector_();
      //std::cout << range_ << std::endl;
      channels[0] = channels[0] + range_; // add scalar

      // merge channels back to patch
      cv::merge(channels,patch_hsv);

      if (!use_hsv_){
          cv::cvtColor(patch_hsv, patch, cv::COLOR_HSV2BGR);
      } else {
          // stay in HSV color space
          patch = patch_hsv;
      }
  }

  if (apply_patch_mirroring_) {
    int flipcode = patch_mirror_rand_() - 1;
    cv::Mat mirror_patch;
    cv::Mat mirror_label;
    cv::flip(patch, mirror_patch, flipcode);
    cv::flip(label, mirror_label, flipcode);
    patch = mirror_patch;
    label = mirror_label;
  }

  if (apply_rotation_) {

    cv::Mat rotate_patch;
    cv::Mat rotate_label;
    cv::Mat tmp_patch;
    cv::Mat tmp_label;

//    int rand_angle = rotation_angle_();

//    if (rand_angle != 0){
//        rotate(patch, rand_angle*1.0, rotate_patch, cv::INTER_LINEAR);
//        rotate(label, rand_angle*1.0, rotate_label, cv::INTER_NEAREST);
//        //std::cout << rand_angle << std::endl;
//    } else {
//        // don't change anything
//        rotate_patch = patch;
//        rotate_label = label;
//    }

    switch (rotation_rand_()) {
      case 0:
        rotate_patch = patch;
        rotate_label = label;
        break;
      case 1:
        tmp_patch = patch.t();
        tmp_label = label.t();
        cv::flip(tmp_patch, rotate_patch, 1);
        cv::flip(tmp_label, rotate_label, 1);
        break;
      case 2:
        cv::flip(patch, rotate_patch, -1);
        cv::flip(label, rotate_label, -1);
        break;
      case 3:
        tmp_patch = patch.t();
        tmp_label = label.t();
        cv::flip(tmp_patch, rotate_patch, 0);
        cv::flip(tmp_label, rotate_label, 0);
        break;
    }

    patch = rotate_patch;
    label = rotate_label;

  }

  cv::Rect roi_rot_patch(0, 0, actual_patch_size - image_crop_,
                         actual_patch_size - image_crop_);

  cv::Rect roi_rot_label(0, 0, actual_label_size - label_crop_,
                         actual_label_size - label_crop_);

  patch = patch(roi_rot_patch);
  label = label(roi_rot_label);

  if (apply_blur_) {
    cv::Size ksize(blur_size_, blur_size_);
    float sigma = blur_random_selector_();
    cv::GaussianBlur(patch, patch, ksize, sigma);
  }

  if (apply_label_hist_eq_ && apply_label_pixel_mask_) {
#pragma omp parallel
    {
      std::function<float()> &randprob =
          label_mask_prob_rand_[omp_get_thread_num()];
#pragma omp for
      for (int y = 0; y < patch_size_; ++y) {
        for (int x = 0; x < patch_size_; ++x) {
          label.at<float>(y, x) =
              label_mask_probability_[label.at<float>(y, x)] >= randprob() ?
                  label.at<float>(y, x) : -1.0;
        }
      }
    }
  }

  std::vector<cv::Mat> patch_label;

  if (label_consolidate_) {
#pragma omp parallel for
    for (int y = 0; y < label.rows; ++y) {
      for (int x = 0; x < label.cols; ++x) {
        label.at<float>(y, x) = label.at<float>(y, x) < 0 ?
            -1.0 : label_consolidate_labels_[label.at<float>(y, x)];
      }
    }
  }


  patch_label.push_back(patch);
  patch_label.push_back(label);

  return patch_label;
}

}
