#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

// normalize the pixels in an image to sum to 1 within each channel
void l1_normalize(image im)
{
  // float sum = 0;
  // for (int c = 0; c < im.c; c++) {
  //   for (int h = 0; h < im.h; h++) {
  //     for (int w = 0; w < im.w; w++) {
  //       sum += get_pixel(im, c, h, w);
  //     }
  //   }
  // }
  // for (int c = 0; c < im.c; c++) {
  //   for (int h = 0; h < im.h; h++) {
  //     for (int w = 0; w < im.w; w++) {
  //       float old_val = get_pixel(im, c, h, w);
  //       set_pixel(im, c, h, w, old_val / sum);
  //     }
  //   }
  // }
    for (int c = 0; c < im.c; c++) {
      float c_sum = 0.0;
      for (int row = 0; row < im.h; row++) {
        for (int col = 0; col < im.w; col++) {
          c_sum += get_pixel(im, c, row, col);
        }
      }

      for (int h = 0; h < im.h; h++) {
        for (int w = 0; w < im.w; w++) {
          set_pixel(im, c, h, w, get_pixel(im, c, h, w) / c_sum);
        }
      }
    }
}

// return a 1 channel image with each pixel of equal weight, all summing to 1
image make_box_filter(int w)
{
    float val = 1.0 / (w * w);
    // printf("%.6f",val);
    image filter = make_image(1, w, w);
    for (int h = 0; h < w; h++) {
      for (int width = 0; width < w; width++) {
        set_pixel(filter, 0, h, width, val);
      }
    }
    l1_normalize(filter);
    return filter;
}


// If filter and im have the same number of channels then it's just a normal convolution. We sum over spatial and channel dimensions and produce a 1 channel image. UNLESS:
// If preserve is set to 1 we should produce an image with the same number of channels as the input. This is useful if, for example, we want to run a box filter over an RGB image and get out an RGB image. This means each channel in the image will be filtered by the corresponding channel in the filter. UNLESS:
// If the filter only has one channel but im has multiple channels we want to apply the filter to each of those channels. Then we either sum between channels or not depending on if preserve is set.

// Also, filter better have either the same number of channels as im or have 1 channel. I check this with an assert.
image convolve_image(image im, image filter, int preserve) 
{

  assert(im.c == filter.c || filter.c == 1);

  if (preserve == 1) {
    image full_product = make_image(im.c, im.h, im.w);
    for (int c = 0; c < im.c; c++) {
        for (int h = 0; h < im.h; h++) {
          for (int w = 0; w < im.w; w++) {
            // get filter value for pixel
            float area_sum = 0;

            // int f_w_bound = (filter.w / 2);
            // int f_h_bound = (filter.h / 2);
            for (int f_h = 0; f_h < filter.h; f_h++) {
              for (int f_w = 0; f_w < filter.w; f_w++) {
                // int f_w_bound = (filter.w / 2);
                // int f_h_bound = (filter.h / 2);
                // get pixel will clamp
                // int img_h = h + f_h - f_h_bound;
                // int img_h = h + f_h - f_h_bound;
                // int img_w = w + f_w - f_w_bound;
                int img_h = h + f_h - filter.h / 2;
                int img_w = w + f_w - filter.w / 2;
                
                float filter_val;
                if (filter.c == 1) {
                  filter_val =  get_pixel(filter, 0, f_h, f_w);
                } else {
                  filter_val =  get_pixel(filter, c, f_h, f_w);
                }
                float img_val = get_pixel(im, c, img_h, img_w);
                area_sum += (filter_val * img_val);
              }
            }
            set_pixel(full_product, c, h, w, area_sum);
          }
        }
    }
    return full_product;

  } else {
    // condense channels
    image product = make_image(1, im.h, im.w);
        for (int h = 0; h < im.h; h++) {
          for (int w = 0; w < im.w; w++) {
            // get filter value for pixel
            float area_sum = 0;
            for (int f_h = 0; f_h < filter.h; f_h++) {
              for (int f_w = 0; f_w < filter.w; f_w++) {
                // int f_w_bound = (filter.w / 2);
                // int f_h_bound = (filter.h / 2);
                // // get_pixel will clamp
                // int img_h = h + f_h - f_h_bound;
                // int img_w = w + f_w - f_w_bound;
                int img_h = h + f_h - filter.h / 2;
                int img_w = w + f_w - filter.w / 2;
                for (int c = 0; c < im.c; c++) {
                  // area_sum += get_pixel(im, c, img_h, img_w);
                  if (filter.c != 1) {
                    area_sum += get_pixel(im, c, img_h, img_w) * get_pixel(filter, c, f_h, f_w);
                  } else {
                    area_sum += get_pixel(im, c, img_h, img_w) * get_pixel(filter, 0, f_h, f_w);
                  }
                }
              }
              
            }
          set_pixel(product, 0, h, w, area_sum);
        } 
    }
    return product;
  }
}

image make_highpass_filter()
{
    image filter = make_image(1, 3, 3);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 1, 1, 4);
    set_pixel(filter, 0, 2, 1, -1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 2, -1);
    set_pixel(filter, 0, 2, 2, 0);
    return filter;
}

image make_sharpen_filter()
{
    image filter = make_image(1, 3, 3);
    set_pixel(filter, 0, 0, 0, 0);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 1, 1, 5);
    set_pixel(filter, 0, 2, 1, -1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 2, -1);
    set_pixel(filter, 0, 2, 2, 0);
    return filter;
}

image make_emboss_filter()
{
    image filter = make_image(1, 3, 3);
    set_pixel(filter, 0, 0, 0, -2);
    set_pixel(filter, 0, 1, 0, -1);
    set_pixel(filter, 0, 2, 0, 0);
    set_pixel(filter, 0, 0, 1, -1);
    set_pixel(filter, 0, 1, 1, 1);
    set_pixel(filter, 0, 2, 1, 1);
    set_pixel(filter, 0, 0, 2, 0);
    set_pixel(filter, 0, 1, 2, 1);
    set_pixel(filter, 0, 2, 2, 2);
    return filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// The emboss and sharpen filters are designed work with an RGB product image. The highpass filter's goal is to measure magnitudes of gradients
// and therefore the product is simply a magnitude image of the changes in the image. The other images aim to bolden color changes, and don't
// solely focus on magnitude of change. 

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// We must futher refine the high-pass filter in order to generate a better image of edges (likely the purpose of using it).
// One pass will lead to a series of small edges attached to the main that might be unwanted.
//
// For all filters, could theoretically end up with negative values for a pixel, which could be problematic and a post process might be
// normalize all to positive values so that more can handle.

image make_gaussian_filter(float sigma)
{
    float sigma_sq = sigma * sigma;
    int kernel_width = sigma * 6;
    printf("%d", kernel_width);
    if (kernel_width % 2 == 0) { // if n is even
        kernel_width += 1;
    }
    image filter = make_image(1, kernel_width, kernel_width);
    // x is w
    // y is h
    for (int h = 0; h < kernel_width; h++) {
      for (int w = 0; w < kernel_width; w++) {
        // move L to R, center around 0
        int x = w - (kernel_width / 2);
        int y = h - (kernel_width / 2);
        // float exp = -1 * ()
        float val = exp(-1 * ((x * x) + (y * y))/ (2 * sigma_sq)) / (TWOPI * sigma_sq );
        set_pixel(filter, 0, h, w, val);
      }
    }
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
  assert(a.w == b.w && a.h == b.h && a.c == b.c );
  image new_img = make_image(a.c, a.h, a.w);
  for (int c = 0; c < a.c; c++) {
    for (int h = 0; h < a.h; h++) {
      for (int w = 0; w < a.w; w++) {
        float val = get_pixel(a, c, h, w) + get_pixel(b, c, h, w);
        set_pixel(new_img, c, h, w, val);
      }
    }
  }
  return new_img;
}

image sub_image(image a, image b)
{
  assert(a.w == b.w && a.h == b.h && a.c == b.c );
  image new_img = make_image(a.c, a.h, a.w);
  for (int c = 0; c < a.c; c++) {
    for (int h = 0; h < a.h; h++) {
      for (int w = 0; w < a.w; w++) {
        float new_val = get_pixel(a, c, h, w) - get_pixel(b, c, h, w);
        set_pixel(new_img, c, h, w, new_val);
      }
    }
  }
  return new_img;
}

image make_gx_filter()
{
  image filter = make_image(1, 3, 3);
  set_pixel(filter, 0, 0, 0, -1);
  set_pixel(filter, 0, 1, 0, -2);
  set_pixel(filter, 0, 2, 0, -1);
  set_pixel(filter, 0, 0, 1, 0);
  set_pixel(filter, 0, 1, 1, 0);
  set_pixel(filter, 0, 2, 1, 0);
  set_pixel(filter, 0, 0, 2, 1);
  set_pixel(filter, 0, 1, 2, 2);
  set_pixel(filter, 0, 2, 2, 1);
  return filter;
}

image make_gy_filter()
{
  image filter = make_image(1, 3, 3);
  set_pixel(filter, 0, 0, 0, -1);
  set_pixel(filter, 0, 1, 0, 0);
  set_pixel(filter, 0, 2, 0, 1);
  set_pixel(filter, 0, 0, 1, -2);
  set_pixel(filter, 0, 1, 1, 0);
  set_pixel(filter, 0, 2, 1, 2);
  set_pixel(filter, 0, 0, 2, -1);
  set_pixel(filter, 0, 1, 2, 0);
  set_pixel(filter, 0, 2, 2, 1);
  return filter;
}

void feature_normalize(image im)
{
  float min = get_pixel(im, 0, 0, 0);
  float max = get_pixel(im, 0, 0, 0);

  for (int c = 0; c < im.c; c++) {
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float val = get_pixel(im, c, h, w);
        if (val < min) {
          val = min;
        }
        if (val > max) {
          val = max;
        }
      }
    }
  }

  float range = max - min;

    for (int c = 0; c < im.c; c++) {
      for (int w = 0; w < im.w; w++) {
        for (int h = 0; h < im.h; h++) {
            float old_val = get_pixel(im, c, h, w);
            set_pixel(im, c, h, w, (old_val - min) / MAX(range, 0));
        }
      }
    }
}

// goal: return two images, gradient mag, gradient dir
image *sobel_image(image im)
{
    image gx = make_gx_filter();
    image gy= make_gy_filter();

    image gx_conv = convolve_image(im, gx, 0);
    image gy_conv = convolve_image(im, gy, 0);

    image mag = make_image(1, im.h, im.w);
    image dir = make_image(1, im.h, im.w);

    for(int h = 0; h < im.h; h++){
      for(int w = 0; w < im.w; w++){
        // get channel 0 since it's flattened
        float gx_conv_val = get_pixel(gx_conv, 0, h, w);
        float gy_conv_val = get_pixel(gy_conv, 0, h, w);

        float mag_val = sqrt((gx_conv_val * gx_conv_val) + (gy_conv_val * gy_conv_val));
        float dir_val = atan2(gy_conv_val, gx_conv_val);

        set_pixel(mag, 0, h, w, mag_val);
        set_pixel(dir, 0, h, w, dir_val);
      }
    }

    image *result = calloc(2, sizeof(image));
    result[0] = mag;
    result[1] = dir;

    return result;
}

// use magnitude to specify saturation, value, will call sobel_image()
image colorize_sobel(image im)
{

  image* sobel_arr = sobel_image(im);
  image mag = sobel_arr[0];
  image dir = sobel_arr[1];
  feature_normalize(mag);
  feature_normalize(dir);
    
  image product = copy_image(im);
  rgb_to_hsv(product);

  for(int h = 0; h < im.h; h++){
      for(int w = 0; w < im.w; w++){
        // hsv is hue sat val
        // hue is dir
        // val is mag
        // sat is mag
          set_pixel(product, 0, h, w, get_pixel(dir, 0, h, w));
          set_pixel(product, 1, h, w, get_pixel(mag, 0, h, w));
          set_pixel(product, 2, h, w, get_pixel(mag, 0, h, w));
      }
  }

    hsv_to_rgb(product);

    return product;
}
