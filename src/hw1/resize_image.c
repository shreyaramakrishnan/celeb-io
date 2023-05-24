#include <math.h>
#include "image.h"

// perform nearest neighbor interpolation
float nn_interpolate(image im, int c, float h, float w)
{
    // make sure to round to nearest int via roundf()
    return get_pixel(im, c, (int) roundf(h), (int) roundf(w));
}


// Create a new image that is h x w and the same number of channels as im
// Use nearest-neighbor interpolate to fill in the image
image nn_resize(image im, int h, int w)
{
  // we're taking h and w samples and sampling from (x - 1) * increment

  // solve system of equations aX + b = Y to find a and b
  // the system transforms the original coords to the new
  // top left corner sample of original image and new image are anchored
  // at -.5 , -.5
  // bottom right corner is now full_val - .5 for each image/direction
  // a * -.5 + b = -.5
  // a * (.5 + x - 0.5) + b = .5 +(y - 0.5)
  // a = (x - 0.5) / (y - 0.5)

  // begin h vars
  float h_a = ((im.h * 1.0 ) / (h * 1.0));
  float h_b = (-0.5 - (h_a * -0.5));

  // begin w vars
  float w_a = ((im.w * 1.0) / (w * 1.0));
  float w_b = (-0.5 - (w_a * -0.5));

    // Loop over the pixels and map back to the old coordinates
  image new_im = make_image(im.c, h, w);
  for (int c = 0; c < im.c; c++) {
    for (int h_idx = 0; h_idx < h; h_idx++) {
      for (int w_idx = 0; w_idx < w; w_idx++) {
        float old_h = (h_idx * h_a) + h_b;
        float old_w = (w_idx * w_a) + w_b;
        set_pixel(new_im, c, h_idx, w_idx, nn_interpolate(im, c, old_h, old_w));
      }
    }
  }
  return new_im;
}

// interpolate bilinearly
float bilinear_interpolate(image im, int c, float h, float w)
{
    int V1_w = (int) w; 
    int V1_h = (int) h;  
    int V2_w = ceil(w);
    int V2_h = (int) h;
    int V3_w = (int) w;
    int V3_h = ceil(h);
    int V4_w = ceil(w);
    int V4_h = ceil(h);
    float d1 = w - (V1_w * 1.0);
    float d2 = 1 - d1;
    float d3 = h - (V2_h * 1.0);
    float d4 = 1 - d3;

    float V1 = get_pixel(im, c, V1_h, V1_w);
    float V2 = get_pixel(im, c, V2_h, V2_w);
    float V3 = get_pixel(im, c, V3_h, V3_w);
    float V4 = get_pixel(im, c, V4_h, V4_w);

    float A1 = d2 * d4;
    float A2 = d1 * d4;
    float A3 = d2 * d3;
    float A4 = d1 * d3;

    return (V1 * A1) + (V2 * A2) + (V3 * A3) + (V4 * A4);
}

// Create a new image that is h x w and the same number of channels as im
// Use bilinear interpolate to fill in the image
image bilinear_resize(image im, int h, int w)
{
  // we're taking h and w samples and sampling from (x - 1) * increment

  // solve system of equations aX + b = Y to find a and b
  // the system transforms the original coords to the new
  // top left corner sample of original image and new image are anchored
  // at -.5 , -.5
  // bottom right corner is now full_val - .5 for each image/direction
  // a * -.5 + b = -.5
  // a * (.5 + x - 0.5) + b = .5 +(y - 0.5)
  // a = (x - 0.5) / (y - 0.5)

  // begin h vars
  float h_a = ((im.h * 1.0 ) / (h * 1.0));
  float h_b = (-0.5 - (h_a * -0.5));

  // begin w vars
  float w_a = ((im.w * 1.0) / (w * 1.0));
  float w_b = (-0.5 - (w_a * -0.5));

    // Loop over the pixels and map back to the old coordinates
  image new_im = make_image(im.c, h, w);
  for (int c = 0; c < im.c; c++) {
    for (int h_idx = 0; h_idx < h; h_idx++) {
      for (int w_idx = 0; w_idx < w; w_idx++) {
        float old_h = (h_idx * h_a) + h_b;
        float old_w = (w_idx * w_a) + w_b;
        set_pixel(new_im, c, h_idx, w_idx, bilinear_interpolate(im, c, old_h, old_w));
      }
    }
  }
  return new_im;
}

