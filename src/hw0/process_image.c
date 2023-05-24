#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"


// clamps the actual value to the nearest value in range 0-limit (inclusive)
int clamp_pixel(int limit, int actual) {
    if (actual < 0) {
      return 0;
    } else {
      if (actual > limit) {
        return limit;
      } else {
        return actual;
      }
    }
}

// return clamp if not in, otherwise return pixel value
float get_pixel(image im, int c, int h, int w)
{
    // TODO Fill this in
    // begin by clamping nums as needed
    int c_val = clamp_pixel(im.c - 1, c);
    int h_val = clamp_pixel(im.h - 1, h);
    int w_val = clamp_pixel(im.w - 1, w);

    int index = w_val + (h_val * im.w) + (c_val * im.w * im.h);
    return im.data[index];
}

int in_range(int limit, int actual) {
  if (actual < 0 || actual >= limit) {
    return 0;  // false
  } else {
    return 1;  // true
  }
}

void set_pixel(image im, int c, int h, int w, float v)
{
    // TODO Fill this in
    if (in_range(im.c, c) && in_range(im.h, h) && in_range(im.w, w)) {
      int index = w + (h * im.w) + (c * im.w * im.h);

      im.data[index] = v;
    }
    // else just return without doing anything
}

image copy_image(image im)
{
    image copy = make_image(im.c, im.h, im.w);

    int size = 4 * (im.c) * (im.w) * (im.h);
    memcpy(copy.data, im.data, size);
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(1, im.h, im.w);
    // TODO Fill this in
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float r = get_pixel(im, 0, h, w);
        float g = get_pixel(im, 1, h, w);
        float b = get_pixel(im, 2, h, w);

        float new_val = (0.299 * r) + (0.587 * g) + (.114 * b);
        gray.data[w + (h * im.w)] = new_val;
      }
    }
    
    return gray;
}

void shift_image(image im, int c, float v)
{
  for (int w = 0; w < im.w; w++) {
    for (int h = 0; h < im.h; h++) {
      float val = get_pixel(im, c, h, w);
      float new_val = val + v;
      set_pixel(im, c, h, w, new_val);
    }
  }
}

void clamp_image(image im)
{
  // TODO Fill this in
  for (int c = 0; c < im.c; c++) {
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float val = get_pixel(im, c, h, w);
        float new_val = val;
        if (val < 0) {
          new_val = 0;
        } else if (val > 1) {
          new_val = 1.0;
        }
        set_pixel(im, c, h, w, new_val);
      }
    }
  }
}

// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float r = get_pixel(im, 0, h, w);
        float g = get_pixel(im, 1, h, w);
        float b = get_pixel(im, 2, h, w);

        float V = three_way_max(r, g, b);
        float m = three_way_min(r, g, b);
        float c = V - m;
        float s = 0;
        if (V == 0) {
          s = 0;
        } else {
          s = c / V;
        }

        float hue = 0;
        if (c != 0) {
          if (V == r) {
            hue = (g - b) / c;
          } else if (V == g) {
            hue = ((b - r) / c) + 2;
          } else {  // == blue
            hue = ((r - g) / c) + 4;
          }
        }

        if (hue < 0) {
          hue = (hue / 6) + 1;
        } else {
          hue = hue / 6;
        }
        // .data[w + (h * im.w)] = new_val;
        set_pixel(im, 0, h, w, hue);
        set_pixel(im, 1, h, w, s);
        set_pixel (im, 2, h, w, V);
      }
    }

}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    for (int w = 0; w < im.w; w++) {
      for (int h = 0; h < im.h; h++) {
        float hue = get_pixel(im, 0, h, w);
        float sat = get_pixel(im, 1, h, w);
        float val = get_pixel(im, 2, h, w);  // max of one of the components

        hue *= 360;

        float r, g, b;
        float c = val * sat;
        float h_prime = hue / 60;
        float x = c * (1 - fabs((fmod(h_prime, 2.0) - 1)));
        if (h_prime >= 0 && h_prime < 1) {
          r = c;
          g = x;
          b = 0;
        } else if (h_prime >= 1 && h_prime < 2) {
          r = x;
          g = c;
          b = 0;
        } else if (h_prime >= 2 && h_prime < 3) {
          r = 0;
          g = c;
          b = x;
        } else if (h_prime >= 3 && h_prime < 4) {
          r = 0;
          g = x;
          b = c;
        } else if (h_prime >= 4 && h_prime < 5) {
          r = x;
          g = 0;
          b = c;
        } else if (h_prime >= 5 && h_prime < 6) {
          r = c;
          g = 0;
          b = x;
        } else {
          r = 0;
          g = 0;
          b = 0;
        }

        float m = val - c;
        r += m;
        g += m;
        b += m;

        set_pixel(im, 0, h, w, r);
        set_pixel(im, 1, h, w, g);
        set_pixel (im, 2, h, w, b);
      }
    }

}

void scale_image(image im, int c, float v) {
  for (int w = 0; w < im.w; w++) {
    for (int h = 0; h < im.h; h++) {
      float val = get_pixel(im, c, h, w);
      float new_val = val * v;
      if (new_val > 1.0) {
        new_val = 1.0;
      }
      set_pixel(im, c, h, w, new_val);
    }
  } 
}