// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include <stdbool.h>
#include "utils.h"
#include "calcDepthOptimized.h"  
#include "calcDepthNaive.h"  
#include <memory.h>
#include <math.h>

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

#define ABS(x) (((x) < 0) ? (-(x)) : (x))  

  
float displacementO(int dx, int dy)
{
  //float squaredDisplacement = dx * dx + dy * dy;
  //float displacement = sqrt(squaredDisplacement);
  return sqrt(dx * dx + dy * dy);    
}


void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{


  #pragma omp parallel for private(A)

  float A[4];
  //__m128 output =  _mm_setzero_ps();
  
  //int feaw = (2*featureWidth+1)/4*4;
  int feaw = 2*featureWidth+1;

  /*
  for (int x = featureWidth; x < imageWidth-featureWidth; x=x+1)
  {
    for (int y = featureHeight; y < imageHeight-featureHeight; y=y+1)
    {
    */
    //#pragma omp parallel for
for (int x = 0; x < imageWidth; x++)
    {      
      for (int y = 0; y < imageHeight; y++)
  {
    
      if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
      {
        depth[y * imageWidth + x] = 0;
        continue;
      }
      

      float minimumSquaredDifference = -1;
      int minimumDy = 0;
      int minimumDx = 0;

      int maxi_y = (featureHeight-y > -maximumDisplacement) ? featureHeight-y : -maximumDisplacement;
      int maxi_x = (featureWidth-x > -maximumDisplacement) ? featureWidth-x : -maximumDisplacement;
      int mini_y = (imageHeight-featureHeight-y > maximumDisplacement) ? maximumDisplacement : imageHeight-featureHeight-y-1;
      int mini_x = (imageWidth-featureWidth-x > maximumDisplacement) ? maximumDisplacement : imageWidth-featureWidth-x-1;

      for (int dx = maxi_x; dx <= mini_x; dx=dx+1)
      {
        for (int dy = maxi_y; dy <= mini_y; dy=dy+1)
        {
        

      /*for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy=dy+1)
      {
        for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx=dx+1)
        {
          if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
          {
            continue;
          }
*/

          
          float squaredDifference = 0;

          for (int boxX = 0; boxX < feaw/8*8; boxX= boxX + 8)
            {

          

          __m128 output =  _mm_setzero_ps();
          __m128 limage;
          __m128 rimage;  
          __m128 diff;

          
           
          int xminusfeaturewidth = x - featureWidth;
          int leftX = xminusfeaturewidth + boxX;
            int rightX = xminusfeaturewidth + dx + boxX;
          for (int boxY = -featureHeight; boxY <= featureHeight; boxY=boxY+1)
          {
            
              //int leftX = x + boxX - featureWidth;
              //int leftY = y + boxY;
              //int rightX = x + dx + boxX - featureWidth;
              //int rightY = y + dy + boxY;
          int leftY = y + boxY;
          int rightY = y + dy + boxY;
              int leftim = leftY * imageWidth;
          int rightim = rightY * imageWidth;

              int le = leftim + leftX;
              int rig = rightim + rightX;

              limage = _mm_loadu_ps(left + le);
              rimage = _mm_loadu_ps(right + rig);
              diff = _mm_sub_ps(limage, rimage);
              diff = _mm_mul_ps(diff, diff);
              output = _mm_add_ps(output, diff);

              //le+=4;
              //rig+=4;

              limage = _mm_loadu_ps(left + le + 4);
              rimage = _mm_loadu_ps(right + rig + 4);  
              diff = _mm_sub_ps(limage, rimage);
              diff = _mm_mul_ps(diff, diff);  
              output = _mm_add_ps(output, diff);   
            
          }  
              _mm_storeu_ps(A, output);
              squaredDifference += (A[0] + A[1] +A[2]+A[3]);  
              

  }


              int xdx = x + dx;

              if(squaredDifference > minimumSquaredDifference && minimumSquaredDifference != -1){ 
                continue;
              }
              for (int boxXX = (feaw/8*8)-featureWidth; boxXX <= featureWidth; boxXX=boxXX+1){


              int leftXX = x + boxXX;
                //int leftY = y + boxY;
                int rightXX = xdx + boxXX;
              
              
              for (int boxY = -featureHeight; boxY <= featureHeight; boxY = boxY+1)
              {
                
                //int rightY = y + dy + boxY;
                int leftY = y + boxY;
              int rightY = y + dy + boxY;
                float difference = left[leftY * imageWidth + leftXX] - right[rightY* imageWidth +  rightXX];
                squaredDifference += difference*difference;
              }
    
            
          }

              
          if ( (minimumSquaredDifference == -1) || ( (minimumSquaredDifference == squaredDifference) &&
           (displacementO(dx, dy) < displacementO(minimumDx, minimumDy))) || ( (minimumSquaredDifference > squaredDifference) && ((minimumSquaredDifference != squaredDifference)) )  )
          {
            minimumSquaredDifference = squaredDifference;
            //minimumSquaredDifference=-1;   
            minimumDx = dx;
            minimumDy = dy;
           
          }
        
        }
      }

      if (minimumSquaredDifference == -1)
      {

       /* if (maximumDisplacement == 0)
        {
          depth[y * imageWidth + x] = 0;
        }
        else
        {
          depth[y * imageWidth + x] = displacementO(minimumDx, minimumDy);
        }*/
          depth[y * imageWidth + x] = 0;

          

      }
      else
      {
        (maximumDisplacement == 0) ?  ( depth[y * imageWidth + x] = 0 ) : ( depth[y * imageWidth + x] = displacementO(minimumDx, minimumDy) );
      }
    }
  }
}
