# CIS 566 Homework 2: Implicit Surfaces
## Catherine Yang, PennKey: catyang

## http://catyang97.github.io/hw02-raymarching-sdfs

## Flying Dinosaur
![](day.png)

---
## References
- IQ- SDF combination operations and primitives: http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
- Jamie Wong- Ray Marching Techniques and Normal Calculations: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
- Ray Marching: CIS 460 and CIS 566 slides
- Toolbox Functions: Toolbox slides

---
## Techniques

### Dinosaur - Geometry and Color
The dinosaur is made of the following SDF shapes:
- Sphere for the head
- Vertical Capsules for the neck and legs
- Round Cone (rotated) for the tail
- Ellipsoid for the body

The shapes were combined through Union and Smooth Blending.

The color of the dinosaur is applied through lambertian shading, and the color can be adjusted through the GUI. The normal that is part of the lambertian shading calculation is estimated using Jamie Wong's calculation which takes the current point on the surface. 

### Sky - Animation and Color
- The sky is a combination of a sky color and clouds. The clouds are created using FBM that uses the time in its calculation. 
- In the 'Night' mode, there is animated lightning as well, which is created using the Sawtooth function to sharply toggle the weight of the sky color and the cloud color. In order to change the color of the dinosaur to reflect the lightning, Smooth Step is applied to the Sawtooth function and this value is used to adjust the light intensity.

![](night1.png) ![](night2.png)

### Raymarching
The raymarching technique discussed in the slides and the resources listed above is used to find the implicit surfaces. A ray is cast using the eye position, reference point position, up vector of the camera, and screen width and height. Then, the direction of this ray is used for raymarching. There is a BVH containing the geometry, with one bounding box for the top of the dinosaur (head and neck) and one for the remaining parts. The BVH is an optimization for raymarching. The rayMarch function returns the depth where the surface is and this is used to calculate the normal at that point on the surface.
