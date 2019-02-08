#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform float u_TimeOfDay;
uniform vec4 u_Color;

in vec2 fs_Pos;
out vec4 out_Col;

int maxSteps = 500;
float epsilon = 0.0001;
float maxDist = 100.0;

// Bounding Box
struct Box {
  vec3 max;
  vec3 min;
};

float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

float interpNoise2D(float x, float y) { // from slides
    float intX = floor(x);
    float fractX = fract(x);
    float intY = floor(y);
    float fractY = fract(y);

    float v1 = random1(vec2(intX, intY), vec2(1.f, 1.f));
    float v2 = random1(vec2(intX + 1.0f, intY), vec2(1.f, 1.f));
    float v3 = random1(vec2(intX, intY + 1.0f), vec2(1.f, 1.f));
    float v4 = random1(vec2(intX + 1.0, intY + 1.0), vec2(1.f, 1.f));

    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);

    return mix(i1, i2, fractY);
}

float fbm(float x, float y) { // from slides
  float total = 0.0f;
  float persistence = 0.5f;
  float octaves = 10.0;

  for (float i = 0.0; i < octaves; i = i + 1.0) {
      float freq = pow(2.0f, i);
      float amp = pow(persistence, i);
      total += interpNoise2D(x * freq, y * freq) * amp;
  }
  return total;
}

float sawtooth(float x, float freq, float amplitude) {
  return (x * freq - floor(x * freq)) * amplitude;
}

// SDF Operations
float unionSdf(float d1, float d2) {
  return min(d1, d2);
}

float subtractionSdf(float d1, float d2) {
  return max(-d1,d2);
}

float intersectionSdf(float d1, float d2) { 
  return max(d1,d2);
}

float smoothUnionSdf(float d1, float d2, float k) {
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h); 
}

// SDF Shapes
float sdSphere(vec3 p, float radius) {
  return length(p) - radius;
}

float sdVerticalCapsule(vec3 p, float h, float r) {
    p.y -= clamp(p.y, 0.0, h);
    return length(p) - r;
}

float sdEllipsoid(vec3 p, vec3 r) {
    float k0 = length(p / r);
    float k1 = length(p / (r * r));
    return k0 * (k0 - 1.0) / k1;
}

float sdRoundCone(vec3 p, float r1, float r2, float h) {
  vec2 q = vec2(length(p.xz), p.y);
  float b = (r1-r2)/h;
  float a = sqrt(1.0-b*b);
  float k = dot(q,vec2(-b,a));
  
  if( k < 0.0 ) return length(q) - r1;
  if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
      
  return dot(q, vec2(a,b) ) - r1;
}

float sceneSDF(vec3 point) {
  vec3 headTrans = vec3(0.85, -2.0, -1.0);
  float head = sdSphere((point + headTrans), 1.0);

  vec3 neckTrans = vec3(0.0, 5.0, -1.0);
  float neck = sdVerticalCapsule(point + neckTrans, 7.0, 0.5);
  
  vec3 bodyTrans = vec3(-1.5, 5.0, -1.0);
  float body = sdEllipsoid(point + bodyTrans, vec3(2.2, 1.25, 1.25));

  vec3 legFrontLeftTrans = vec3(0.0, 7.5, -0.5);
  float legFrontLeft = sdVerticalCapsule(point + legFrontLeftTrans, 1.5, 0.35);

  vec3 legFrontRightTrans = vec3(0.0, 7.5, -1.5);
  float legFrontRight = sdVerticalCapsule(point + legFrontRightTrans, 1.5, 0.35);

  vec3 legBackLeftTrans = vec3(-2.5, 7.5, -0.5);
  float legBackLeft = sdVerticalCapsule(point + legBackLeftTrans, 1.5, 0.35);

  vec3 legBackRightTrans = vec3(-2.5, 7.5, -1.5);
  float legBackRight = sdVerticalCapsule(point + legBackRightTrans, 1.5, 0.35);

  mat3 tailRot = mat3(vec3(cos(45.0), sin(45.0), 0),
                      vec3(-sin(45.0), cos(45.0), 0),
                      vec3(0, 0, 1)
                      );
  vec3 tailTrans = vec3(2.5, 7.5, -1.0);
  float tail = sdRoundCone(point * tailRot + tailTrans, 0.1, 0.5, 2.0);

  body = unionSdf(body, tail);
  float backLegs = smoothUnionSdf(legBackLeft, legBackRight, 0.4);
  float frontLegs = smoothUnionSdf(legFrontLeft, legFrontRight, 0.4);
  float legs = smoothUnionSdf(backLegs, frontLegs, 0.5);
  float bodyLeg = smoothUnionSdf(body, legs, 1.0);
  return smoothUnionSdf(smoothUnionSdf(head, neck, 1.0), bodyLeg, 1.0);
}

vec4 skyColor() {
  vec3 skyColor, cloudColor;
  float clouds = fbm(fs_Pos.x + (u_Time / 150.0), fs_Pos.y);
  if (u_TimeOfDay == 1.0) {
    skyColor = vec3(121.0 / 255.0, 195.0 / 255.0, 249.0 / 255.0);
    cloudColor = vec3(1.0, 1.0, 1.0);
    clouds -= 0.5;
  } else {
    skyColor = vec3(0.0, 20.0 / 255.0, 63.0 / 255.0);
    cloudColor = vec3(196.0 / 255.0, 197.0 / 255.0, 198.0 / 255.0);
    float lightning = sawtooth((cos(u_Time/ 30.0)), 2.0, 0.5);
    clouds -= lightning;
  }
  return vec4(clouds * skyColor + (1.0 - clouds) * cloudColor, 1.0);
}

vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + epsilon, p.y, p.z)) - sceneSDF(vec3(p.x - epsilon, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + epsilon, p.z)) - sceneSDF(vec3(p.x, p.y - epsilon, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + epsilon)) - sceneSDF(vec3(p.x, p.y, p.z - epsilon))
    ));
    return vec3(1.0, 1.0, 1.0);
}

Box dinosaurTopBox() {
  Box box;
  box.min = vec3(-2.0, -2.0, -2.0);
  box.max = vec3(2.0, 2.0, 2.0);
  return box;
}

Box dinosaurBottomBox() {
  Box box;
  box.min = vec3(-3.0, -3.0, -3.0);
  box.max = vec3(3.0, 3.0, 3.0);
  return box;
}

float intersectCube(vec3 origin, vec3 dir, Box box) { // from slides
  float tNear = - 1000000.0;
  float tFar = 1000000.0;

  // Check each of the x, y, z slabs
  for (int i = 0; i < 3; i++) {
    if (dir[i] == 0.0) {
      if (origin[i] < box.min[i] || origin[i] > box.max[i]) {
        return tFar;
      }
    }
    
    float t0 = (box.min[i] - origin[i]) / dir[i];
    float t1 = (box.max[i] - origin[i]) / dir[i];

    if (t0 > t1) {
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }

    if (t0 > tNear) {tNear = t0;}
    if (t1 < tFar) {tFar = t1;}
  }

  // bounding box missed
  if (tNear > tFar) {
    return tFar;
  }
  return tNear;
}

float bvh(vec3 point, vec3 dir, Box list[2]) {
  float t = 1000000.0;
  for (int i = 0; i < list.length(); i++) {
    float newT = intersectCube(point, dir, list[i]);
    if (t > newT) {
      t = newT;
    }
  }
  return t;
}

float rayMarch(vec3 rayDir) {
  Box boxList[2];
  boxList[0] = dinosaurBottomBox();
  boxList[1] = dinosaurTopBox();

  if (bvh(u_Eye, rayDir, boxList) > 100.0) {
    return 100000.0;
  }

  float depth = 0.0;
  for (int i = 0; i < maxSteps; i++) {

    float dist = sceneSDF(u_Eye + depth * rayDir);
    if (dist < epsilon) {
        // We're inside the scene surface!
        return depth;
    }
    // Move along the ray
    depth += dist;

    if (depth >= maxDist) {
        // Gone too far
        return maxDist;
    }
  }
  return maxDist;
}

vec3 castRay() {
  vec3 F = normalize(u_Ref - u_Eye);
  vec3 R = normalize(cross(F, u_Up));
  float len = length(u_Ref - u_Eye);
  float aspect = u_Dimensions.x / u_Dimensions.y;
  float fov = radians(75.0);

  vec3 V = u_Up * len * tan(fov / 2.0);
  vec3 H = R * len * aspect * tan(fov / 2.0);
  vec3 p = u_Ref + fs_Pos.x * H + fs_Pos.y * V;
  vec3 ray_Dir = normalize(p - u_Eye);
  return ray_Dir;
}

vec4 lambert(vec3 color, vec3 normal) {
  vec4 diffuseColor = vec4(color, 1.0);
  vec3 lightDir = vec3(1.0, 1.0, -0.7);
  float diffuseTerm = dot(normalize(normal), normalize(lightDir));
  diffuseTerm = clamp(diffuseTerm, 0.0, 1.0);
  float ambientTerm = 0.2;
  float lightIntensity = diffuseTerm + ambientTerm; 
  if (u_TimeOfDay == 0.0) {
    float lightning = sawtooth((cos(u_Time/ 30.0)), 2.0, 0.5);
    lightning = smoothstep(lightning, 2.0, 1.0);
    lightIntensity /= (lightning * 5.0);
  }
  return vec4(diffuseColor.rgb * lightIntensity, diffuseColor.a);
}

void main() {
  vec3 dir = castRay();
  float depth = rayMarch(dir);

  if (depth < maxDist) {
    vec3 color = vec3(u_Color);
    vec3 pos = u_Eye + depth * dir;
    vec3 normal = estimateNormal(pos);
    out_Col = lambert(color, normal);
  } else {
    out_Col = skyColor();
  }
}