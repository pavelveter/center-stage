
// scaling is such that a
// circle of this radius
// fits on screen:
float radius = .5;

// click on screen to
// enable color shifting.

// the cogwheel icon in the
// texture preview below 
// allows other filtering
// or warp settings.

vec2 cmul(vec2 a, vec2 b) {
  return mat2(a, -a.y, a.x)*b;
}
vec2 sqr(vec2 x) {
  return x*x;
}

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
vec3 hsv(float h, float s, float v) {
  return v * mix(vec3(1), clamp(abs(6.*fract(h + vec3(3,2,1)/3.) - 3.) - 1., 0., 1.), s);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
  float w = min(iResolution.x, iResolution.y);
  vec2 uv = (2.*fragCoord-iResolution.xy)/w;
  vec2 m = iMouse.xy/iResolution.xy;
    
  // comment out if your mobile cam does not mirror:
  //uv.x = -uv.x;
  vec2 z = uv*radius;
  
  // comment out all of the following for no distortion:
  //
  // burning ship formula,
  // use with mandelbrot formula:
  //z.x = abs(z.x);
  //z.y = abs(z.y);
  //
  // mandelbrot formula:
  z = cmul(z, z);
  
  vec2 ts = vec2(textureSize(iChannel0,1));
  float aspect = ts.x/ts.y;
  vec2 tv = z;
  tv.y *= aspect;
  tv = tv*.5+.5;
  // no transform if mapped beyond texture
  if(fract(tv) != tv) {
    tv = uv;
    tv.y *= aspect;
    tv = tv*.5+.5;
  }
  vec4 t = texture(iChannel0, tv);
  
  // color rotation
  t.rgb = rgb2hsv(t.rgb);
  t.r = mod(t.r + m.x, 1.);
  t.rgb = hsv(t.r, t.g, t.b);
  
  fragColor = t;
  //fragColor.rg += fragColor.b;
  //fragColor.rgb = pow(fragColor.rgb, vec3(1.,2.,2.));
  // fragColor = pow(fragColor, vec4(2.));
}
