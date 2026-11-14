vec4 get_hsv( vec4 a )
{
    float cmax = max( max(a.x,a.y), a.z );
    float cmin = min( min(a.x,a.y), a.z );
    float del = cmax-cmin;
    
    float hue;
    if (del == 0.) {
        hue = 0.;
    } else if (cmax == a.r) {
        hue = 60.*(a.g-a.b)/del;
    } else if (cmax == a.g) {
        hue = 60.*((a.b-a.r)/del + 2.);
    } else {
        hue = 60.*((a.r-a.g)/del + 4.);
    }
    
    
    float l = (cmax+cmin)/2.;
    
    float sat;
    if (del == 0.) {
        sat = 0.;
    } else {
        del/(1.- abs(2.*l-1.));
    }
    
    return vec4(hue,sat,l,1.);
} 

vec3 rgb2hsv(vec3 c)    // don't ask why there are two function to get hsv from rgb
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec4 change_saturation( vec4 a, float s ) {
    return mix( get_hsv(a).zzzz, a, s );
}

