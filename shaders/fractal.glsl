const float NUM_STRIPES = 25.0;
const float STRRENGTH = 1.0;
const float SOFTNESS = 0.0005;

// TODO:
// - compress the image
// - add blur dependent on local strength

float displacement(float x, float num_stripes, float strength) {

    float modulus = 1.0 / num_stripes;
    
    return mod(x, modulus) * strength;
}

float fractal_glass(float x) {

    float d = 0.0;
    for (int i = -5; i <= 5; i++) {
       d += displacement(x + float(i) * SOFTNESS, NUM_STRIPES, STRRENGTH);
    }

    d = d / 11.0;
    
    return x + d;
}



void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
 
    uv.x = fractal_glass(uv.x);
    
    
    vec3 col = texture(iChannel0, uv).rgb;

    // Output to screen
    fragColor = vec4(col, 1.0);
}



