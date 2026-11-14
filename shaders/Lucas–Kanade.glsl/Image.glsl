/*

This is mostly a straightforard implementation of the Lucas-Kanade method
for sparse optical flow. I'm also using the eigen values to reject bad features.

I've also added some temporal denoising to the webcam input, and the number of
accumulated frames can be changed in Buffer A.

The window size can be set in Buffer B

*/


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 mov = textureLod(iChannel0, uv, 0.).xy;
    
    float mag = length(mov.xy);
    float ang = atan(mov.y, mov.x)/(2.*3.14159) + .5;
    
    // Remove noisey small values and scale down
    mag = mag * .5 * smoothstep(0., 1., mag);
    
    vec3 col = vec3( ang, 1., mag );
    col = hsv2rgb(col);
    
    fragColor = vec4(col,1.0);
}