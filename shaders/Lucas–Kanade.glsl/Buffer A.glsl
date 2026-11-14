// This buffer solves for dI/dx, dI/dy, dI/dt

# define FRAMES 2.
# define LOD 0.

float lum( vec4 col ) {
    return dot( col, vec4(.333,.333,.333,.0));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 texel = pow(2.,LOD)/iResolution.xy;
    
    vec4 N = textureLod(iChannel0, uv + vec2( 0., 1.)*texel, LOD);
    vec4 E = textureLod(iChannel0, uv + vec2( 1., 0.)*texel, LOD);
    vec4 S = textureLod(iChannel0, uv + vec2( 0.,-1.)*texel, LOD);
    vec4 W = textureLod(iChannel0, uv + vec2(-1., 0.)*texel, LOD);
    
    vec4 currCol = textureLod(iChannel0, uv, LOD);
    float curr = lum(currCol);
    float prev = textureLod(iChannel1, uv, LOD).a;
    
    // Temporal denoising
    curr = mix(prev, curr, 1./(1.+ FRAMES ) );
    
    float dIdx = lum( (E - W)/2. );
    float dIdy = lum( (N - S)/2. );
    float dIdt = curr-prev;
    
    fragColor = vec4(dIdx, dIdy, dIdt, curr);
}