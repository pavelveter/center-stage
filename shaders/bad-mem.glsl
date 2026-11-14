//////////////////////////////////////////////////////////////
//            2600++-style color garbage shader             //
//////////////////////////////////////////////////////////////

// Toggle effect
#define ENABLE_2600PP     1      // set to 0 to disable

// Size of the coarse tiles (in pixels)
#define TILE_W            8.0
#define TILE_H            6.0

// How often tiles become heavy glitch blocks (0..1)
#define GLITCH_DENSITY    0.9

// How strong horizontal scanline color jitter is
#define SCANLINE_JITTER   0.3

// Master random seed
#define RAND_SEED         341.73

//////////////////////////////////////////////////////////////

float rand(float x){
    return fract(sin(x * 12.9898 + RAND_SEED) * 43758.5453123);
}

float rand2(vec2 p){
    return rand(dot(p, vec2(37.0, 57.0)));
}

// Very small Atari-ish palette
vec3 atariPalette(float n){
    int idx = int(floor(fract(n) * 8.0)); // 8 colors
    if (idx == 0) return vec3(0.0, 0.0, 0.0);    // black
    if (idx == 1) return vec3(0.9, 0.1, 0.1);    // red
    if (idx == 2) return vec3(1.0, 0.6, 0.0);    // orange
    if (idx == 3) return vec3(1.0, 1.0, 0.0);    // yellow
    if (idx == 4) return vec3(0.0, 0.8, 0.4);    // green
    if (idx == 5) return vec3(0.0, 0.5, 1.0);    // blue
    if (idx == 6) return vec3(0.7, 0.2, 0.9);    // purple
    return           vec3(0.9, 0.9, 0.9);        // white-ish
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 res = iResolution.xy;
    vec2 uv  = fragCoord / res;

    vec3 src = texture(iChannel0, uv).rgb;

#if ENABLE_2600PP

    // --- Tile coordinates and local position ---
    vec2 tileSize = vec2(TILE_W, TILE_H);
    vec2 tileId   = floor(fragCoord / tileSize);
    vec2 local    = fract(fragCoord / tileSize);   // 0..1 inside tile

    float W = res.x;
    float H = res.y;

    // Per-tile random seed
    float sTile = rand2(tileId);

    // Basic sample coord: center of the tile
    vec2 basePos = (tileId + 0.5) * tileSize;
    basePos = clamp(basePos, vec2(0.0), res - 1.0);

    // Random offset inside the tile (so we don't always sample the center)
    vec2 jitter = (vec2(rand(sTile+1.0), rand(sTile+2.0)) - 0.5) * tileSize;
    vec2 samplePos = basePos + jitter * 0.4;  // 0.4 so we stay mostly inside

    samplePos = clamp(samplePos, vec2(0.0), res - 1.0);
    vec2 sampleUV = samplePos / res;
    vec3 c = texture(iChannel0, sampleUV).rgb;

    // Per-scanline color jitter (gives horizontal “video noise”)
    float scanSeed = floor(fragCoord.y / TILE_H);
    float scanJit  = SCANLINE_JITTER * (rand(scanSeed + 99.0) - 0.5);

    // Quantize value to pick palette index
    float lum  = dot(c, vec3(0.299, 0.587, 0.114));
    float palN = lum + scanJit;

    vec3 palColor = atariPalette(palN);

    // Decide glitch mode for this tile
    float rMode = rand(sTile + 10.0);

    vec3 outCol;

    if (rMode < GLITCH_DENSITY) {
        // Heavy glitch tile
        float pattern = rand2(tileId + floor(local * 4.0)); // 4x4 subcells
        if (pattern < 0.33) {
            // pure palette color
            outCol = palColor;
        } else if (pattern < 0.66) {
            // swapped channels + palette
            vec3 swapped = c.bgr;
            outCol = mix(swapped, palColor, 0.7);
        } else {
            // random bright color blended with source
            vec3 rnd = atariPalette(rand(sTile + pattern * 37.0));
            outCol = mix(c, rnd, 0.8);
        }
    } else {
        // Mild glitch: quantize original color to palette + some noise
        outCol = mix(c, palColor, 0.6);
    }

    fragColor = vec4(outCol, 1.0);

#else

    fragColor = vec4(src, 1.0);

#endif
}
