// PIXEL_SIZE 2.0: a 1px checkerboard does not survive the virtual camera's
// video encoding (turns to blotchy spots); 2px dither is codec-stable.
#define PIXEL_SIZE 2.0
#define CONTRAST_BOOST 0.98
#define BRIGHTNESS_OFFSET -0.09

const vec3 COLOR_DARK = vec3(0.157, 0.078, 0.188);  // Тёмно-фиолетовый (#281430)
const vec3 COLOR_LIGHT = vec3(0.949, 0.478, 0.176); // Ярко-оранжевый (#F27A2D)

const int bayer[64] = int[64](
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21
);

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 pixelatedCoord = floor(fragCoord / PIXEL_SIZE);
    vec2 uv = (pixelatedCoord * PIXEL_SIZE) / iResolution.xy;

    vec3 texColor = texture(iChannel0, uv).rgb;

    float luma = dot(texColor, vec3(0.299, 0.587, 0.114));

    luma = (luma - 0.5) * CONTRAST_BOOST + 0.5 + BRIGHTNESS_OFFSET;
    luma = clamp(luma, 0.0, 1.0);

    int x = int(mod(pixelatedCoord.x, 8.0));
    int y = int(mod(pixelatedCoord.y, 8.0));
    
    float ditherLimit = float(bayer[x + y * 8]) / 64.0;

    float ditherResult = step(ditherLimit, luma);

    vec3 finalColor = mix(COLOR_DARK, COLOR_LIGHT, ditherResult);

    fragColor = vec4(finalColor, 1.0);
}