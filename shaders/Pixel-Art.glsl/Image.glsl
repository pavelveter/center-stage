// pixelation and color quantization
// hyper-stage: 1=noise 2=Buffer B

float quantlevel = 2.;
float contrast = 1.2;
float brightness = 0.;
float saturation = 2.;

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    
    
    //float M = floor(iMouse.x/iResolution.x*5.);
    float quant = pow(2.,quantlevel);
    uv = floor(uv*(iResolution.xy/quant))/(iResolution.xy/quant);
    
    vec4 col = texture(iChannel2, uv);
    
    col = (col-0.5)*contrast+0.5+brightness;
    
    ivec2 noiseSize = textureSize(iChannel1, 0);
    float noisePixelSize = quant;
    ivec2 coord = ivec2(
        int(mod(fragCoord.x/noisePixelSize, float(noiseSize.x))),
        int(mod(fragCoord.y/noisePixelSize, float(noiseSize.y)))
    );
    vec4 noise = texelFetch(iChannel1,coord,0)*2. - 1.;
    
    float noiseAmm = 0.02;
    
    col += noise.x*noiseAmm;
    
    // color quantization
    float ncol = 10.;
    col = floor(col*ncol)/ncol + 1./(2.*ncol);
    
    fragColor = change_saturation( col, saturation);
    
    // tanh tone mapping, try removing it 
    fragColor = tanh(fragColor*1.2);
}