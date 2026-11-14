// image shapening pass 
// set sharpness variable to zero to see it without shaprness

float sharpnessSize = 1.;   
float sharpness = 1.5;
float hbfkernel[9] = float[9](0., -1., 0., -1., 5., -1., 0., -1., 0.);


vec4 Convolve(float kernel[9], vec2 uv, vec2 texelSize){
    vec4 col = vec4(0.0); // hyper-stage: zero-init (WebGL zeroes locals, desktop GL does not)
    int index = 0;
    for (float i=-1.; i<=1.; i++){
        for (float j=-1.; j<=1.; j++){
            col += texture(iChannel0, uv + vec2(texelSize.x * j,texelSize.y * i)) * kernel[index];
            index++;
        }
    }
    return col;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 texelSize = 1./iResolution.xy*sharpnessSize;

    vec4 col = texture(iChannel0,uv);
    vec4 sharpCol = Convolve(hbfkernel, uv, texelSize);
    
    
    fragColor = mix(col, sharpCol, sharpness);
}