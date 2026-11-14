// median denoise pass. Replace the luminance of each pixel with the median value of luminance in a kernel
// set denoiseLevel to zero to see it without denoising 

const int denoiseLevel = 2;    
const int denoiseKernelSize = (denoiseLevel*2+1)*(denoiseLevel*2+1);

void countSort(inout float arr[denoiseKernelSize], float exp){
    int count[10] = int[10](0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // hyper-stage: zero-init (WebGL zeroes locals, desktop GL does not)
    float outArr[denoiseKernelSize];
    
    for(int i=0; i<denoiseKernelSize; i++){
        count[int(mod(arr[i]*exp,10.))]++;
    }
    
    for (int i=1; i<10; i++){
        count[i] += count[i-1];
    }
    
    for (int i=denoiseKernelSize-1; i>=0; i--){
        outArr[count[int(mod(arr[i]*exp,10.))] - 1] = arr[i];
        count[int(mod(arr[i]*exp,10.))]--;
    }
    
    for (int i=0; i<denoiseKernelSize; i++){
        arr[i] = outArr[i];
    }
}

void radixSort(inout float arr[denoiseKernelSize]){
    for (float i=1000.;i>=10.;i/=10.){
        countSort(arr, i);
    }
}

float findmedian(vec2 uv, vec2 texelSize){
    float lumaArr[denoiseKernelSize];
    int index=0;
    //create sorted array lumaArr
    for (int i=-denoiseLevel; i<=denoiseLevel; i++){
        for (int j=-denoiseLevel; j<=denoiseLevel; j++){
            vec4 col = texture(iChannel0, uv + vec2(texelSize.x * float(j),texelSize.y * float(i)));
            float colLuma = rgb2hsv(col.xyz).z;
            lumaArr[index] = colLuma;
/*            for (int k=index; k>0; k--){
                if (lumaArr[k-1] < colLuma) break;
                lumaArr[k] = lumaArr[k-1];
                lumaArr[k-1] = colLuma;
            } */
            index++;
        }
    }
    radixSort(lumaArr);
    
    return lumaArr[(denoiseKernelSize-1)/2];
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 texelSize = 1./iResolution.xy;
    
    vec3 hsvcol = rgb2hsv(texture(iChannel0,uv).xyz);
    hsvcol.z = findmedian(uv, texelSize);
    
    fragColor = vec4(hsv2rgb(hsvcol),1.);
}