float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

float inkSurrounding(vec2 uv, int r){
    vec2 res = iChannelResolution[0].xy;
    vec2 texel = vec2(1) / res;
    
    float v = 0.0;
    float w = 0.0;
    for(int x = -r; x < r; x++)
    for(int y = -r; y < r; y++){
        vec2 offset = vec2(x, y);
        float lum = luminance(texture(iChannel0, uv + texel * offset).xyz);
        float weight = 0.0;
        if(x != 0 || y != 0){
            weight = (1.41 * float(r)) / length(offset);
        }else{
            weight = 1.0;
        }
        
        v += lum / weight;
        w += weight;
    }
    float b = v / w;
    return smoothstep(0.2, 0.3, b);
}

float hash21(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
float noise21(vec2 p){
    vec2 bl = floor(p);
    vec2 tr = ceil(p);
    vec2 br = vec2(tr.x, bl.y);
    vec2 tl = vec2(bl.x, tr.y);
    
    vec2 t = fract(p);
    
    float blv = hash21(bl);
    float trv = hash21(tr);
    float brv = hash21(br);
    float tlv = hash21(tl);
    
    return mix(mix(blv, brv, t.x), mix(tlv, trv, t.x), t.y);
}

float fbm(vec2 p){
    float freq = 1.0;
    float amp = 1.0;
    float v = 0.0;
    for(int i = 0; i < 10; i++){
        v += noise21(p * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return v * 0.5;
}

float edgeDetect(sampler2D tex, vec2 uv, vec2 texel, float threshold) {
    float c  = luminance(texture(tex, uv).rgb);
    vec2 m = vec2(1.07, 1.07);//if your stamp is seeming a little light on the edges in one direction
    float l  = luminance(texture(tex, uv + vec2(-texel.x, 0.0)).rgb) * m.x;
    float r  = luminance(texture(tex, uv + vec2( texel.x, 0.0)).rgb) * m.x;
    float u  = luminance(texture(tex, uv + vec2(0.0,  texel.y)).rgb) * m.y;
    float d  = luminance(texture(tex, uv + vec2(0.0, -texel.y)).rgb) * m.y;

    float diff = 0.0;
    diff = max(diff, abs(c - l));
    diff = max(diff, abs(c - r));
    diff = max(diff, abs(c - u));
    diff = max(diff, abs(c - d));

    return step(threshold, diff);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    //uv.x = uv.x * -1.0 + 1.0;
    // Main texture in ShaderToy
    vec3 tex = texture(iChannel0, uv).rgb;
    float lum = luminance(tex);

    vec2 texel = 1.0 / iResolution.xy;

    // Detect outlines from local contrast
    float edge = edgeDetect(iChannel0, uv, texel, 0.15);
    
    float noiseSquish = 0.3; //squish the noise. 1 or more -> 1-noiseSquish, and vice versa
    //increasing noiseSquish makes it more defined in middle, but makes it lose definition
    //in especially dark or light areas
    
    // Fill rule: dark parts become stamp body
    float noise = hash21(floor(uv * 1000.0)) * (1.0 - noiseSquish * 2.0) + noiseSquish;
    float fill = 1.0 - smoothstep(noise - 0.07, noise + 0.07, lum);

    fill = max(fill, edgeDetect(iChannel0, uv, texel, 0.1));
    

    // Combine fill and outline
    float stamp = max(fill, edge);
    float stmpHeightFlux = fbm(uv * 10.0);

    // soft organic expansion
    float stampBleed = smoothstep(0.7, 1.0, stmpHeightFlux);
    stampBleed = smoothstep(noise - 0.2, noise + 0.2, stampBleed);

    // only expand existing ink
    float surroundingInk = inkSurrounding(uv, 3);
    stamp = max(stamp, stampBleed * surroundingInk);
    
    stamp += (fbm(uv * 5.0) * 0.5 - 0.25) * lum; //i forgot what this does but might as well keep it for fun
    
    stamp = clamp(stamp, 0.0, 1.0);
    
    vec3 paper = vec3(0.95, 0.92, 0.85);
    vec3 ink   = vec3(0.65, 0.10, 0.10);
    //ink = tex; //this looks very strange its not even a stamp anymore
    vec3 col = mix(paper, ink, stamp);
    fragColor = vec4(col, 1.0);
}