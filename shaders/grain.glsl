
float grain (vec2 st) {
    return fract(sin(dot(st.xy, vec2(17.0,180.)))* 2500. + iTime);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    //Coords
	vec2 uv = fragCoord.xy / iResolution.xy;
    
    //Produce some noise based on the coords
    vec3 grainPlate = vec3(grain(uv));
    
    //Get the webcam
    vec4 webcam = texture(iChannel0, uv);
    
    //Mix the two signals together
    vec3 mixer = mix(webcam.rgb, grainPlate, .1);
	
    
    fragColor = vec4(mixer,1.0); 
}