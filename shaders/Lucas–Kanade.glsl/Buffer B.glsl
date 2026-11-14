// Lucas-Kanade Method
// Implemented straight from Wikipedia https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method

#define halfSize 5.

// Trace
float tr2( mat2 A ) {
    return A[0][0] + A[1][1];
}

float det( mat2 A ) {
    return (A[0][0] * A[1][1]) - (A[1][0] * A[0][1]);
}

// 3Blue1Brown https://www.youtube.com/watch?v=e50Bj7jn9IQ
vec2 eigenValues( mat2 A ) {
    float m = .5*tr2(A);
    float p = det(A);
    return vec2( m + sqrt(m*m-p), m - sqrt(m*m-p) );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec2 texel = 1./iResolution.xy;
    
    mat2 structureTensor = mat2(0.,0.,0.,0.);
    vec2 Atb = vec2(0.,0.);
    for( float i = -halfSize; i < halfSize; i++) {
        for( float j = -halfSize; j < halfSize; j++) {
            vec2 loc = uv + vec2(i,j)*texel;
            vec2 dis = loc - uv;
            float weight = exp( - dot(dis,dis)/3. );
            vec4 deriv = texture(iChannel0, loc);
            structureTensor += mat2( weight*deriv.x*deriv.x, weight*deriv.x*deriv.y, weight*deriv.x*deriv.y, weight*deriv.y*deriv.y);
            Atb -= vec2( weight*deriv.x*deriv.z, weight*deriv.y*deriv.z );
        }
    }
    vec2 motion = inverse(structureTensor)*Atb;
    
    // Remove bad features
    vec2 e = eigenValues( structureTensor );
    if( e.x < 0.001 || e.y < 0.001 ) {
        motion = vec2(0.,0.);
    } 
    
    fragColor = vec4(motion,0.,1.0);
}