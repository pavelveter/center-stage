void mainImage( out vec4 O, in vec2 I )
{
    vec2 R = iResolution.xy,
         M = iMouse.xy,
         U = I/R.xy,
         P = vec2(120., 120.*R.y/R.x),
         C = floor(U*P)/P;
    U -= C;
    U *= (M/R-.5)*6.; // distortion based on mouse
    U += C;
    O = vec4(texture(iChannel0, U).rgb,1.);
}