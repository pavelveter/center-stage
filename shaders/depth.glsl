#define FOV_SCALE .5
#define ITER 96
#define EPSILON .03
#define NORMAL_EPSILON .02
#define CAMERA_SPEED .5

#define NO_SHADING
#define SHADOW

const vec3 ambient = vec3(.7);
const vec3 diffuse = vec3(.3);
const float minBright = 0.3;
const float intensity = 7.;


float ground(in vec3 rp) {
    
    float d = rp.y;
    
/*    float intensity = ((cos(rp.x*4.)*.3+cos(rp.z*4.)*.3)+
                       (cos(rp.x*.45345)+cos(rp.z*.4324)) + 4.)*1.;*/
    //if (length(max(abs(rp.xz)-vec2(5.),0.)) < 0.001) 
        d -= max(0.,length(texture(iChannel0,rp.xz*0.1).xyz)-minBright)*intensity;
        
    return d*(1./intensity);
}

float df(in vec3 rp) {

    return ground(rp);
}
const vec3 ne = vec3(NORMAL_EPSILON,0.,0.);
vec3 normal(in vec3 rp) {
    return normalize(vec3(df(rp+ne)-df(rp-ne),
                          df(rp+ne.yxz)-df(rp-ne.yxz),
                          df(rp+ne.yzx)-df(rp-ne.yzx)));
}


const int AO_ITERATIONS= 4;
const float AO_PRECISION = 0.2;
const float AO_INTENSITY = 0.7;
const float AO_ATTEN = 0.5;

float ao(in vec3 p, in vec3 norm) {
	float sum = 0.0;
	float atten = 1.0;
	float s = AO_PRECISION;
    
    return 1.;
    
    float hitDist = 0.0;
    
    
	for (int i = 0; i < AO_ITERATIONS; i++) {
        hitDist = df(p+norm*s);
		sum += (s-hitDist)*atten;
		s += AO_PRECISION;

		atten *= AO_ATTEN;
                     //  if ( i == 1) return atten;

	}
    
    return sum;
	
	return 1.0-max(0.0,sum*AO_INTENSITY);
}


               
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy - vec2(.5);
    uv.y *= iResolution.y/iResolution.x;

    float cst = iTime*CAMERA_SPEED;
    vec3 rp = vec3(5.,15.,4.);
    vec3 rd = normalize(vec3(uv.x,-1.,uv.y));
    
    float d,s = 0.;
    for (int i = 0; i < ITER; i++) {
        d = df(rp);
        if (d < EPSILON || s > 30.) break;
        
        rp += d*rd;
    	s += d;
    }

	if (d > EPSILON) {
       fragColor = vec4(0.);
    	return;
	}
    
    
    vec3 c = texture(iChannel0,rp.xz*0.1).xyz;
    #ifdef NO_SHADING
    fragColor = vec4(c,1.);
    return;
    #endif
    
    vec3 lightPos = vec3(iMouse.x/iResolution.x*10.,10.,iMouse.y/iResolution.y*10.);
    vec3 n = normal(rp);
    vec3 ld = -normalize(rp-lightPos);
    vec3 hp = rp;
    
    float sh = 1.;
    #ifdef SHADOW
    sh = s = 0.;
    rp += n*EPSILON;//ld*EPSILON*2.;
    for (int i = 0; i < ITER; i++) {
        d = df(rp);
        if (d < EPSILON || s > 30.) break;
        
       sh += max(0.,min(s/abs(ld.y)*.5,.1)-d);
        rp += ld*d;
        s += d;
    }
    if (d < EPSILON) {
        sh = 0.;
    } else {
        sh = max(0.,1.-sh);
    }
    #endif
    
    c = ao(hp,n)*ambient*c+
        	max(0.,dot(ld,n))*sh*diffuse*c;
    
    fragColor = vec4(c,1.);
}