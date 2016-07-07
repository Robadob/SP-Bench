#version 430
in vec3 u_normal;
flat in float count;

out vec4 outColor;
//HSV interpolation is more attractive than rgb interpolation
//http://www.cs.rit.edu/~ncs/color/t_convert.html
//hsv(0-360,0-1,0-1)
vec3 hsv2rgb(in vec3 hsv)
{
  if(hsv.g==0)//Grey
    return vec3(hsv.b);

  float h = hsv.r/60;
  int i = int(floor(h));
  float f = h-i;
  float p = hsv.b * (1-hsv.g);
  float q = hsv.b * (1-hsv.g * f);
  float t = hsv.b * (1-hsv.g * (1-f));
  switch(i)
  {
    case 0:
      return vec3(hsv.b,t,p);
    case 1:
      return vec3(q,hsv.b,p);
    case 2:
      return vec3(p,hsv.b,p);
    case 3:
      return vec3(p,q,hsv.b);
    case 4:
      return vec3(t,p,hsv.b);
    default: //case 5
      return vec3(hsv.b,p,q);
  }

}
float normalDist(in float x, in float mean, in float variance)
{
#define M_PI 3.1415926535897932384626433832795
    float scale = sqrt(variance);
    return (1.0f/(scale*sqrt(2.0f*M_PI))) * exp(-pow(x-mean,2.0f)/(2.0f*variance));
} 
void main()
{
//0 is red, ~120 is green
  //int icount = int(count);
  //float minCount= max(0,(0.5-((count<=0.001)?0.5:count))*120);//120-min(120, count);

vec3 pastelPurple = hsv2rgb(vec3(normalDist(count, -0.49f,pow(0.42f,2.0f))*240,1.0f,1.0f));//Normal dist returns a value 0->0.5 and 1->0
//Flat shading
  vec3 N  = normalize(cross(dFdx(u_normal), dFdy(u_normal)));//Face Normal
  vec3 L = normalize(vec3(0,0,0)-u_normal);
  vec3 diffuse = pastelPurple * max(dot(L, N), 0.0);

  outColor = vec4(diffuse,1.0);//vec4(diffuse.xyz,1.0);
 // gl_FragColor = vec4(1.0,0.0,0.0,1.0);//vec4(diffuse.xyz,1.0);
}