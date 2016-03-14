#version 430
in vec3 u_normal;
flat in float count;

out vec4 outColor;
//HSV interpolation is more attractive than rgb interpolation
//http://www.cs.rit.edu/~ncs/color/t_convert.html
//hsv(0-360,0-1,0-1)
vec3 hsv2rgb(vec3 hsv)
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
void main()
{
//0 is red, ~120 is green
  float minCount= (3-count)*120;//120-min(120, count);

  vec3 pastelPurple = hsv2rgb(vec3(minCount, 1.0, 1.0));
//Flat shading
  vec3 N  = normalize(cross(dFdx(u_normal), dFdy(u_normal)));//Face Normal
  vec3 L = normalize(vec3(0,0,0)-u_normal);
  vec3 diffuse = pastelPurple * max(dot(L, N), 0.0);

  outColor = vec4(diffuse,1.0);//vec4(diffuse.xyz,1.0);
 // gl_FragColor = vec4(1.0,0.0,0.0,1.0);//vec4(diffuse.xyz,1.0);
}