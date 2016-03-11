#version 430
in vec3 u_normal;
out vec4 outColor;

void main()
{
  vec3 pastelPurple = vec3(0.60, 0.47, 0.71);
//Flat shading
  vec3 N  = normalize(cross(dFdx(u_normal), dFdy(u_normal)));//Face Normal
  vec3 L = normalize(vec3(0,0,0)-u_normal);
  vec3 diffuse = pastelPurple * max(dot(L, N), 0.0);

  outColor = vec4(diffuse,1.0);//vec4(diffuse.xyz,1.0);
 // gl_FragColor = vec4(1.0,0.0,0.0,1.0);//vec4(diffuse.xyz,1.0);
}