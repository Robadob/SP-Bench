#version 430

//gl_InstanceID //attribute in uint agent_index;

layout(location = 0) in vec3 _vertex;
out vec3 u_normal;
out flat float count; //Count as provided by texture, to use in frag shader for colouring

layout(location = 0) uniform mat4 _modelViewMat;
layout(location = 1) uniform mat4 _projectionMat;

uniform samplerBuffer tex_locX;
uniform samplerBuffer tex_locY;
uniform samplerBuffer tex_count;

void main(){
  //Grab model offset from textures
  vec3 loc_data = vec3(
                       texelFetch(tex_locX, gl_InstanceID).x,
                       texelFetch(tex_locY, gl_InstanceID).x,
                       0
                     );
  //Output vert loc to be interpolated for shader to calc norm
  vec3 t_position = _vertex + loc_data;
  u_normal = vec4(_modelViewMat * vec4(t_position,1.0)).xyz;
  gl_Position = _projectionMat * _modelViewMat * vec4(t_position,1.0);
  count = texelFetch(tex_count, gl_InstanceID).x;
}