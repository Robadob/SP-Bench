#version 430
#extension GL_EXT_gpu_shader4 : require 
//Use gpu_shader4 extension because I can't get the texture or texture1D functions to work as required.

//gl_InstanceID //attribute in uint agent_index;

layout(location = 0) in vec3 in_position;
out vec3 u_normal;

layout(location = 0) uniform mat4 modelview_matrix;
layout(location = 1) uniform mat4 projection_matrix;

layout(binding=2) uniform samplerBuffer tex_locX;
layout(binding=3) uniform samplerBuffer tex_locY;
layout(binding=4) uniform samplerBuffer tex_locZ;

void main(){
  //Grab model offset from textures
  vec3 loc_data = vec3(
                       texelFetchBuffer(tex_locX, gl_InstanceID).x,
                       texelFetchBuffer(tex_locY, gl_InstanceID).x,
                       texelFetchBuffer(tex_locZ, gl_InstanceID).x
                     );
  //Output vert loc to be interpolated for shader to calc norm
  vec3 t_position = in_position + loc_data;
  u_normal = vec4(modelview_matrix * vec4(t_position,1.0)).xyz;
  gl_Position = projection_matrix * modelview_matrix * vec4(t_position,1.0);

}