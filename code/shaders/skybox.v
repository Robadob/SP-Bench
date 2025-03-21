#version 430

in vec3 _vertex;
//layout(location = 1) in vec3 in_normal;
out vec3 tex_coords;

uniform mat4 _modelViewProjectionMat;


void main () {
  tex_coords = _vertex;
  gl_Position = _modelViewProjectionMat * vec4(_vertex, 1.0);
}