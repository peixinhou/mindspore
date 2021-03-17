#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void Abs(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x >= 0 ? result.x : -result.x;
  result.y = result.y >= 0 ? result.y : -result.y;
  result.z = result.z >= 0 ? result.z : -result.z;
  result.w = result.w >= 0 ? result.w : -result.w;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Cos(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = cos(result.x);
  result.y = cos(result.y);
  result.z = cos(result.z);
  result.w = cos(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Sin(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = sin(result.x);
  result.y = sin(result.y);
  result.z = sin(result.z);
  result.w = sin(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Neg(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = -result.x;
  result.y = -result.y;
  result.z = -result.z;
  result.w = -result.w;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Exp(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = exp(result.x);
  result.y = exp(result.y);
  result.z = exp(result.z);
  result.w = exp(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Log(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x > 0 ? log(result.x) : HUGE_VALF;
  result.y = result.y > 0 ? log(result.y) : HUGE_VALF;
  result.z = result.z > 0 ? log(result.z) : HUGE_VALF;
  result.w = result.w > 0 ? log(result.w) : HUGE_VALF;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Square(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x * result.x;
  result.y = result.y * result.y;
  result.z = result.z * result.z;
  result.w = result.w * result.w;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Sqrt(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x > 0 ? sqrt(result.x) : HUGE_VALF;
  result.y = result.y > 0 ? sqrt(result.y) : HUGE_VALF;
  result.z = result.z > 0 ? sqrt(result.z) : HUGE_VALF;
  result.w = result.w > 0 ? sqrt(result.w) : HUGE_VALF;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Rsqrt(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x > 0 ? 1.0f / sqrt(result.x) : HUGE_VALF;
  result.y = result.y > 0 ? 1.0f / sqrt(result.y) : HUGE_VALF;
  result.z = result.z > 0 ? 1.0f / sqrt(result.z) : HUGE_VALF;
  result.w = result.w > 0 ? 1.0f / sqrt(result.w) : HUGE_VALF;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void LogicalNot(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = result.x > 0 || result.x < 0 ? false : true;
  result.y = result.y > 0 || result.y < 0 ? false : true;
  result.z = result.z > 0 || result.z < 0 ? false : true;
  result.w = result.w > 0 || result.w < 0 ? false : true;
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Floor(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = floor(result.x);
  result.y = floor(result.y);
  result.z = floor(result.z);
  result.w = floor(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Ceil(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = ceil(result.x);
  result.y = ceil(result.y);
  result.z = ceil(result.z);
  result.w = ceil(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Round(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  result.x = round(result.x);
  result.y = round(result.y);
  result.z = round(result.z);
  result.w = round(result.w);
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), result);
}

__kernel void Erf(__read_only image2d_t input0, __write_only image2d_t output, int4 output_shape) {
  int X = get_global_id(0);  // N*H
  int Y = get_global_id(1);  // W
  int Z = get_global_id(2);  // c/4
  if (X >= output_shape.x * output_shape.y || Y >= output_shape.z || Z >= output_shape.w) {
    return;
  }
  FLT4 result = READ_IMAGE(input0, smp_none, (int2)((Y)*output_shape.w + Z, (X)));
  WRITE_IMAGE(output, (int2)((Y)*output_shape.w + Z, (X)), erf(result));
}
