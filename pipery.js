const glsl = x => x;
const glslCanvas = document.getElementById('glslCanvas');
glslCanvas.onclick = () => {
  console.log('clicked');
};
const gl = glslCanvas.getContext('webgl2');
const W = 256;
const H = 256;
const data = new Float32Array(W * H * 4); // RGBA per pixel
for (let i = 0; i < data.length; i += 4) {
    data[i + 0] = Math.random(); // R
    data[i + 1] = Math.random(); // G
    data[i + 2] = Math.random(); // B
    data[i + 3] = 1.0;           // A
}
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, data);
(function(source) {
  const toy = new ShaderToyLite('glslCanvas');
  toy.addTexture(texture, 'game');
  toy.setCommon("");
  toy.setImage({source, iChannel0: 'game'});
  toy.play();
})(glsl`

// Based on https://www.shadertoy.com/view/Xds3zN by Inigo Quilez.
#define AA 1

//------------------------------------------------------------------
float dot2(in vec2 v) {
  return dot(v, v);
}
float dot2(in vec3 v) {
  return dot(v, v);
}
float ndot(in vec2 a, in vec2 b) {
  return a.x * b.x - a.y * b.y;
}

float sdPlane(vec3 p) {
  return p.y;
}

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

float sdBox(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

// vertical
float sdCylinder(vec3 p, vec2 h) {
  vec2 d = abs(vec2(length(p.xz), p.y)) - h;
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

// arbitrary orientation
float sdCylinder(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a;
  vec3 ba = b - a;
  float baba = dot(ba, ba);
  float paba = dot(pa, ba);

  float x = length(pa * baba - ba * paba) - r * baba;
  float y = abs(paba - baba * 0.5) - baba * 0.5;
  float x2 = x * x;
  float y2 = y * y * baba;
  float d = (max(x, y) < 0.0) ? -min(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
  return sign(d) * sqrt(abs(d)) / baba;
}

// c is the sin/cos of the desired cone angle
float sdSolidAngle(vec3 pos, vec2 c, float ra) {
  vec2 p = vec2(length(pos.xz), pos.y);
  float l = length(p) - ra;
  float m = length(p - c * clamp(dot(p, c), 0.0, ra));
  return max(l, m * sign(c.y * p.x - c.x * p.y));
}

float sdHorseshoe(in vec3 p, in vec2 c, in float r, in float le, vec2 w) {
  p.x = abs(p.x);
  float l = length(p.xy);
  p.xy = mat2(-c.x, c.y, c.y, c.x) * p.xy;
  p.xy = vec2((p.y > 0.0 || p.x > 0.0) ? p.x : l * sign(-c.x), (p.x > 0.0) ? p.y : l);
  p.xy = vec2(p.x, abs(p.y - r)) - vec2(le, 0.0);

  vec2 q = vec2(length(max(p.xy, 0.0)) + min(0.0, max(p.x, p.y)), p.z);
  vec2 d = abs(q) - w;
  return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdU(in vec3 p, in float r, in float le, vec2 w) {
  p.x = (p.y > 0.0) ? abs(p.x) : length(p.xy);
  p.x = abs(p.x - r);
  p.y = p.y - le;
  float k = max(p.x, p.y);
  vec2 q = vec2((k < 0.0) ? -k : length(max(p.xy, 0.0)), abs(p.z)) - w;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

//------------------------------------------------------------------

vec2 opU(vec2 d1, vec2 d2) {
  return (d1.x < d2.x) ? d1 : d2;
}

//------------------------------------------------------------------

#define ZERO (min(iFrame,0))



// From https://www.shadertoy.com/view/WtSfWK by Iñigo Quílez.
int mod3( int n ) {
    return (n<0) ? 2-((2-n)%3) : n%3;
}
ivec2 hexagonID( vec2 p ) {
  const float k3 = 1.732050807;
  vec2 q = vec2( p.x, p.y*k3*0.5 + p.x*0.5 );

  ivec2 pi = ivec2(floor(q));
  vec2  pf =       fract(q);

  int v = mod3(pi.x+pi.y);

  int   ca = (v<1)?0:1;
  int   cb = (v<2)?0:1;
  ivec2 ma = (pf.x>pf.y)?ivec2(0,1):ivec2(1,0);

  ivec2 id = pi + ca - cb*ma;

  return ivec2( id.x, id.y - (id.x+id.y)/3 );
}
vec2 hexagonCenFromID( in ivec2 id ) {
    const float k3 = 1.732050807;
    return vec2(float(id.x),float(id.y)*k3);
}

// Repeat space along a hexagonal grid of cell size 's'
vec2 opRepHex(vec2 p, float s) {
    vec2 a = s * vec2(1.0, 0.0);
    vec2 b = s * vec2(0.5, 0.8660254);
    mat2 m  = mat2(a, b);          // columns = basis
    mat2 inv = inverse(m);         // change of basis
    vec2 uv = inv * p;             // into hex space
    uv = fract(uv) - 0.5;          // keep within [-0.5, 0.5]
    return m * uv;                 // back to cartesian
}

float sdTorus( vec3 p, vec2 t ) {
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

vec3 rot30(in vec3 p) {
  float c = 0.8660254;
  float s = 0.5;
  mat2 r = mat2(c, -s, s, c);
  p.xz = r * p.xz;
  return p;
}

vec2 map(in vec3 pos) {
  vec2 res = vec2(pos.y, 0.0);
  ivec2 cell = hexagonID(pos.xz*5.);
  vec2 center = hexagonCenFromID(cell) * 0.2;
  float c = texture(iChannel0, vec2(cell)*0.01).r;
  pos -= vec3(center.x, 0.0, center.y);
  res = opU(res, vec2(sdCapsule(rot30(pos), vec3(0., 0.04, -0.5), vec3(0., 0.04, 0.5), 0.04), 3.5));
  res = opU(res, vec2(sdTorus(pos+vec3(0.,0.,0.), vec2(0.2-c*.1, 0.04)), 4.5+c*4.));
  // res = opU(res, vec2(sdSphere(pos+vec3(0.,0.3,0.), .5), 5.5+float(cell.x+100)));
  return res;
}

// https://iquilezles.org/articles/boxfunctions
vec2 iBox(in vec3 ro, in vec3 rd, in vec3 rad) {
  vec3 m = 1.0 / rd;
  vec3 n = m * ro;
  vec3 k = abs(m) * rad;
  vec3 t1 = -n - k;
  vec3 t2 = -n + k;
  return vec2(max(max(t1.x, t1.y), t1.z), min(min(t2.x, t2.y), t2.z));
}

vec2 raycast(in vec3 ro, in vec3 rd) {
  vec2 res = vec2(-1.0, -1.0);

  float tmin = 1.0;
  float tmax = 20.0;

    // raytrace floor plane
  float tp1 = (0.0 - ro.y) / rd.y;
  if(tp1 > 0.0) {
    tmax = min(tmax, tp1);
    res = vec2(tp1, 1.0);
  }
    //else return res;

    // raymarch primitives
  vec2 tb = iBox(ro - vec3(0.0, 0.0, 0.0), rd, vec3(10., 0.5, 10.0));
  if(tb.x < tb.y && tb.y > 0.0 && tb.x < tmax) {
        //return vec2(tb.x,2.0);
    tmin = max(tb.x, tmin);
    tmax = min(tb.y, tmax);

    float t = tmin;
    for(int i = 0; i < 70 && t < tmax; i++) {
      vec2 h = map(ro + rd * t);
      if(abs(h.x) < (0.0001 * t)) {
        res = vec2(t, h.y);
        break;
      }
      t += h.x;
    }
  }

  return res;
}

// https://iquilezles.org/articles/rmshadows
float calcSoftshadow(in vec3 ro, in vec3 rd, in float mint, in float tmax) {
    // bounding volume
  float tp = (0.8 - ro.y) / rd.y;
  if(tp > 0.0)
    tmax = min(tmax, tp);

  float res = 1.0;
  float t = mint;
  for(int i = ZERO; i < 24; i++) {
    float h = map(ro + rd * t).x;
    float s = clamp(8.0 * h / t, 0.0, 1.0);
    res = min(res, s);
    t += clamp(h, 0.01, 0.2);
    if(res < 0.004 || t > tmax)
      break;
  }
  res = clamp(res, 0.0, 1.0);
  return res * res * (3.0 - 2.0 * res);
}

// https://iquilezles.org/articles/normalsSDF
vec3 calcNormal(in vec3 pos) {
#if 0
    // do NOT call map() many times inside calcNormal()
  vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
  return normalize(e.xyy * map(pos + e.xyy).x +
    e.yyx * map(pos + e.yyx).x +
    e.yxy * map(pos + e.yxy).x +
    e.xxx * map(pos + e.xxx).x);
#else
    // instead put it only once and in a loop to prevent
    // code expansion - inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
  vec3 n = vec3(0.0);
  for(int i = ZERO; i < 4; i++) {
    vec3 e = 0.5773 * (2.0 * vec3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
    n += e * map(pos + 0.0005 * e).x;
      //if( n.x+n.y+n.z>100.0 ) break;
  }
  return normalize(n);
#endif
}

// https://iquilezles.org/articles/nvscene2008/rwwtt.pdf
float calcAO(in vec3 pos, in vec3 nor) {
  float occ = 0.0;
  float sca = 1.0;
  for(int i = ZERO; i < 5; i++) {
    float h = 0.01 + 0.12 * float(i) / 4.0;
    float d = map(pos + h * nor).x;
    occ += (h - d) * sca;
    sca *= 0.95;
    if(occ > 0.35)
      break;
  }
  return clamp(1.0 - 3.0 * occ, 0.0, 1.0) * (0.5 + 0.5 * nor.y);
}

// https://iquilezles.org/articles/checkerfiltering
float checkersGradBox(in vec2 p, in vec2 dpdx, in vec2 dpdy) {
    // filter kernel
  vec2 w = abs(dpdx) + abs(dpdy) + 0.001;
    // analytical integral (box filter)
  vec2 i = 2.0 * (abs(fract((p - 0.5 * w) * 0.5) - 0.5) - abs(fract((p + 0.5 * w) * 0.5) - 0.5)) / w;
    // xor pattern
  return 0.5 - 0.5 * i.x * i.y;
}

vec3 render(in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy) {
    // background
  vec3 col = vec3(0.7, 0.7, 0.9) - max(rd.y, 0.0) * 0.3;

    // raycast scene
  vec2 res = raycast(ro, rd);
  float t = res.x;
  float m = res.y;
  if(m > -0.5) {
    vec3 pos = ro + t * rd;
    vec3 nor = (m < 1.5) ? vec3(0.0, 1.0, 0.0) : calcNormal(pos);
    vec3 ref = reflect(rd, nor);

        // material
    col = 0.2 + 0.2 * sin(m * 2.0 + vec3(0.0, 1.0, 2.0));
    float ks = 1.0;

    if(m < 1.5) {
            // project pixel footprint into the plane
      vec3 dpdx = ro.y * (rd / rd.y - rdx / rdx.y);
      vec3 dpdy = ro.y * (rd / rd.y - rdy / rdy.y);

      float f = checkersGradBox(3.0 * pos.xz, 3.0 * dpdx.xz, 3.0 * dpdy.xz);
      col = 0.15 + f * vec3(0.05);
      ks = 0.4;
    }

        // lighting
    float occ = calcAO(pos, nor);

    vec3 lin = vec3(0.0);

        // sun
    {
      vec3 lig = normalize(vec3(-0.5, 0.4, -0.6));
      vec3 hal = normalize(lig - rd);
      float dif = clamp(dot(nor, lig), 0.0, 1.0);
          //if( dif>0.0001 )
      dif *= calcSoftshadow(pos, lig, 0.02, 2.5);
      float spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 16.0);
      spe *= dif;
      spe *= 0.04 + 0.96 * pow(clamp(1.0 - dot(hal, lig), 0.0, 1.0), 5.0);
                //spe *= 0.04+0.96*pow(clamp(1.0-sqrt(0.5*(1.0-dot(rd,lig))),0.0,1.0),5.0);
      lin += col * 2.20 * dif * vec3(1.30, 1.00, 0.70);
      lin += 5.00 * spe * vec3(1.30, 1.00, 0.70) * ks;
    }
        // sky
    {
      float dif = sqrt(clamp(0.5 + 0.5 * nor.y, 0.0, 1.0));
      dif *= occ;
      float spe = smoothstep(-0.2, 0.2, ref.y);
      spe *= dif;
      spe *= 0.04 + 0.96 * pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 5.0);
          //if( spe>0.001 )
      spe *= calcSoftshadow(pos, ref, 0.02, 2.5);
      lin += col * 0.60 * dif * vec3(0.40, 0.60, 1.15);
      lin += 2.00 * spe * vec3(0.40, 0.60, 1.30) * ks;
    }
        // back
    {
      float dif = clamp(dot(nor, normalize(vec3(0.5, 0.0, 0.6))), 0.0, 1.0) * clamp(1.0 - pos.y, 0.0, 1.0);
      dif *= occ;
      lin += col * 0.55 * dif * vec3(0.25, 0.25, 0.25);
    }
        // sss
    {
      float dif = pow(clamp(1.0 + dot(nor, rd), 0.0, 1.0), 2.0);
      dif *= occ;
      lin += col * 0.25 * dif * vec3(1.00, 1.00, 1.00);
    }

    col = lin;

    col = mix(col, vec3(0.7, 0.7, 0.9), 1.0 - exp(-0.0001 * t * t * t));
  }

  return vec3(clamp(col, 0.0, 1.0));
}

mat3 setCamera(in vec3 ro, in vec3 ta, float cr) {
  vec3 cw = normalize(ta - ro);
  vec3 cp = vec3(sin(cr), cos(cr), 0.0);
  vec3 cu = normalize(cross(cw, cp));
  vec3 cv = (cross(cu, cw));
  return mat3(cu, cv, cw);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
  vec2 mo = iMouse.xy / iResolution.xy;
  float time = 32.0;// + iTime * 1.5;

    // camera
  vec3 ta = vec3(0., 0., 0.);
  vec3 ro = ta + vec3(4.5 * cos(0.1 * time + 7.0 * mo.x), 5.2, 4.5 * sin(0.1 * time + 7.0 * mo.x));
    // camera-to-world transformation
  mat3 ca = setCamera(ro, ta, 0.0);

  vec3 tot = vec3(0.0);
#if AA>1
  for(int m = ZERO; m < AA; m++) for(int n = ZERO; n < AA; n++) {
        // pixel coordinates
      vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
      vec2 p = (2.0 * (fragCoord + o) - iResolution.xy) / iResolution.y;
#else
      vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
#endif

        // focal length
      const float fl = 2.5;

        // ray direction
      vec3 rd = ca * normalize(vec3(p, fl));

         // ray differentials
      vec2 px = (2.0 * (fragCoord + vec2(1.0, 0.0)) - iResolution.xy) / iResolution.y;
      vec2 py = (2.0 * (fragCoord + vec2(0.0, 1.0)) - iResolution.xy) / iResolution.y;
      vec3 rdx = ca * normalize(vec3(px, fl));
      vec3 rdy = ca * normalize(vec3(py, fl));

        // render
      vec3 col = render(ro, rd, rdx, rdy);

        // gain
        // col = col*3.0/(2.5+col);

		// gamma
      col = pow(col, vec3(0.4545));

      tot += col;
#if AA>1
    }
  tot /= float(AA * AA);
#endif

  fragColor = vec4(tot, 1.0);
}

`);
