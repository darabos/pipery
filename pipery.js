const glsl = (strings, ...values) =>
  values.reduce((acc, v, i) => acc + `${v}` + strings[i + 1], strings[0]);
let currentCanvasUpdate = 0;
const glslCanvas = document.getElementById('glslCanvas');
function onResize() {
  resolutionSlider.max = window.devicePixelRatio || 3;
  const dpr = resolutionSlider.valueAsNumber;
  const rect = glslCanvas.getBoundingClientRect();
  glslCanvas.width = rect.width * dpr;
  glslCanvas.height = rect.height * dpr;
}
onResize();
window.addEventListener('resize', onResize);
resolutionSlider.oninput = (e) => {
  onResize();
}
glslCanvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  cameraPos[0] += e.deltaY*0.01;
  cameraPos[2] -= e.deltaX*0.01;
});
glslCanvas.addEventListener('contextmenu', function(e) {
    e.preventDefault();
    return false;
}, false);
glslCanvas.onmousedown = (e) => {
  const x = (2 * e.offsetX - glslCanvas.clientWidth) / glslCanvas.clientHeight;
  const y = 1 - 2 * e.offsetY / glslCanvas.clientHeight;
  const [cx, cy] = pixelToCell(x, y);
  const key = `${cy-1}_${cx}`;
  const cell = game.board.get(key);
  if (!cell) {
    return;
  }
  if (e.button === 0) {
    if (e.shiftKey) {
      cell.pipesRotateAnticlockwise();
    } else {
      cell.pipesRotateClockwise();
    }
  } else if (e.button === 2) {
    cell.locked = !cell.locked;
  }
};
function pixelToCell(x, y) {
  const fl = 3.5;
  let p = [x, y, fl];
  p = p.map(v => v / Math.hypot(...p));
  const rd = [-0.707*p[1] - 0.707 * p[2], 0.707 * p[1] - 0.707 * p[2], -p[0]];
  const ro = [cameraPos[0] + 5., cameraPos[1] + 5., cameraPos[2] + 0.];
  const len = (ro[1]-0.2) / rd[1];
  const ix = ro[0] - rd[0] * len + R/3;
  const iy = ro[2] - rd[2] * len + R/3;
  function mod3(n) {
      return (n<0) ? 2-((2-n)%3) : n%3;
  }
  function hexagonID(x, y) {
    const k3 = 1.732050807;
    const q = [x, y*k3*0.5 + x*0.5];
    const pi = q.map(v => Math.floor(v));
    const pf = q.map(v => v - Math.floor(v));
    const v = mod3(pi[0]+pi[1]);
    const ca = (v<1)?0:1;
    const cb = (v<2)?0:1;
    const ma = (pf[0]>pf[1])?[0,1]:[1,0];
    const id = [pi[0] + ca - cb*ma[0], pi[1] + ca - cb*ma[1]];
    return [ id[0], id[1] - Math.floor((id[0]+id[1])/3) ];
  }
  return hexagonID(ix*5., iy*5.);
}
let camV = [0, 0, 0];
let cameraPos = [0, 0, 0];
function onDraw() {
  const CAM_SPEED = 0.01;
  camV[0] *= 0.9;
  camV[1] *= 0.9;
  camV[2] *= 0.9;
  if (keysDown['ArrowUp'] || keysDown['w']) {
    camV[0] -= CAM_SPEED;
  }
  if (keysDown['ArrowDown'] || keysDown['s']) {
    camV[0] += CAM_SPEED;
  }
  if (keysDown['ArrowRight'] || keysDown['d']) {
    camV[2] -= CAM_SPEED;
  }
  if (keysDown['ArrowLeft'] || keysDown['a']) {
    camV[2] += CAM_SPEED;
  }
  cameraPos[0] += camV[0];
  cameraPos[1] += camV[1];
  cameraPos[2] += camV[2];
  const program = gl.getParameter(gl.CURRENT_PROGRAM);
  if (program) {
    const loc = gl.getUniformLocation(program, "cameraPos");
    gl.uniform3fv(loc, cameraPos);
  }
  updateCells();
}
const keysDown = {};
window.onkeydown = (e) => {
  keysDown[e.key] = true;
};
window.onkeyup = (e) => {
  keysDown[e.key] = false;
};

const pixel = new Float32Array([0, 0, 0, 0]);
function updateCells() {
  boardToTextureData();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, W, H, gl.RGBA, gl.FLOAT, data);
}
function polyToPixel(poly) {
  const tnr = pipesToTypesAndRotations[poly.pipes];
  pixel[0] = Math.PI / 3 * (tnr.rotation - poly.pipesRotationDisplay);
  pixel[1] = tnr.type / TYPE_MAX;
  pixel[2] = (poly.hasLight ? poly.hasCycle ? 4.7 : 5.5 : 1.5) / 10;
  pixel[3] = (poly.locked ? 3.0 : 3.5) / 10;
}
const blockTypes = {
  0b000000: 0,
  0b000001: 1,
  0b000011: 2,
  0b000101: 3,
  0b000111: 4,
  0b001001: 5,
  0b001011: 6,
  0b001101: 7,
  0b001111: 8,
  0b010101: 9,
  0b010111: 10,
  0b011011: 11,
  0b011111: 12,
  0b111111: 13,
};
const pipesToTypesAndRotations = {};
for (let s = 0; s < 64; s++) {
  pipesToTypesAndRotations[s] = normalizePipes(s);
}
function normalizePipes(s) {
  // try all rotations to find a known type
  for (let rot = 0; rot < 6; rot++) {
    const rs =
      ((s >> rot) | (s << (6 - rot))) & 0b111111;
    if (blockTypes[rs] !== undefined) {
      return {
        type: blockTypes[rs],
        rotation: (6 - rot) % 6,
      };
      break;
    }
  }
}
const gl = glslCanvas.getContext('webgl2');
const R = 15;
const W = R * 4;
const H = R * 4;
const TYPE_MAX = 20;
const data = new Float32Array(W * H * 4); // RGBA per pixel
game.generateGame("hexagon", R, R);
boardToTextureData();
function boardToTextureData() {
	for (const [key, poly] of game.board) {
    const [x, y] = key.split('_').map(Number);
    const i = (y * W + x + 1 + R * W) * 4;
    poly.updatePipesRotationDisplay();
    polyToPixel(poly);
    data[i + 0] = pixel[0];
    data[i + 1] = pixel[1];
    data[i + 2] = pixel[2];
    data[i + 3] = pixel[3];
  }
}
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, data);
(function(source) {
  const toy = new ShaderToyLite('glslCanvas');
  toy.addTexture(texture, 'game');
  toy.setCommon("");
  toy.setImage({source, iChannel0: 'game'});
  toy.setOnDraw(onDraw);
  toy.play();
})(glsl`  // Based on https://www.shadertoy.com/view/Xds3zN by Inigo Quilez.

uniform vec3 cameraPos;

#define AA 1

float sdSphere(vec3 p, float s) {
  return length(p) - s;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

float sdCylinder(vec3 p, float w) {
  return length(p.yz) - w;
}

float sdTorus( vec3 p, vec2 t ) {
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

vec2 opU(vec2 d1, vec2 d2) {
  return (d1.x < d2.x) ? d1 : d2;
}

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

vec3 rot(in float angle, in vec3 p) {
  float c = cos(angle);
  float s = sin(angle);
  mat2 r = mat2(c, -s, s, c);
  p.xz = r * p.xz;
  return p;
}

vec2 map(in vec3 pos) {
  vec2 res = vec2(pos.y, 0.0);
  pos.xz += float(${R/3});
  pos.y -= 0.1;
  ivec2 cell = hexagonID(pos.xz*5.);
  vec2 center = hexagonCenFromID(cell) * 0.2;
  vec2 uv = vec2(cell).yx;
  uv.y += ${R}.;
  vec4 data = texture(iChannel0, uv / ${W}.);
  float ty = data.g * ${TYPE_MAX}.;
  float r = data.r;
  float color = data.b * 10.;
  pos -= vec3(center.x, 0.0, center.y);
  if (ty < 0.5) { // 000000
    // Off the board. We should look at two neighbor cells and take the lowest distance.
    // But for now we just underestimate the distance instead, to reduce the glitches at the edges.
    return res * 0.5;
  }
  vec2 seg = vec2(sdCapsule(rot(r, pos), vec3(0., .05, 0.), vec3(0.4, .05, 0.0), 0.04), color);
  vec2 seg60 = vec2(sdCapsule(rot(r-1.05, pos), vec3(0., .05, 0.), vec3(0.4, .05, 0.0), 0.04), color);
  vec2 seg120 = vec2(sdCapsule(rot(r-2.1, pos), vec3(0., .05, 0.), vec3(0.4, .05, 0.0), 0.04), color);
  vec2 seg240 = vec2(sdCapsule(rot(r-4.2, pos), vec3(0., .05, 0.), vec3(0.4, .05, 0.0), 0.04), color);
  vec2 straight = vec2(sdCylinder(rot(r, pos+vec3(0.,-.05,0.)), 0.04), color);
  vec2 straight60 = vec2(sdCylinder(rot(r-1.05, pos+vec3(0.,-.05,0.)), 0.04), color);
  vec2 straight120 = vec2(sdCylinder(rot(r-2.1, pos+vec3(0.,-.05,0.)), 0.04), color);
  vec2 bigbend = vec2(sdTorus(rot(r+2.1, pos)+vec3(0.4,-.05,0.), vec2(0.3464, 0.04)), color);
  vec2 bigbend60 = vec2(sdTorus(rot(r+1.05, pos)+vec3(0.4,-.05,0.), vec2(0.3464, 0.04)), color);
  vec2 smallbend = vec2(sdTorus(rot(r+2.1, pos)+vec3(0.2,-.05,0.1155), vec2(0.1155, 0.04)), color);
  if (ty < 1.5) { // 000001
    res = opU(res, seg);
  } else if (ty < 2.5) { // 000011
    res = opU(res, smallbend);
  } else if (ty < 3.5) { // 000101
    res = opU(res, bigbend);
  } else if (ty < 4.5) { // 000111
    res = opU(res, bigbend);
    res = opU(res, smallbend);
  } else if (ty < 5.5) { // 001001
    res = opU(res, straight);
  } else if (ty < 6.5) { // 001011
    res = opU(res, straight);
    res = opU(res, bigbend60);
  } else if (ty < 7.5) { // 001101
    res = opU(res, straight);
    res = opU(res, bigbend);
  } else if (ty < 8.5) { // 001111
    res = opU(res, bigbend);
    res = opU(res, bigbend60);
  } else if (ty < 9.5) { // 010101
    res = opU(res, seg);
    res = opU(res, seg120);
    res = opU(res, seg240);
  } else if (ty < 10.5) { // 010111
    res = opU(res, straight60);
    res = opU(res, bigbend);
  } else if (ty < 11.5) { // 011011
    res = opU(res, straight);
    res = opU(res, straight60);
  } else if (ty < 12.5) { // 011111
    res = opU(res, straight);
    res = opU(res, straight60);
    res = opU(res, seg120);
  } else { // 111111
    res = opU(res, straight);
    res = opU(res, straight60);
    res = opU(res, straight120);
  }
  res = opU(res, vec2(sdSphere(pos+vec3(0.,0.45,0.), .5), data.a * 10.));
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
    for(int i = 0; i < 20 && t < tmax; i++) {
      vec2 h = map(ro + rd * t);
      if(abs(h.x) < (0.001 * t)) {
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
  for(int i = ZERO; i < 12; i++) {
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
      float spe = pow(clamp(dot(nor, hal), 0.0, 1.0), 36.0);
      spe *= dif;
      spe *= 0.14 + 0.96 * pow(clamp(1.0 - dot(hal, lig), 0.0, 1.0), 5.0);
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
      // spe *= calcSoftshadow(pos, ref, 0.02, 2.5);
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
    // camera
  vec3 ta = cameraPos;
  vec3 ro = ta + vec3(5., 5., 0.);
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
      const float fl = 3.5;

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
