"""GLSL shader engine: Shadertoy-style single and multibuffer shaders."""

import glob
import os
import re
import time

import cv2
import moderngl
import numpy as np


# Vertex source for the shader pass: a full-screen quad passing pixel
# coordinates through. gl_FragCoord in the fragment shader is Shadertoy's
# fragCoord (pixels, origin bottom-left).
SHADER_VERTEX_SOURCE = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# The user file provides mainImage(out vec4 fragColor, in vec2 fragCoord),
# Shadertoy style; this wrapper supplies the entry point, the input texture
# (texture0, holding the BGR frame), and the iResolution/iTime uniforms. The
# frame data and the rendered result are both BGR, so shaders that preserve
# channel order (all the brightness/pattern effects this is for) need no
# reordering.
SHADER_FRAGMENT_TEMPLATE = """
#version 330
uniform sampler2D texture0;
uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform vec2 iResolution;
uniform float iTime;
uniform vec4 iMouse;
#define iChannel0 texture0
#define iChannel1 texture1
#define iChannel2 texture2
#define iChannel3 texture3
uniform vec3 iChannelResolution[4];
{user_source}
out vec4 out_Color;
void main() {{
    vec4 color;
    mainImage(color, gl_FragCoord.xy);
    out_Color = color;
}}
"""


def set_channel_resolution(program, textures):
    """Upload Shadertoy's iChannelResolution[4] from the bound textures.

    Defensive like the other uniforms: a program that never reads it has it
    optimized out. The Apple GL driver reports array_length as the highest
    referenced element index + 1, so upload exactly that many rows (packed
    arrays cannot skip elements).
    """
    if "iChannelResolution" not in program:
        return
    uniform = program["iChannelResolution"]
    length = uniform.array_length
    if length <= 0:
        return
    rows = [
        (float(tex.width), float(tex.height), 1.0)
        for tex in textures[:length]
        if tex is not None
    ]
    rows.extend([(0.0, 0.0, 1.0)] * (length - len(rows)))
    if length == 1:
        uniform.value = rows[0]
    else:
        uniform.value = rows


def list_shader_files(primary_path):
    """Return sorted, de-duplicated shader entries usable as TUI choices.

    Mirrors list_lut_files: looks next to the configured shader, in the
    current directory, and in shaders/. Entries are single .glsl files or
    multibuffer pack folders named <name>.glsl (Shadertoy exports with
    Common/Buffer/Image passes).
    """
    folders = []
    if primary_path:
        folders.append(os.path.dirname(os.path.abspath(primary_path)))
    folders.append(os.getcwd())
    folders.append(os.path.join(os.getcwd(), "shaders"))
    found = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, "*.glsl")):
            found.setdefault(os.path.abspath(path), None)
    return sorted(found)


def load_shader(path):
    """Read a .glsl fragment shader; None when the file is unusable."""
    if not path:
        return None
    try:
        with open(path) as f:
            source = f.read()
    except OSError as exc:
        print(f"warning: shader file not readable: {path} ({exc})")
        return None
    if "void mainImage" not in source:
        print(f"warning: shader {path} has no mainImage(); ignoring")
        return None
    return source


class MultiShader:
    """A Shadertoy multibuffer export: a <name>.glsl folder of passes.

    The export format stores pass wiring outside the files, so this uses a
    fixed convention plus per-pass directives. Without directives the
    standard layout (Common.glsl, Buffer A/B/... .glsl, Image.glsl) works
    like the common export: Common is prepended to every pass, buffers run
    in file-name order, the first buffer's iChannel0 is the camera frame
    and each following pass reads the previous pass output, every buffer's
    iChannel1 is its own previous frame (ping-pong feedback, so temporal
    denoising like Lucas-Kanade's Buffer A works), and Image.glsl reads the
    last buffer on iChannel0. Any channel a pass samples but that has no
    directive defaults to the noise texture (unbound samplers read garbage
    on macOS). Exports with other wiring (say Image.glsl reading Buffer B
    on iChannel2) add a directive comment to the pass file, e.g.

        // hyper-stage: 2=Buffer B

    Names: camera, noise, font, feedback (the pass's own previous frame),
    or a buffer pass name like "Buffer B" (its latest output, which may be
    a previous frame when read before that buffer runs).

    Unlike single-file shaders (BGR in/out, matching the camera), a pack is
    a verbatim Shadertoy import, so the camera frame is converted to RGB
    before the first pass and the final render back to BGR.
    """

    # Directive names that resolve to auxiliary textures or feedback.
    AUX_NAMES = ("camera", "noise", "font", "feedback")

    def __init__(self, ctx, quad_buffer, folder, start, aux_texture):
        self._ctx = ctx
        self._folder = folder
        self._start = start
        self._buffer = quad_buffer
        self._aux_texture = aux_texture
        self._camera = None
        # Per buffer pass: [ping texture, pong texture] and matching FBOs;
        # rendering goes to the front entry, which then rotates to the back
        # so the next frame reads it as feedback.
        self._textures = []
        self._fbos = []
        self._passes = []  # (program, quad VAO, channels) per buffer pass
        self._final = None  # (program, quad VAO, channels) for Image.glsl
        self._size = (0, 0)

    @staticmethod
    def parse_channels(source, buffer_names):
        """Map unit number -> channel name from // hyper-stage: directives.

        Accepts the aux names (camera/noise/font/feedback) and references
        to a buffer pass file like "Buffer B" (spaces optional, case
        ignored). Unknown names are ignored, leaving the default wiring.
        """
        channels = {}
        for line in source.splitlines():
            marker = line.find("hyper-stage:")
            if marker < 0:
                continue
            matches = re.findall(
                r"(\d+)\s*=\s*(\w+(?:\s+\w+)*?)(?=\s+\d+\s*=|$)",
                line[marker:],
            )
            for unit, name in matches:
                key = name.strip().lower().replace(" ", "")
                if key in MultiShader.AUX_NAMES:
                    channels[int(unit)] = key
                else:
                    for i, buf in enumerate(buffer_names):
                        stem = buf[: -len(".glsl")].lower().replace(" ", "")
                        if key == stem:
                            channels[int(unit)] = ("buffer", i)
                            break
        return channels

    def close(self):
        """Release everything, including the compiled pass programs."""
        self._release_targets()
        for entry in self._passes:
            self._release_pass(entry)
        if self._final is not None:
            self._release_pass(self._final)
        self._passes = []
        self._final = None
        self._size = (0, 0)

    @staticmethod
    def _release_pass(entry):
        program, quad = entry[0], entry[1]
        if quad is not None:
            quad.release()
        program.release()

    def _release_targets(self):
        """Release render targets; programs and quads stay usable."""
        for texs in self._textures:
            for tex in texs:
                tex.release()
        for fbos in self._fbos:
            for fbo in fbos:
                fbo.release()
        self._textures = []
        self._fbos = []
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def _compile(self, source):
        program = self._ctx.program(
            vertex_shader=SHADER_VERTEX_SOURCE,
            fragment_shader=SHADER_FRAGMENT_TEMPLATE.format(user_source=source),
        )
        # The quad VAO must belong to the program it renders with.
        quad = self._ctx.vertex_array(program, [(self._buffer, "2f", "in_pos")])
        return program, quad

    def _set_clock(self, program):
        # Defensive upload: unused uniforms are optimized out of a program.
        if "iResolution" in program:
            program["iResolution"].value = (
                float(self._size[0]),
                float(self._size[1]),
            )
        if "iTime" in program:
            program["iTime"].value = time.monotonic() - self._start

    def _default_channel(self, pass_index, unit):
        """The fixed convention for a pass channel without a directive."""
        if pass_index == len(self._passes):
            # Image.glsl: read the last buffer on iChannel0.
            return ("buffer", len(self._passes) - 1) if unit == 0 else "noise"
        if unit == 0:
            # First buffer reads the camera, the rest the previous pass.
            return "camera" if pass_index == 0 else ("buffer", pass_index - 1)
        if unit == 1:
            return ("feedback", pass_index)
        return "noise"

    def _resolve_channel(self, name, pass_index):
        """Texture for a channel name at pass runtime; None keeps it unbound."""
        if name == "camera":
            return self._camera
        if name in ("noise", "font"):
            return self._aux_texture(name)
        if isinstance(name, tuple):
            kind, index = name
            if kind == "feedback":
                # At render time the back entry is the previous frame; the
                # pass about to run has not rotated yet.
                return self._textures[index][1]
            if kind == "buffer":
                # That buffer's latest output (after its rotation this
                # frame, or its previous frame if it has not run yet).
                return self._textures[index][1]
        return None

    def _bind_channels(self, program, channels, pass_index):
        """Bind all four samplers, defaults included, for one pass."""
        bound = []
        for unit in range(4):
            name = channels.get(
                unit, self._default_channel(pass_index, unit)
            )
            tex = self._resolve_channel(name, pass_index)
            if tex is None:
                # Unknown/unresolvable names (or an aux that could not be
                # created) fall back to noise; unbound reads garbage.
                tex = self._aux_texture("noise")
            bound.append(tex)
            tex.use(unit)
            sampler = (
                program[f"texture{unit}"]
                if f"texture{unit}" in program
                else None
            )
            if sampler is not None:
                sampler.value = unit
        set_channel_resolution(program, bound)

    def build(self):
        """Compile every pass; None (after a warning) on any failure."""
        try:
            names = sorted(os.listdir(self._folder))
        except OSError as exc:
            print(f"warning: shader pack {self._folder} unreadable: {exc}")
            return None
        names = [n for n in names if n.endswith(".glsl")]
        common = ""
        if "Common.glsl" in names:
            with open(os.path.join(self._folder, "Common.glsl")) as f:
                common = f.read() + "\n"
        buffers = [n for n in names if n not in ("Common.glsl", "Image.glsl")]
        if "Image.glsl" not in names or not buffers:
            print(
                f"warning: shader pack {self._folder} needs buffer passes "
                "and Image.glsl"
            )
            return None
        try:
            for name in buffers:
                with open(os.path.join(self._folder, name)) as f:
                    source = common + f.read()
                channels = self.parse_channels(source, buffers)
                self._passes.append(
                    (*self._compile(source), channels)
                )
            with open(os.path.join(self._folder, "Image.glsl")) as f:
                source = common + f.read()
            channels = self.parse_channels(source, buffers)
            self._final = (*self._compile(source), channels)
        except Exception as exc:
            print(f"warning: shader pack {self._folder} failed: {exc}")
            self.close()
            return None
        return self

    def _ensure_size(self, width, height):
        if self._size == (width, height):
            return True
        self._release_targets()
        try:
            self._camera = self._ctx.texture((width, height), 3, dtype="f1")
            # NEAREST, not LINEAR: imported shaders sample at uv = k/N,
            # which lands exactly on texel boundaries; linear would average
            # a 2x2 texel block per sample (a blur that shifts luma across
            # dither thresholds - 1bit.glsl turned speckled). Nearest keeps
            # the same-resolution mapping a crisp texel-per-pixel copy,
            # which is what Shadertoy's larger webcam texture effectively
            # gives those uv's too.
            self._camera.filter = (moderngl.NEAREST, moderngl.NEAREST)
            for _ in self._passes:
                # Two ping-pong targets per buffer: the framebuffer wraps the
                # texture, so what is rendered is exactly what the next pass
                # (and the next frame's feedback) samples. Buffers are float
                # (RGBA16F) like Shadertoy's, so passes can store negative or
                # fractional data (Lucas-Kanade's motion vectors) between
                # passes and frames.
                texs = [
                    self._ctx.texture((width, height), 4, dtype="f2")
                    for _ in range(2)
                ]
                for tex in texs:
                    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                    tex.repeat_x = False
                    tex.repeat_y = False
                self._textures.append(texs)
                self._fbos.append([self._ctx.framebuffer(tex) for tex in texs])
        except Exception as exc:
            print(f"warning: shader pack {self._folder} size error: {exc}")
            self._release_targets()
            return False
        self._size = (width, height)
        return True

    def run(self, output_fbo, frame_bgr):
        """Process one BGR camera frame through the pass chain."""
        height, width = frame_bgr.shape[:2]
        if not self._ensure_size(width, height):
            return frame_bgr
        # Same boundary flips as the single-file path: upload flipped so the
        # camera texture reads like a Shadertoy asset (t=0 at image bottom),
        # final read flipped back to top-down. Internal buffer passes render
        # and sample with the same uv mapping, so they need no flips.
        rgb = cv2.cvtColor(cv2.flip(frame_bgr, 0), cv2.COLOR_BGR2RGB)
        self._camera.write(rgb)
        for i, (program, quad, channels) in enumerate(self._passes):
            textures = self._textures[i]
            fbos = self._fbos[i]
            self._bind_channels(program, channels, i)
            self._set_clock(program)
            fbos[0].use()
            quad.render(moderngl.TRIANGLE_STRIP)
            # Rotate so the just-written texture becomes next frame's
            # feedback and the old feedback becomes the render target.
            textures.append(textures.pop(0))
            fbos.append(fbos.pop(0))
            # Shadertoy keeps mipmaps on its buffer textures; passes that
            # texelFetch an explicit LOD (Ben-Day-Dots downsampling) read
            # black from a level that does not exist without this.
            textures[1].build_mipmaps()
        program, quad, channels = self._final
        self._bind_channels(program, channels, len(self._passes))
        self._set_clock(program)
        output_fbo.use()
        quad.render(moderngl.TRIANGLE_STRIP)
        data = output_fbo.read(components=3, alignment=1)
        # Undo the bottom-up GL readback (see the upload flip above).
        rgb_out = cv2.flip(
            np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), 0
        )
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)



class ShaderEngine:
    """Compiles and runs the active .glsl shader on the GPU.

    One moderngl standalone context lives inside the video daemon thread and
    is created lazily on first use, keeping GL affinity in one place. Programs
    are cached per file so switching shaders or re-selecting the same file
    never recompiles; a shader that fails to compile disables the effect
    (the live frame passes through) with a single warning. A multibuffer
    pack (a <name>.glsl folder) is compiled into a MultiShader instead.
    """

    VERTEX = SHADER_VERTEX_SOURCE

    # Auxiliary textures a shader can bind instead of the camera, selectable
    # per channel with a directive comment in the shader file, e.g.
    #   // hyper-stage: 0=font 1=camera
    # Channels not mentioned default to: 0=camera, others noise (unbound
    # samplers read garbage on macOS, never black).
    AUX_KEYS = ("camera", "noise", "font")

    def __init__(self):
        self._ctx = None
        self._buffer = None
        self._fbo = None
        self._texture = None
        self._size = (0, 0)
        self._start = time.monotonic()
        self._aux = {}  # name -> texture, created lazily per context
        # path -> (program, quad VAO) / MultiShader, or None (broken)
        self._programs = {}
        self._broken = set()

    @staticmethod
    def parse_channels(source):
        """Map unit number -> aux name from // hyper-stage: directives."""
        channels = {}
        for line in source.splitlines():
            marker = line.find("hyper-stage:")
            if marker < 0:
                continue
            for unit, name in re.findall(
                r"(\d+)\s*=\s*(\w+)", line[marker:]
            ):
                if name in ShaderEngine.AUX_KEYS:
                    channels[int(unit)] = name
        return channels

    def _aux_texture(self, name):
        """Return the named auxiliary texture, creating it on first use."""
        if name in self._aux:
            return self._aux[name]
        if name == "noise":
            rng = np.random.default_rng(1234)
            data = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
            tex = self._ctx.texture((256, 256), 4, data.tobytes())
            # Noise lookups often go outside [0,1] on purpose.
            tex.repeat_x = True
            tex.repeat_y = True
        elif name == "font":
            # A 16x16 grid of ASCII glyphs standing in for Shadertoy's font
            # texture (built with OpenCV's vector font, not pixel-perfect).
            cell = 16
            atlas = np.zeros((256, 256, 4), dtype=np.uint8)
            for code in range(32, 127):
                x = (code % 16) * cell
                y = (code // 16) * cell
                glyph = np.zeros((cell, cell), dtype=np.uint8)
                cv2.putText(
                    glyph, chr(code), (1, cell - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA,
                )
                atlas[y : y + cell, x : x + cell, 0] = glyph
                # Approximate the SDF-ish alpha some shaders read for
                # outlines: a slightly dilated glyph.
                atlas[y : y + cell, x : x + cell, 3] = cv2.dilate(
                    glyph, np.ones((3, 3), np.uint8)
                )
            tex = self._ctx.texture((256, 256), 4, atlas.tobytes())
        else:
            return None
        self._aux[name] = tex
        return tex

    def _ensure_size(self, width, height):
        """Create the GL objects for a frame size; recreate if it changed."""
        if self._ctx is None:
            try:
                self._ctx = moderngl.create_standalone_context()
            except Exception as exc:
                print(f"warning: GPU shader context unavailable ({exc})")
                return False
        if self._size != (width, height):
            self._buffer = self._ctx.buffer(
                np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32).tobytes()
            )
            self._texture = self._ctx.texture((width, height), 3, dtype="f1")
            # NEAREST, not LINEAR: imported shaders sample at uv = k/N,
            # which lands exactly on texel boundaries; linear would average
            # a 2x2 texel block per sample (a blur that shifts luma across
            # dither thresholds - 1bit.glsl turned speckled). Nearest keeps
            # the same-resolution mapping a crisp texel-per-pixel copy.
            self._texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._texture.repeat_x = False
            self._texture.repeat_y = False
            self._fbo = self._ctx.simple_framebuffer((width, height), 3)
            self._size = (width, height)
            # The compiled VAOs reference the old-size-independent buffer, so
            # they stay valid; drop nothing.
        return True

    def apply(self, frame_bgr, path):
        """Run the shader on frame_bgr; return the result or the input frame."""
        height, width = frame_bgr.shape[:2]
        if not self._ensure_size(width, height):
            return frame_bgr
        if os.path.isdir(path):
            return self._apply_multi(frame_bgr, path)
        entry = self._programs.get(path, False)
        if entry is False:
            source = load_shader(path)
            entry = None
            if source is not None:
                try:
                    program = self._ctx.program(
                        vertex_shader=self.VERTEX,
                        fragment_shader=SHADER_FRAGMENT_TEMPLATE.format(
                            user_source=source
                        ),
                    )
                    # The quad VAO must belong to the program it renders with.
                    entry = (
                        program,
                        self._ctx.vertex_array(
                            program, [(self._buffer, "2f", "in_pos")]
                        ),
                        self.parse_channels(source),
                    )
                except Exception as exc:
                    if path not in self._broken:
                        self._broken.add(path)
                        print(f"warning: shader {path} failed to compile: {exc}")
            self._programs[path] = entry
        if entry is None:
            return frame_bgr
        program, quad, channels = entry

        # Verbatim Shadertoy import: the camera frame goes in as RGB (BGR
        # -> RGB on upload, back on readback), so shaders hardcoding RGB
        # palette colors (1bit.glsl, smiley.glsl) render their intended
        # hues. Pass-through shaders are unaffected - both swaps cancel.
        rgb_in = cv2.cvtColor(cv2.flip(frame_bgr, 0), cv2.COLOR_BGR2RGB)
        self._texture.write(rgb_in)
        bound = []
        for unit in range(4):
            name = channels.get(unit, "camera" if unit == 0 else "noise")
            # The camera texture on the requested unit; auxiliary stand-ins
            # (noise/font) for decorative channels. Unbound samplers read
            # garbage on macOS, so every declared unit gets a real texture.
            tex = self._texture if name == "camera" else self._aux_texture(name)
            bound.append(tex)
            if tex is not None:
                tex.use(unit)
            sampler = program[f"texture{unit}"] if f"texture{unit}" in program else None
            if sampler is not None:
                sampler.value = unit
        set_channel_resolution(program, bound)
        if "iResolution" in program:
            program["iResolution"].value = (float(width), float(height))
        if "iTime" in program:
            # Seconds since the engine started; shader effects use it as
            # their animation clock (monotonic, no wall-clock jumps).
            program["iTime"].value = time.monotonic() - self._start
        self._fbo.use()
        quad.render(moderngl.TRIANGLE_STRIP)
        data = self._fbo.read(components=3, alignment=1)
        # Undo the bottom-up GL readback and the BGR->RGB upload swap (see
        # the upload comment above).
        return cv2.cvtColor(
            cv2.flip(
                np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3), 0
            ),
            cv2.COLOR_RGB2BGR,
        )

    def _apply_multi(self, frame_bgr, path):
        """Run a multibuffer pack folder; input frame passes through if broken."""
        entry = self._programs.get(path, False)
        if entry is False:
            entry = MultiShader(
                self._ctx, self._buffer, path, self._start, self._aux_texture
            ).build()
            if entry is None:
                self._broken.add(path)
            self._programs[path] = entry
        if entry is None:
            return frame_bgr
        return entry.run(self._fbo, frame_bgr)

    def reload(self):
        """Drop cached programs so the next frame recompiles from disk."""
        for path, entry in list(self._programs.items()):
            if isinstance(entry, MultiShader):
                entry.close()
            elif entry:
                entry[1].release()
                entry[0].release()
            self._programs.pop(path, None)
        self._broken.clear()

    def release(self):
        """Release every GL object; must run on the thread that created them."""
        for entry in self._programs.values():
            if isinstance(entry, MultiShader):
                entry.close()
            elif entry:
                entry[1].release()  # quad VAO
                entry[0].release()  # program
        self._programs.clear()
        self._aux.clear()
        self._buffer = self._fbo = self._texture = None
        if self._ctx is not None:
            self._ctx.release()
            self._ctx = None
        self._size = (0, 0)
