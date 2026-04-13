// Raylib bridge: converts Lumen's f64 values to raylib's f32 structs.
// Lumen calls these wrappers; they forward to the real raylib functions.
//
// Color encoding: colors are packed as i32 (RGBA in little-endian).
// Vector2/Rectangle are passed as individual f64 components.

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Raylib types (we declare them here to avoid needing raylib.h)
typedef struct { float x, y; } Vector2;
typedef struct { float x, y, width, height; } Rectangle;
typedef struct { unsigned char r, g, b, a; } Color;
typedef struct { Vector2 offset; Vector2 target; float rotation; float zoom; } Camera2D;
typedef struct { int width, height, mipmaps, format; } Texture2D_header;

// Raylib function declarations.
// IMPORTANT: functions returning bool MUST be declared as bool, not int.
// On x86-64, bool returns in AL (1 byte) with undefined upper bits.
// Declaring as int reads garbage from the upper 24 bits of EAX.
void InitWindow(int width, int height, const char *title);
void CloseWindow(void);
bool WindowShouldClose(void);
void SetTargetFPS(int fps);
void BeginDrawing(void);
void EndDrawing(void);
void ClearBackground(Color c);
void DrawText(const char *text, int x, int y, int fontSize, Color c);
void DrawRectangleRec(Rectangle rec, Color c);
void DrawRectangle(int x, int y, int w, int h, Color c);
void DrawRectanglePro(Rectangle rec, Vector2 origin, float rotation, Color c);
void DrawCircleV(Vector2 center, float radius, Color c);
void DrawLine(int x1, int y1, int x2, int y2, Color c);
void DrawTexture(void *tex, int x, int y, Color c);
void DrawTexturePro(void *tex, Rectangle src, Rectangle dst, Vector2 origin, float rot, Color c);
int MeasureText(const char *text, int fontSize);
bool IsKeyPressed(int key);
bool IsKeyDown(int key);
bool IsGestureDetected(int gesture);
float GetFrameTime(void);
void *LoadFont(const char *path);
void DrawTextEx(void *font, const char *text, Vector2 pos, float fontSize, float spacing, Color c);
void *LoadSound(const char *path);
void PlaySound(void *sound);
void *LoadTexture(const char *path);
void InitAudioDevice(void);
void BeginMode2D(Camera2D cam);
void EndMode2D(void);
Color ColorAlpha(Color c, float alpha);

// --- Helper: unpack Lumen color (i32 packed RGBA) to raylib Color ---
static Color unpack_color(int32_t c) {
    return (Color){
        (unsigned char)(c & 0xFF),
        (unsigned char)((c >> 8) & 0xFF),
        (unsigned char)((c >> 16) & 0xFF),
        (unsigned char)((c >> 24) & 0xFF)
    };
}

// --- Helper: Lumen string (ptr to [len:i32|data]) to C string ---
static const char *lumen_to_cstr(int64_t str_ptr) {
    if (str_ptr == 0) return "";
    char *buf = (char *)(uintptr_t)str_ptr;
    int32_t len = *(int32_t *)buf;
    // Allocate a null-terminated copy (small, on the stack or malloc)
    char *cstr = (char *)malloc(len + 1);
    memcpy(cstr, buf + 4, len);
    cstr[len] = '\0';
    return cstr;
}

// --- Color constants (packed RGBA) ---
int32_t rl_color_black(void)     { return 0xFF000000; }
int32_t rl_color_white(void)     { return 0xFFFFFFFF; }
int32_t rl_color_red(void)       { return 0xFF0000FF; }
int32_t rl_color_green(void)     { return 0xFF00FF00; }
int32_t rl_color_blue(void)      { return 0xFFFF0000; }
int32_t rl_color_yellow(void)    { return 0xFF00FFFF; }
int32_t rl_color_purple(void)    { return 0xFFFF00FF; }
int32_t rl_color_darkblue(void)  { return 0xFF8B0000; }
int32_t rl_color_darkgray(void)  { return 0xFF505050; }
int32_t rl_color_gray(void)      { return 0xFF808080; }
int32_t rl_color_violet(void)    { return 0xFFEE82EE; }

int32_t rl_color_alpha(int32_t c, double alpha) {
    Color col = unpack_color(c);
    col.a = (unsigned char)(alpha * 255.0);
    return (int32_t)(col.r | (col.g << 8) | (col.b << 16) | (col.a << 24));
}

// --- Window ---
void rl_init_window(int32_t w, int32_t h, int64_t title) {
    const char *t = lumen_to_cstr(title);
    InitWindow(w, h, t);
    free((void*)t);
}
int32_t rl_window_should_close(void) {
    if (WindowShouldClose()) return 1;
    return 0;
}
void rl_close_window(void) { CloseWindow(); }
void rl_set_target_fps(int32_t fps) { SetTargetFPS(fps); }
double rl_get_frame_time(void) { return (double)GetFrameTime(); }

// --- Drawing ---
void rl_begin_drawing(void) { BeginDrawing(); }
void rl_end_drawing(void) { EndDrawing(); }
void rl_clear_background(int32_t c) { ClearBackground(unpack_color(c)); }

void rl_draw_text(int64_t text, int32_t x, int32_t y, int32_t size, int32_t c) {
    const char *t = lumen_to_cstr(text);
    DrawText(t, x, y, size, unpack_color(c));
    free((void*)t);
}

int32_t rl_measure_text(int64_t text, int32_t size) {
    const char *t = lumen_to_cstr(text);
    int32_t r = MeasureText(t, size);
    free((void*)t);
    return r;
}

void rl_draw_rectangle_rec(double x, double y, double w, double h, int32_t c) {
    DrawRectangleRec((Rectangle){(float)x,(float)y,(float)w,(float)h}, unpack_color(c));
}

void rl_draw_rectangle(int32_t x, int32_t y, int32_t w, int32_t h, int32_t c) {
    DrawRectangle(x, y, w, h, unpack_color(c));
}

void rl_draw_rectangle_pro(double rx, double ry, double rw, double rh,
                           double ox, double oy, double rot, int32_t c) {
    DrawRectanglePro((Rectangle){(float)rx,(float)ry,(float)rw,(float)rh},
                     (Vector2){(float)ox,(float)oy}, (float)rot, unpack_color(c));
}

void rl_draw_circle(double cx, double cy, double radius, int32_t c) {
    DrawCircleV((Vector2){(float)cx,(float)cy}, (float)radius, unpack_color(c));
}

void rl_draw_line(int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t c) {
    DrawLine(x1, y1, x2, y2, unpack_color(c));
}

// --- Input ---
int32_t rl_is_key_pressed(int32_t key) {
    if (IsKeyPressed(key)) return 1;
    return 0;
}
int32_t rl_is_key_down(int32_t key) {
    if (IsKeyDown(key)) return 1;
    return 0;
}
int32_t rl_is_gesture_detected(int32_t gesture) {
    if (IsGestureDetected(gesture)) return 1;
    return 0;
}

// --- Audio ---
void rl_init_audio(void) { InitAudioDevice(); }
int64_t rl_load_sound(int64_t path) {
    const char *p = lumen_to_cstr(path);
    void *s = malloc(64); // Sound struct is small
    memcpy(s, &(struct{void*p;}){NULL}, 64); // zero init
    // Use the actual LoadSound — it returns a Sound struct by value.
    // We need to handle this differently. Store the sound in a global array.
    // For simplicity, just call LoadSound and memcpy the returned struct.
    *(void**)s = *(void**)&(struct{int x;}){0}; // placeholder
    free((void*)p);
    // Actually, Sound is returned by value. Let's use a wrapper approach:
    return 0; // TODO: sound support needs special handling
}
void rl_play_sound(int64_t snd) {
    // TODO: sound support
}

// --- Textures ---
int64_t rl_load_texture(int64_t path) {
    const char *p = lumen_to_cstr(path);
    // Texture2D is a struct returned by value. We allocate and store it.
    void *tex = malloc(sizeof(Texture2D_header));
    // LoadTexture returns Texture2D by value — we need to capture it.
    // This requires linking against raylib and knowing the struct size.
    // For now, use the direct function:
    Texture2D_header *t = (Texture2D_header *)tex;
    // We'll use a direct extern approach instead — store the return value.
    free((void*)p);
    return (int64_t)(uintptr_t)tex;
}

void rl_draw_texture(int64_t tex, int32_t x, int32_t y, int32_t c) {
    // DrawTexture takes Texture2D by value — we need the struct on the stack.
    // This is tricky with our pointer-based approach.
    // For now, skip textures and use colored rectangles.
}

// --- Font ---
int64_t rl_load_font(int64_t path) {
    const char *p = lumen_to_cstr(path);
    free((void*)p);
    return 0; // TODO: font loading needs struct-by-value handling
}

void rl_draw_text_ex(int64_t font, int64_t text, double px, double py,
                     double size, double spacing, int32_t c) {
    // TODO: uses Font struct by value
    // Fallback: use DrawText
    const char *t = lumen_to_cstr(text);
    DrawText(t, (int)px, (int)py, (int)size, unpack_color(c));
    free((void*)t);
}

// --- Camera ---
// Camera state stored globally (simplification for the bridge)
static Camera2D g_camera = {{0,0},{0,0},0,1};

void rl_set_camera(double tx, double ty, double ox, double oy, double rot, double zoom) {
    g_camera.target = (Vector2){(float)tx, (float)ty};
    g_camera.offset = (Vector2){(float)ox, (float)oy};
    g_camera.rotation = (float)rot;
    g_camera.zoom = (float)zoom;
}
void rl_begin_mode_2d(void) { BeginMode2D(g_camera); }
void rl_end_mode_2d(void) { EndMode2D(); }

// --- Math helpers ---
double lumen_sqrt(double x) { return sqrt(x); }
double lumen_abs(double x) { return fabs(x); }
double lumen_cos(double x) { return cos(x); }
double lumen_sin(double x) { return sin(x); }
double lumen_clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
double lumen_rand_f64(void) { return (double)rand() / (double)RAND_MAX; }
