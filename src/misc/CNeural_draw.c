//
// Created by ledai on 11/01/2025.
//

#define CLAY_IMPLEMENTATION
#include "clay-main/clay.h"
#include "clay-main/renderers/raylib/clay_renderer_raylib.c"
#include <stdlib.h>

int CNeural_draw(void);

int main() {
    CNeural_draw();
}

int CNeural_draw(void) {
    Clay_Raylib_Initialize(FLAG_WINDOW_RESIZABLE);

    uint64_t clayRequiredMemory = Clay_MinMemorySize();
    Clay_Arena clayMemory = (Clay_Arena) {
        .memory = malloc(clayRequiredMemory),
        .capacity = clayRequiredMemory
    };
    Clay_Initialize(clayMemory, (Clay_Dimensions) {
        .width = GetScreenWidth(),
        .height = GetScreenHeight()
    });
    return 0;
}
